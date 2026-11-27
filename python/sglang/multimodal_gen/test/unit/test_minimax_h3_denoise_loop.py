# SPDX-License-Identifier: Apache-2.0
"""Numerical contract for request-static H3 denoise metadata."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.multimodal_gen.configs.models.dits.minimax_h3 import (
    MINIMAX_H3_ADALN_MODALITY_NUM,
)
from sglang.multimodal_gen.runtime.managers.forward_context import get_forward_context
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.denoise_loop import (
    MiniMaxH3DenoiseBranch,
    _build_local_embedding_layout,
    _minimax_h3_update_target_rows_,
    minimax_h3_denoise_loop,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_sequence import (
    minimax_h3_packed_sequence,
    minimax_h3_packed_sequence_ref2va_blocks,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages import (
    denoising,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.denoising import (
    MiniMaxH3DenoisingStage,
    _build_cube_attn_metadata,
    _precompute_refined_prompt_embeds,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.time_request import (
    minimax_h3_time_shift_sigmas,
)


def _minimax_h3_rf_v_to_x0(
    xt: torch.Tensor,
    v: torch.Tensor,
    timestep: torch.Tensor,
) -> torch.Tensor:
    cond_t = timestep.to(device=xt.device, dtype=xt.dtype)
    while cond_t.ndim < xt.ndim:
        cond_t = cond_t.unsqueeze(-1)
    sigma_t = 1 - cond_t
    return xt + sigma_t * v


def _minimax_h3_euler_eta0_step(
    state: torch.Tensor,
    denoised: torch.Tensor,
    *,
    sigma_curr: float,
    sigma_ratio: torch.Tensor,
) -> torch.Tensor:
    if sigma_curr == 0.0:
        return state
    compute_dtype = torch.float32
    if state.dtype not in (torch.float16, torch.bfloat16):
        compute_dtype = state.dtype
    ratio = sigma_ratio.to(device=state.device, dtype=compute_dtype)
    out = ratio * state.to(dtype=compute_dtype) + (1.0 - ratio) * denoised.to(
        dtype=compute_dtype
    )
    return out.to(dtype=state.dtype)


def _branch(
    mode: str, token_tags: torch.Tensor | None = None
) -> MiniMaxH3DenoiseBranch:
    common = dict(text_len=3, latent_t=2, latent_h=4, latent_w=4, audio_t=3)
    if mode == "t2va":
        packed = minimax_h3_packed_sequence(
            **common,
            include_keyframe_cond=False,
        )
    elif mode == "fl2va":
        packed = minimax_h3_packed_sequence(
            **common,
            include_keyframe_cond=True,
            keyframe_frame_indices=[0, -1],
            frame_count=5,
        )
    else:
        packed = minimax_h3_packed_sequence_ref2va_blocks(
            **common,
            ref_blocks=[
                {"kind": "image", "latent_h": 4, "latent_w": 4},
                {"kind": "audio", "ref_audio_t": 2},
            ],
        )
    return MiniMaxH3DenoiseBranch(
        packed=packed,
        text_embeddings=torch.zeros(3, 5120),
        token_tags=packed["token_tags"] if token_tags is None else token_tags,
        device=torch.device("cpu"),
    )


def test_full_loop_accepts_out_of_tree_platform():
    """Accept OOT platforms without loading a model or accessing hardware."""

    class DeviceLookupReached(Exception):
        pass

    platform = Mock()
    for kind in ("cuda", "hip", "cpu", "mps", "npu", "xpu"):
        getattr(platform, f"is_{kind}").return_value = False
    platform.is_out_of_tree.return_value = True
    platform.get_local_torch_device.side_effect = DeviceLookupReached
    stage = MiniMaxH3DenoisingStage.__new__(MiniMaxH3DenoisingStage)

    with (
        patch.object(denoising, "current_platform", platform),
        patch.object(denoising, "_resolve_full_loop_context"),
        pytest.raises(DeviceLookupReached),
    ):
        stage._run_full_loop(None, None)
    platform.get_local_torch_device.assert_called_once_with()


def test_time_shift_sigmas_build_exact_denoise_step_count():
    for num_steps in (1, 2, 10, 50):
        for shift_scale in (3.0, 12.0):
            sigmas = minimax_h3_time_shift_sigmas(
                num_steps=num_steps,
                shift_scale=shift_scale,
            )

            assert len(sigmas) == num_steps + 1
            assert sigmas[0] == 1.0
            assert sigmas[-1] == 0.0
            assert all(
                current > following for current, following in zip(sigmas, sigmas[1:])
            )
            # Independent closed form for N Euler intervals, including sigma=0.
            expected = [
                shift_scale
                * (num_steps - i)
                / (num_steps + (shift_scale - 1) * (num_steps - i))
                for i in range(num_steps + 1)
            ]
            torch.testing.assert_close(
                torch.tensor(sigmas), torch.tensor(expected), rtol=0, atol=2e-7
            )


def test_precomputed_timestep_plan_matches_full_unique_reference():
    """Preplanning must preserve fp32 collisions and every packed row class."""

    for mode in ("t2va", "fl2va", "ref2va"):
        branch = _branch(mode)
        assert branch.static_kwargs["skip_mask_out_condition"]
        assert "token_tags" not in branch.static_kwargs
        assert not bool((branch.static_kwargs["block_token_tags"] < 0).any())
        torch.testing.assert_close(
            branch.static_kwargs["img_pos_for_infer_output_info"]["position_ids"],
            branch.img_pos_dev[branch.update_mask_dev],
            rtol=0,
            atol=0,
        )
        video_steps = [0.75, 0.1]
        audio_steps = [0.625, 0.2]
        plan = branch.prepare_timestep_plan(
            video_timesteps=video_steps,
            audio_timesteps=audio_steps,
            imgvid_cond_noise_aug=0.6,
            audio_ref_cond_noise_aug=0.4,
        )

        assert branch.static_kwargs["packed_seq_params"]["cu_seqlens_q_host"] == tuple(
            int(value)
            for value in branch.static_kwargs["packed_seq_params"][
                "cu_seqlens_q"
            ].tolist()
        )
        assert branch.static_kwargs["refiner_packed_seq_params"][
            "cu_seqlens_q_host"
        ] == (0, 3, 3)

        for step, (video_t, audio_t) in enumerate(
            zip(video_steps, audio_steps, strict=True)
        ):
            reference = torch.full((branch.seq_len,), video_t, dtype=torch.float32)
            reference[branch.img_cond_seq_idx] = max(video_t, 0.6)
            reference[branch.audio_target_seq_idx] = audio_t
            reference[branch.audio_ref_seq_idx] = max(audio_t, 0.4)
            expected = torch.unique(reference, sorted=True, return_inverse=True)
            assert plan[step].fp32_bits == tuple(expected[0].view(torch.int32).tolist())
            torch.testing.assert_close(plan[step][0], expected[0], rtol=0, atol=0)
            torch.testing.assert_close(plan[step][1], expected[1], rtol=0, atol=0)
            torch.testing.assert_close(
                plan[step][2],
                branch.static_kwargs["block_token_tags"]
                + expected[1] * MINIMAX_H3_ADALN_MODALITY_NUM,
                rtol=0,
                atol=0,
            )

        repeated_plan = branch.prepare_timestep_plan(
            video_timesteps=[0.0, 0.1, 0.2],
            audio_timesteps=[0.0, 0.2, 0.4],
            imgvid_cond_noise_aug=0.999,
            audio_ref_cond_noise_aug=1.0,
        )
        assert repeated_plan[1][1] is repeated_plan[2][1]
        assert repeated_plan[1][2] is repeated_plan[2][2]


def test_inplace_target_update_matches_scheduler_math():
    generator = torch.Generator().manual_seed(7)
    for sigma_curr, sigma_next in ((1.0, 0.7), (0.2, 0.0), (0.0, 0.0)):
        state = torch.randn(11, 32, generator=generator)
        velocity = torch.randn(11, 32, generator=generator)
        timestep = torch.tensor(1.0 - sigma_curr)
        ratio = torch.tensor(0.0 if sigma_curr == 0.0 else sigma_next / sigma_curr)
        denoised = _minimax_h3_rf_v_to_x0(state, velocity, timestep)
        expected = _minimax_h3_euler_eta0_step(
            state,
            denoised,
            sigma_curr=sigma_curr,
            sigma_ratio=ratio,
        )

        actual = state.clone()
        velocity_scratch = velocity.clone()
        _minimax_h3_update_target_rows_(
            actual,
            velocity_scratch,
            sigma_t=1.0 - timestep,
            sigma_curr=sigma_curr,
            sigma_ratio=ratio,
            one_minus_sigma_ratio=1.0 - ratio,
            denoised_scratch=torch.empty_like(actual),
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_local_text_layout_is_a_contiguous_prefix_per_ulysses_rank():
    for mode in ("t2va", "fl2va", "ref2va"):
        branch = _branch(mode)
        text_len = int(branch.static_kwargs["prompt_embeds"].shape[0])
        for world_size in (1, 2, 4, 8):
            for rank in range(world_size):
                layout = _build_local_embedding_layout(
                    seq_len=branch.seq_len,
                    text_pos=torch.arange(text_len),
                    img_pos=branch.img_pos,
                    audio_pos=branch.audio_pos,
                    world_size=world_size,
                    rank=rank,
                    device=torch.device("cpu"),
                )
                start = int(layout["text_source_start"])
                stop = int(layout["text_source_stop"])
                row_start = rank * (branch.seq_len // world_size)
                expected = torch.nonzero(
                    (torch.arange(text_len) >= row_start)
                    & (
                        torch.arange(text_len)
                        < row_start + branch.seq_len // world_size
                    )
                ).view(-1)
                assert expected.tolist() == list(range(start, stop))


def test_rank_local_token_tags_match_reference_slice():
    for mode in ("t2va", "fl2va", "ref2va"):
        seq_len = _branch(mode).seq_len
        token_tags = torch.arange(seq_len, dtype=torch.long) - seq_len // 2
        for world_size in (1, 2, 4, 8):
            for rank in range(world_size):
                with patch(
                    "sglang.multimodal_gen.runtime.pipelines_core.stages."
                    "model_specific_stages.minimax_h3.denoise_loop.get_ulysses_ctx",
                    return_value=(world_size, rank),
                ):
                    branch = _branch(mode, token_tags=token_tags)
                local_rows = branch.seq_len // world_size
                expected = token_tags[
                    rank * local_rows : (rank + 1) * local_rows
                ].clamp(min=0)
                torch.testing.assert_close(
                    branch.static_kwargs["block_token_tags"], expected, rtol=0, atol=0
                )


def test_cube_metadata_builder_uses_packed_layout_and_validates_step_count():
    packed = minimax_h3_packed_sequence(
        text_len=3,
        latent_t=2,
        latent_h=8,
        latent_w=8,
        audio_t=3,
        include_keyframe_cond=False,
    )
    server_args = SimpleNamespace(
        attention_backend="cube_sparse_attn",
        component_attention_backends={},
        attention_backend_config={
            "local_cube_size": [4, 4, 4],
            "topk_ratio_list": [1.0, 0.5],
        },
    )

    metadata = _build_cube_attn_metadata(
        server_args,
        packed=packed,
        num_steps=2,
        device=torch.device("cpu"),
    )
    assert metadata.topk_ratio_list == [1.0, 0.5]
    assert metadata.precomputed.layout.cube_token_size == 64

    with pytest.raises(ValueError, match="denoise steps"):
        _build_cube_attn_metadata(
            server_args,
            packed=packed,
            num_steps=3,
            device=torch.device("cpu"),
        )


def test_cube_metadata_follows_transformer_backend_override():
    packed = minimax_h3_packed_sequence(
        text_len=3,
        latent_t=2,
        latent_h=8,
        latent_w=8,
        audio_t=3,
        include_keyframe_cond=False,
    )
    server_args = SimpleNamespace(
        attention_backend="fa",
        component_attention_backends={"transformer": "cube_sparse_attn"},
        attention_backend_config={
            "local_cube_size": [4, 4, 4],
            "topk_ratio_list": [0.5],
        },
    )
    assert (
        _build_cube_attn_metadata(
            server_args,
            packed=packed,
            num_steps=1,
            device=torch.device("cpu"),
        )
        is not None
    )

    server_args.attention_backend = "cube_sparse_attn"
    server_args.component_attention_backends["transformer"] = "fa"
    assert (
        _build_cube_attn_metadata(
            server_args,
            packed=packed,
            num_steps=1,
            device=torch.device("cpu"),
        )
        is None
    )


def test_cube_metadata_is_updated_per_step():
    branch = _branch("t2va")
    metadata = SimpleNamespace(current_timestep=-1, topk_ratio_list=[1.0, 0.25])
    seen = []

    def model_forward(_model, _kwargs, step):
        seen.append((step, metadata.current_timestep))
        return (
            torch.zeros(int(branch.update_mask.sum()), 96),
            torch.zeros(branch.audio_pos.numel(), 32),
        )

    minimax_h3_denoise_loop(
        model=SimpleNamespace(prepare_adaln_plans=lambda _: None),
        model_forward=model_forward,
        positive=branch,
        initial_video_rows=torch.zeros(branch.img_pos.numel(), 96),
        initial_audio_rows=torch.zeros(branch.audio_pos.numel(), 32),
        keyframe_cond_rows=None,
        sigmas_video=[1.0, 0.5, 0.0],
        sigmas_audio=[1.0, 0.5, 0.0],
        device=torch.device("cpu"),
        attn_metadata=metadata,
    )

    assert seen == [(0, 0), (1, 1)]


@pytest.mark.parametrize("precision", ["bf16", "fp8"])
@pytest.mark.parametrize("use_model_forward", [False, True])
def test_frozen_fa_metadata_preserves_step_context(precision, use_model_forward):
    from sglang.multimodal_gen.runtime.layers.attention.backends.ascend_fa import (
        AscendFAMetadataBuilder,
    )

    branch = _branch("t2va")
    config = {"precision": precision, "head_chunk_size": 4, "qk_rotation": True}
    if precision == "fp8":
        config["fp8_fa_mode"] = "C8V16_TILING512"
    metadata = AscendFAMetadataBuilder().build(attention_backend_config=config)
    seen = []

    class Model:
        def prepare_adaln_plans(self, _):
            return None

        def __call__(self, **_kwargs):
            context = get_forward_context()
            seen.append((context.current_timestep, context.attn_metadata))
            return (
                torch.zeros(int(branch.update_mask.sum()), 96),
                torch.zeros(branch.audio_pos.numel(), 32),
            )

    batch = SimpleNamespace()
    stage = MiniMaxH3DenoisingStage.__new__(MiniMaxH3DenoisingStage)

    def model_forward(model, kwargs, step):
        return stage._forward_dit(
            model, kwargs, step, batch=batch, attn_metadata=metadata
        )

    with patch.object(
        MiniMaxH3DenoisingStage, "_maybe_get_bcg_runner", return_value=None
    ):
        minimax_h3_denoise_loop(
            model=Model(),
            model_forward=model_forward if use_model_forward else None,
            positive=branch,
            initial_video_rows=torch.zeros(branch.img_pos.numel(), 96),
            initial_audio_rows=torch.zeros(branch.audio_pos.numel(), 32),
            keyframe_cond_rows=None,
            sigmas_video=[1.0, 0.5, 0.0],
            sigmas_audio=[1.0, 0.5, 0.0],
            device=torch.device("cpu"),
            attn_metadata=metadata,
        )

    assert [step for step, _ in seen] == [0, 1]
    assert all(actual is metadata for _, actual in seen)
    assert metadata == AscendFAMetadataBuilder().build(attention_backend_config=config)


def test_native_dit_forward_publishes_cube_metadata_in_forward_context():
    metadata = SimpleNamespace(current_timestep=0, topk_ratio_list=[0.5])
    batch = SimpleNamespace()

    def model(**_kwargs):
        context = get_forward_context()
        assert context.current_timestep == 0
        assert context.attn_metadata is metadata
        assert context.forward_batch is batch
        return torch.zeros(1, 96), torch.zeros(1, 32)

    stage = MiniMaxH3DenoisingStage.__new__(MiniMaxH3DenoisingStage)
    with patch.object(
        MiniMaxH3DenoisingStage,
        "_maybe_get_bcg_runner",
        return_value=None,
    ):
        video, audio = stage._forward_dit(
            model,
            {},
            0,
            batch=batch,
            attn_metadata=metadata,
        )
    assert video.shape == (1, 96)
    assert audio.shape == (1, 32)


def test_grouped_outputs_share_prompt_refinement():
    class Refiner:
        calls = 0

        def refine_prompt_embeds(self, prompt_embeds, refiner_cu, *, device):
            del refiner_cu
            self.calls += 1
            return torch.ones(
                prompt_embeds.shape[0], 5376, dtype=prompt_embeds.dtype, device=device
            )

    model = Refiner()
    conditioning = {}
    first, second = _branch("t2va"), _branch("t2va")

    for branch in (first, second):
        assert _precompute_refined_prompt_embeds(
            model,
            branch,
            device=torch.device("cpu"),
            shared_conditioning=conditioning,
        )

    assert model.calls == 1
    assert first.static_kwargs["prompt_embeds"] is second.static_kwargs["prompt_embeds"]


@pytest.mark.parametrize("steps,global_steps", [(4, 0), (4, 1), (8, 2)])
def test_windowed_loop_reuses_global_state_and_native_updates(steps, global_steps):
    from functools import partial

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.windowing import (
        prepare_windows,
    )

    device = torch.device("cpu")
    shape = (12, 4, 4)
    packed = minimax_h3_packed_sequence(
        text_len=3,
        latent_t=shape[0],
        latent_h=shape[1],
        latent_w=shape[2],
        audio_t=60,
        include_keyframe_cond=False,
    )
    embeddings = torch.zeros(3, 8)
    sigmas = {
        name: minimax_h3_time_shift_sigmas(num_steps=steps, shift_scale=scale)
        for name, scale in (("video", 12), ("audio", 3))
    }
    counts = []
    observed = []

    def forward(_model, kwargs, step):
        counts.append(step)
        video_pos = kwargs["img_pos_info"]["position_ids"]
        audio_pos = kwargs["audio_pos_info"]["position_ids"]
        positions = kwargs["img_position_ids"][0]
        video = kwargs["x"][0, video_pos]
        audio = kwargs["audio_x"][0, audio_pos]
        observed.append(
            (
                video.clone(),
                audio.clone(),
                positions[video_pos].clone(),
                positions[audio_pos].clone(),
            )
        )
        # Depend on the current shared state and global coordinates.
        return (
            video * 0.25 + positions[video_pos, :1] / 100,
            audio * 0.5 + positions[audio_pos, :1] / 100,
        )

    model = SimpleNamespace(prepare_adaln_plans=lambda _: None)
    window = prepare_windows(
        packed,
        packed["token_tags"],
        embeddings,
        latent_shape=shape,
        audio_t=60,
        sigmas=sigmas,
        overlap=0.2,
        device=device,
    )
    video = torch.arange(48 * 96, dtype=torch.float32).reshape(48, 96) / 100
    audio = torch.arange(120 * 32, dtype=torch.float32).reshape(120, 32) / 100
    original_video, original_audio = video.clone(), audio.clone()

    def run(window_forward, prefix=global_steps, policy=None):
        return minimax_h3_denoise_loop(
            model=model,
            model_forward=forward,
            window_forward=window_forward,
            global_steps=prefix,
            window_policy=policy,
            positive=MiniMaxH3DenoiseBranch(
                packed=packed,
                text_embeddings=embeddings,
                token_tags=packed["token_tags"],
                device=device,
            ),
            initial_video_rows=video,
            initial_audio_rows=audio,
            keyframe_cond_rows=None,
            sigmas_video=sigmas["video"],
            sigmas_audio=sigmas["audio"],
            device=device,
        )

    expected = run(None)
    full_inputs = observed.copy()
    counts.clear()
    observed.clear()

    # A full window uses the original branch/scheduler directly (W0).
    def unexpected_window(*args):
        pytest.fail("A full window must use the native forward")

    actual_single = run(unexpected_window, prefix=steps)
    assert counts == list(range(steps))
    for actual, reference in zip(actual_single, expected):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)

    for seen, reference in zip(observed, full_inputs):
        for value, expected_value in zip(seen, reference):
            torch.testing.assert_close(value, expected_value, rtol=0, atol=0)

    counts.clear()
    observed.clear()
    repacked = prepare_windows(
        packed,
        packed["token_tags"],
        embeddings,
        latent_shape=shape,
        audio_t=60,
        sigmas=sigmas,
        overlap=0.2,
        device=device,
        full_window=True,
    )
    actual_repack = run(partial(repacked.predict, forward), policy="L" * steps)
    assert counts == list(range(steps))
    for actual, reference in zip(actual_repack, expected):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    for seen, reference in zip(observed, full_inputs):
        for value, expected_value in zip(seen, reference):
            torch.testing.assert_close(value, expected_value, rtol=0, atol=0)

    counts.clear()
    observed.clear()
    actual = run(partial(window.predict, forward))
    assert counts == [
        step for step in range(steps) for _ in range(1 if step < global_steps else 2)
    ]
    for actual_state, expected_state in zip(actual, expected):
        torch.testing.assert_close(actual_state, expected_state, rtol=2e-6, atol=2e-6)
    for index, step in enumerate(counts[global_steps:], start=global_steps):
        local = window.windows[(index - global_steps) % 2]
        v, a, vp, ap = full_inputs[step]
        for seen, reference in zip(
            observed[index],
            (
                v[local.video],
                a[local.audio.flatten()],
                vp[local.video],
                ap[local.audio.flatten()],
            ),
        ):
            torch.testing.assert_close(seen, reference, rtol=2e-6, atol=2e-6)
    assert torch.equal(video, original_video)
    assert torch.equal(audio, original_audio)

    counts.clear()
    policy = "L" * (steps - 2) + "GL"
    actual_late = run(partial(window.predict, forward), policy=policy)
    assert counts == [
        step for step, mode in enumerate(policy) for _ in range(1 if mode == "G" else 2)
    ]
    for actual_state, expected_state in zip(actual_late, expected):
        torch.testing.assert_close(actual_state, expected_state, rtol=2e-6, atol=2e-6)
