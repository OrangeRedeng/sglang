# SPDX-License-Identifier: Apache-2.0

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def windowing():
    # Exercise the actual prediction/fusion code without importing GPU kernels.
    root = Path(__file__).resolve().parents[4]
    source = next((root / "python").rglob("minimax_h3/windowing.py"))
    tree = ast.parse(source.read_text())
    tree.body = [
        node
        for node in tree.body
        if not isinstance(node, ast.ImportFrom) or node.module == "dataclasses"
    ]
    namespace = {
        "MiniMaxH3DenoiseBranch": object,
        "maybe_record_function": lambda *args, **kw: nullcontext(),
    }
    exec(compile(tree, str(source), "exec"), namespace)
    return SimpleNamespace(**namespace)


@pytest.mark.parametrize("cosine", [False, True])
def test_shifted_overlap_preserves_av_order_and_coverage(windowing, cosine):
    video = torch.arange(11, dtype=torch.float32)[:, None]
    audio = torch.arange(34, dtype=torch.float32)[:, None]
    branch = SimpleNamespace(forward_kwargs=lambda **kwargs: kwargs)
    offset = windowing.boundary_offsets(11, 0.2)[1]
    assert offset == 1
    plan = windowing.WindowPredictions(
        (
            windowing.Window(
                slice(0, 7),
                torch.arange(34).reshape(2, 17)[:, :10],
                branch,
                [0],
                torch.tensor([10]),
            ),
            windowing.Window(
                slice(5, 11),
                torch.arange(34).reshape(2, 17)[:, 7:],
                branch,
                [0],
                torch.tensor([0]),
            ),
        ),
        windowing._ramp(2, "cpu", cosine),
        windowing._ramp(3, "cpu", cosine),
        ((0, 7, 0, 10), (5, 11, 7, 17)),
        (11, 1, 1),
        None,
        None,
    )
    calls = []

    def forward(_, kwargs, step):
        calls.append((step, kwargs["video_rows"].shape[0]))
        return kwargs["video_rows"], kwargs["audio_rows"]

    evidence = {}
    v, a = plan.predict(forward, None, video, audio, 0, None, diagnostics=evidence)
    torch.testing.assert_close(v, video)
    torch.testing.assert_close(a, audio)
    assert calls == [(0, 8), (0, 7)]
    assert evidence == {"video_disagreement": 0, "audio_disagreement": 0}


def test_adaptive_bucket_moves_away_from_large_temporal_change(windowing):
    def plan(start, stop):
        return SimpleNamespace(
            latent_shape=(21, 1, 1),
            windows=[SimpleNamespace(audio=torch.zeros(2, 15))],
            ranges=((0, stop, 0, stop), (start, 21, start, 21)),
        )

    plans = {0: plan(9, 12), 3: plan(12, 15), -3: plan(6, 9)}
    video = torch.cat((torch.zeros(11), torch.ones(10)))[:, None]
    audio = video.repeat(2, 1)
    assert windowing.choose_boundary(plans, video, audio)[0] == 3
    assert windowing.choose_boundary(plans, video * float("nan"), audio)[0] == 0


def test_real_packer_preserves_global_positions_and_padding(windowing):
    source = next(
        (Path(__file__).resolve().parents[4] / "python").rglob(
            "minimax_h3/packed_sequence.py"
        )
    )
    tree = ast.parse(source.read_text())
    tree.body = [
        node
        for node in tree.body
        if not isinstance(node, ast.ImportFrom)
        or not (node.module or "").startswith("sglang")
    ]
    symbols = {"MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT": 64}
    exec(compile(tree, str(source), "exec"), symbols)
    pack = symbols["minimax_h3_packed_sequence"]

    class Branch:
        def __init__(self, *, packed, **kwargs):
            self.packed = packed

        def prepare_timestep_plan(self, **kwargs):
            return []

    prepare = windowing.prepare_windows
    prepare.__globals__.update(
        minimax_h3_packed_sequence=pack, MiniMaxH3DenoiseBranch=Branch
    )
    original = pack(
        text_len=7,
        latent_t=11,
        latent_h=4,
        latent_w=4,
        audio_t=70,
        include_keyframe_cond=False,
    )
    plan = prepare(
        original,
        original["token_tags"],
        None,
        latent_shape=(11, 4, 4),
        audio_t=70,
        sigmas={"video": [1, 0], "audio": [1, 0]},
        overlap=0.2,
        offset=1,
        anchors=True,
        device="cpu",
    )
    for window in plan.windows:
        local = window.branch.packed
        indices = torch.arange(original["img_pos"].numel())[window.video]
        if window.context is not None:
            assert set(indices.tolist()).isdisjoint(window.context.tolist())
            indices = torch.cat((indices, window.context))
        torch.testing.assert_close(
            local["img_position_ids"][local["img_pos"]],
            original["img_position_ids"][original["img_pos"][indices]],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            local["img_position_ids"][local["audio_pos"]],
            original["img_position_ids"][original["audio_pos"][window.audio.flatten()]],
            rtol=0,
            atol=0,
        )
        assert (local["token_tags"][-1] == -1).item()
