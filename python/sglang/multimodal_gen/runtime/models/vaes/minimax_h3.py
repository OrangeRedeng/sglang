# SPDX-License-Identifier: Apache-2.0

import torch

from sglang.multimodal_gen.configs.models.vaes.minimax_h3_audio import (
    MiniMaxH3AudioVAEConfig,
)
from sglang.multimodal_gen.configs.models.vaes.minimax_h3_video import (
    MiniMaxH3VideoVAEConfig,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
)
from sglang.multimodal_gen.runtime.models.vaes.fast_path_gate import (
    VaeFastPathGate,
    register_vae_fast_path_gate,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_audio_vae import (
    DacAudioVAE,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae import (
    AutoencoderKLLegacy,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.klvae import (
    get_tile_parallel_state,
    tile_owners,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_vae_npu_graph import (
    TileDecodeGraphCache,
    npu_graph_api_available,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class MiniMaxH3VideoVAE(AutoencoderKLLegacy, LayerwiseOffloadableModuleMixin):
    layerwise_offload_dit_group_enabled = False
    # EncoderFCN3D indexes its down containers instead of calling them, so they
    # cannot host layerwise hooks. Keep the small encoder resident.
    layer_names = ["decoder.transformer_blocks"]

    def __init__(self, config: MiniMaxH3VideoVAEConfig) -> None:
        arch = config.arch_config
        parallel_decode_mode = config.resolved_parallel_decode_mode()
        use_tiled_decode = config.use_tiling and parallel_decode_mode == "tiled"
        super().__init__(
            in_channels=3,
            out_ch=3,
            ch=128,
            embed_dim=24,
            z_channels=24,
            use_3d_conv=True,
            zq_ch_encoder=None,
            zq_ch_decoder=None,
            num_res_blocks=2,
            num_res_blocks_decoder=None,
            ch_mult=[1, 2, 2, 4, 4, 8],
            space_down=[2, 2, 2, 2, 1, 1],
            space_up=[1, 2, 2, 2, 2, 1],
            time_down=[1, 2, 2, 1, 1, 1],
            time_up=None,
            padding_mode="reflect",
            padding_mode_t=None,
            use_t_isolated_gn=True,
            causal_encoder=True,
            causal_decoder=False,
            use_vit_decoder=True,
            vit_decoder_2d=arch.vit_decoder_2d,
            vit_decoder_kwargs=arch.vit_decoder_kwargs,
            shift_factor=0.0,
            scaling_factor=1.0,
            pixel_norm_type="imagenet",
            clip_length=arch.vae_clip_length,
            token_drop=arch.vae_token_drop,
            encoder_tiling=bool(arch.vae_encoder_tiling),
            decoder_tiling=use_tiled_decode,
            parallel_tiling=use_tiled_decode
            and config.use_parallel_decode
            and config.use_parallel_tiling
            and bool(arch.vae_parallel_tiling),
            decoder_tile_batch=config.decoder_tile_batch,
            window_batch=config.window_batch,
            tile_size=int(arch.vae_tile_size),
            tile_overlap_min=int(arch.vae_tile_overlap_min),
            encoder_parallel=False,
            decoder_parallel=False,
            chunk_dim=int(arch.vae_chunk_dim),
        )
        self.sglang_config = config
        self.use_parallel_decode = config.use_parallel_decode
        self.parallel_decode_mode = parallel_decode_mode
        # CUDA/ROCm register this gate in their fast-path installer. NPU only
        # opts into batching, without installing those platforms' fused ops.
        self._sgl_gate = VaeFastPathGate()
        # Opt-in capture/replay of per-tile-group decoder forwards (R06).
        # Enabled via SGLANG_H3_VAE_NPU_GRAPH=1; every miss/error runs eager.
        self._graph_epoch = 0
        self._tile_graph_cache = None
        if current_platform.is_npu() and npu_graph_api_available():
            self._tile_graph_cache = TileDecodeGraphCache(
                revision=lambda: (self._graph_epoch, self._decode_dtype_cache_tag)
            )
            if self._tile_graph_cache.enabled:
                logger.info(
                    "MiniMax H3 VAE NPU graph capture enabled (opt-in, %s=1)",
                    "SGLANG_H3_VAE_NPU_GRAPH",
                )
        if current_platform.is_npu():
            self.decoder_tile_batch = (
                min(config.decoder_tile_batch, 2) if config.decoder_tile_batch else 2
            )
            self.window_batch = 1
            register_vae_fast_path_gate(self, self._sgl_gate)
            logger.info(
                "MiniMax H3 NPU VAE: up to %d tiles per decoder forward for "
                "lossless/high quality; exact keeps one tile",
                self.decoder_tile_batch,
            )

    def _run_tile_tasks(
        self, tiles, tile_indices, forward_fn, cls_agg=None, *, tile_batch: int = 1
    ):
        # Opt-in graph replay for repeated per-group video tile decodes. Only
        # the stateless group decode is captured; gathering, blending and the
        # temporal state stay outside, and every miss/error path runs the
        # unwrapped forward.
        cache = self._tile_graph_cache
        if (
            cache is not None
            and cache.enabled
            and not self.training
            and tiles
            and tiles[0].device.type == "npu"
            and forward_fn == self.decode
        ):
            forward_fn = cache.wrap(forward_fn)
        return super()._run_tile_tasks(
            tiles, tile_indices, forward_fn, cls_agg, tile_batch=tile_batch
        )

    def prepare_decoder_autocast_weights(self, dtype) -> int:
        stale = self.decoder.prepare_autocast_linear_weights(dtype)
        if stale and self._tile_graph_cache is not None:
            # Captured graphs hold references to the previous weight copies.
            self._graph_epoch += 1
            self._tile_graph_cache.clear(reason="decoder weights recast")
        return stale


class MiniMaxH3ImageVAE(MiniMaxH3VideoVAE):
    """H3 CNN encoder with the 2D ViT image decoder (``vit_decoder_2d``)."""

    # encode() is only quant_conv(encoder(x)); tile assignment/gather and the
    # seeded posterior sample happen outside this shared cache boundary.
    conditioning_encode_is_rank_local = True

    def __init__(self, config: MiniMaxH3VideoVAEConfig) -> None:
        super().__init__(config)
        # Share the base VAE's gate so the CUDA installer's attention fusion
        # and this tiled image decode switch together.
        self.decode_fast_path_gate = self._sgl_gate
        register_vae_fast_path_gate(self, self.decode_fast_path_gate)

    def _run_tile_tasks(
        self, tiles, tile_indices, forward_fn, cls_agg=None, *, tile_batch: int = 1
    ):
        # Reuse native overlapping-tile assembly and batched ViT attention.
        # Bound activation memory as the image canvas grows. Encoding and exact
        # requests retain their original scheduling and numerical path.
        if (
            self.decode_fast_path_gate.enabled
            and not self.training
            and tile_indices
            and tiles[0].is_cuda
            and forward_fn == self.decode
        ):
            tile_rank, tile_world_size = get_tile_parallel_state()
            parallel_decode = (
                self.use_parallel_decode
                and self.sglang_config.use_parallel_tiling
                and not self.parallel_tiling
                and 1 < tile_world_size <= len(tiles)
                and tile_indices == list(range(len(tiles)))
            )
            if parallel_decode:
                owners = tile_owners(len(tiles), tile_world_size)
                tile_indices = self._local_tile_indices(owners, tile_rank)
            # Stack up to 128 samples of tiles per decoder forward.
            chunk_size = max(1, 128 // tiles[0].shape[0])
            outputs = super()._run_tile_tasks(
                tiles, tile_indices, forward_fn, cls_agg, tile_batch=chunk_size
            )
            if parallel_decode:
                return self._all_gather_tiled_results(outputs, owners)
            return outputs
        return super()._run_tile_tasks(
            tiles, tile_indices, forward_fn, cls_agg, tile_batch=tile_batch
        )

    def prepare_decoder_autocast_weights(self, dtype) -> int:
        # Cast every decoder-side parameter and floating buffer to ``dtype``,
        # as diffusers from_pretrained(torch_dtype=...) does.
        stale = sum(
            tensor.is_floating_point() and tensor.dtype != dtype
            for module in (self.post_quant_conv, self.decoder)
            for tensor in (*module.parameters(), *module.buffers())
        )
        if stale:
            self.post_quant_conv.to(dtype)
            self.decoder.to(dtype)
        return stale

    @torch.no_grad()
    def decode_image_raw(self, raw_latents: torch.Tensor) -> torch.Tensor:
        """Raw BC1HW latents to BCHW pixels: BF16 decoder under BF16 autocast."""
        self.prepare_decoder_autocast_weights(torch.bfloat16)
        with torch.autocast(device_type=raw_latents.device.type, dtype=torch.bfloat16):
            decoded = self.decode_base(raw_latents, process_image=True, frame_num=1)
        return self.processor.revert_tensor(decoded).squeeze(2)


class MiniMaxH3AudioVAE(DacAudioVAE, LayerwiseOffloadableModuleMixin):
    layerwise_offload_dit_group_enabled = False
    # BigVGAN stores each executable upsampler inside a one-element ModuleList.
    # The outer ``decoder.ups`` containers are indexed but never called, so hooks
    # must target the inner lists whose ConvTranspose1d modules run forward.
    layer_names = [
        "encoder.block",
        *(f"decoder.ups.{index}" for index in range(7)),
        "decoder.resblocks",
    ]

    def __init__(self, config: MiniMaxH3AudioVAEConfig) -> None:
        super().__init__(
            encoder_dim=64,
            encoder_rates=[2, 4, 4, 5, 5],
            latent_dim=2048,
            decoder_dim=1024,
            decoder_rates=[5, 5, 2, 2, 2, 2, 2],
            sample_rate=32000,
            vae_latent_channels=32,
            attn_proj=True,
            decoder_type="bigvgan",
        )
        self.config = config


EntryClass = [MiniMaxH3VideoVAE, MiniMaxH3ImageVAE, MiniMaxH3AudioVAE]

__all__ = ["MiniMaxH3AudioVAE", "MiniMaxH3ImageVAE", "MiniMaxH3VideoVAE"]
