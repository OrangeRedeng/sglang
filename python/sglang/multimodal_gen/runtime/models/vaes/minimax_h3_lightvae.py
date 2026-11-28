# SPDX-License-Identifier: Apache-2.0
"""Load the standard ModelTC H3 LightVAE decoder into the native video VAE.

Layout conversion follows ModelTC/LightVAE's comfyui/h3_vae.py: Diffusers
Q/K/V become head-interleaved QKV, and the FFN's up/gate halves are swapped.
"""

import json
import os

import torch
from safetensors import safe_open
from torch import nn

from sglang.multimodal_gen.runtime.loader.utils import load_model_state_dict
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3 import MiniMaxH3VideoVAE
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)
_DECODER_PREFIXES = ("decoder.", "post_quant_conv.")


def _native_decoder_state(state, heads):
    converted = {}
    for name, tensor in state.items():
        if ".ff.net.0.proj." in name:
            up, gate = tensor.chunk(2, dim=0)
            tensor = torch.cat((gate, up), dim=0)
        name = (
            name.replace(".attn.to_out.0.", ".attn.to_out.")
            .replace("decoder.proj_in.", "decoder.x_embedder.")
            .replace(".ff.net.0.proj.", ".ff.w1.")
            .replace(".ff.net.2.", ".ff.w2.")
        )
        if name in converted:
            raise ValueError(f"Duplicate LightVAE tensor after conversion: {name}")
        converted[name] = tensor
    for name in list(converted):
        if ".attn.to_q." not in name:
            continue
        keys = [name.replace(".to_q.", f".to_{part}.") for part in ("q", "k", "v")]
        tensors = [converted.pop(key) for key in keys]
        shape = tensors[0].shape
        if any(t.shape != shape for t in tensors) or shape[0] % heads:
            raise ValueError(f"Invalid LightVAE QKV shapes at {name}")
        packed = torch.stack(
            [t.reshape(heads, -1, *shape[1:]) for t in tensors], dim=1
        )
        converted[name.replace(".to_q.", ".to_qkv.")] = packed.reshape(
            3 * shape[0], *shape[1:]
        )
    return converted


def load_minimax_h3_lightvae_decoder(vae, path):
    """Replace decoder weights/depth only; preserve original encoder and audio."""
    if type(vae) is not MiniMaxH3VideoVAE or vae.vit_decoder_2d:
        raise ValueError("LightVAE requires the native MiniMax H3 VIDEO VAE")
    path = os.path.realpath(os.path.expanduser(path))
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        if "architecture" not in metadata:
            raise ValueError("LightVAE checkpoint is missing architecture metadata")
        architecture = json.loads(metadata["architecture"])
        if not isinstance(architecture, dict):
            raise ValueError("LightVAE architecture must be a JSON object")
        teacher = architecture.get("teacher_config", architecture)
        if not isinstance(teacher, dict):
            raise ValueError("LightVAE teacher_config must be a JSON object")
        kept = architecture.get("kept_layers")
        depth = len(vae.decoder.transformer_blocks)
        if (
            not isinstance(kept, list)
            or not kept
            or any(type(index) is not int for index in kept)
            or kept != sorted(set(kept))
            or kept[0] < 0
            or kept[-1] >= depth
            or len(kept) >= depth
        ):
            raise ValueError("LightVAE kept_layers must describe a pruned H3 decoder")
        native = vae.decoder.config
        expected = {
            "family": "h3",
            "decoder_num_layers": depth,
            "decoder_num_attention_heads": native.heads,
            "decoder_attention_head_dim": native.dim_head,
            "decoder_num_register_tokens": native.num_register_tokens,
            "decoder_rope_theta": native.rope_theta,
            "decoder_rope_dim_ratio": native.rope_dim_ratio,
            "decoder_norm_eps": native.eps,
            "clip_length": vae.clip_length,
            "token_drop": vae.token_drop,
        }
        for name, value in expected.items():
            if name in teacher and teacher[name] != value:
                raise ValueError(
                    f"LightVAE {name}={teacher[name]!r} differs from native H3 {value!r}"
                )
        for name in ("latents_mean", "latents_std"):
            value = getattr(vae.sglang_config.arch_config, name)
            if name in teacher and not (
                len(teacher[name]) == len(value)
                and all(abs(a - b) < 1e-6 for a, b in zip(teacher[name], value))
            ):
                raise ValueError(f"LightVAE {name} differs from native H3")
        state = {
            name: handle.get_tensor(name)
            for name in handle.keys()
            if name.startswith(_DECODER_PREFIXES)
        }
    if not state or any(not t.is_floating_point() for t in state.values()):
        raise ValueError("Use the standard, non-quantized H3 LightVAE decoder file")
    state = _native_decoder_state(state, native.heads)
    vae.decoder.transformer_blocks = nn.ModuleList(
        [vae.decoder.transformer_blocks[index] for index in kept]
    )
    target = {
        name: tensor
        for name, tensor in vae.state_dict().items()
        if name.startswith(_DECODER_PREFIXES)
    }
    # Unused native training buffer, absent from the Diffusers export.
    if "decoder.mask_token" in target and "decoder.mask_token" not in state:
        state["decoder.mask_token"] = torch.zeros(
            target["decoder.mask_token"].shape,
            dtype=target["decoder.mask_token"].dtype,
        )
    missing = sorted(target.keys() - state.keys())
    unexpected = sorted(state.keys() - target.keys())
    wrong_shapes = [
        name
        for name in target.keys() & state.keys()
        if target[name].shape != state[name].shape
    ]
    if missing or unexpected or wrong_shapes:
        raise ValueError(
            f"Incompatible H3 LightVAE decoder: missing={missing[:8]}, "
            f"unexpected={unexpected[:8]}, wrong_shapes={wrong_shapes[:8]}"
        )
    for prefix, module in (
        ("decoder.", vae.decoder),
        ("post_quant_conv.", vae.post_quant_conv),
    ):
        load_model_state_dict(
            module,
            {
                name[len(prefix) :]: tensor
                for name, tensor in state.items()
                if name.startswith(prefix)
            },
            strict=True,
            assign=next(module.parameters()).device.type == "cpu",
        )
    vae.decoder.register_to_config(num_layers=len(kept))
    stat = os.stat(path)
    vae._decode_dtype_cache_tag = f"|lightvae:{path}:{stat.st_size}:{stat.st_mtime_ns}"
    # Graphs captured against the previous decoder reference stale weights.
    graph_cache = getattr(vae, "_tile_graph_cache", None)
    if graph_cache is not None:
        vae._graph_epoch += 1
        graph_cache.clear(reason="LightVAE decoder loaded")
    logger.info(
        "MiniMax H3 LightVAE decoder loaded: %d -> %d blocks from %s; original encoder retained",
        depth,
        len(kept),
        path,
    )
