"""Validate fused Gemma RoPE/KV against the existing rotary and scatter path.

CPU indexing diagnostic: SGLANG_USE_CPU_ENGINE=1 TRITON_INTERPRET=1 PYTHONPATH=python python3 <this-file>
    --cpu --tokens 0 1 15 --samples 1 --output /tmp/rope-cpu.json
CPU diagnostics never produce an admissible NPU validation report.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from sglang.kernels.ops.attention.gemma4_rope_kv_npu import _launch
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.gemma4_ops import npu_gemma4_rope_kv
from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding
from sglang.srt.layers.rotary_embedding.rope_variant import Gemma4RotaryEmbedding

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gemma4.candidate_validation import ROPE_TOKENS, compile_case, new_report


def validate(tokens, dim, dtype, mapped, cache_fp32, samples, *, cpu=False):
    device = "cpu" if cpu else "npu"
    if not cpu:
        import torch_npu

    qh, kh = (16, 8) if dim == 256 else (16, 2)
    with envs.SGLANG_ROPE_CACHE_FP32.override(cache_fp32):
        if dim == 512:
            rotary = Gemma4RotaryEmbedding(dim, 128, 8192, 1e6, True, dtype)
        else:
            rotary = RotaryEmbedding(dim, dim, 8192, 10000, True, dtype)
    rotary = rotary.to(device)
    source = torch.randn(tokens, (qh + 2 * kh) * dim + 17, device=device, dtype=dtype)
    work = source.clone()
    q, k, v, sentinel = work.split([qh * dim, kh * dim, kh * dim, 17], -1)
    if tokens:
        # V is already normalized when Gemma's rotary stage is entered.
        vv = v.reshape(tokens, kh, dim).float()
        v.copy_(
            (vv * torch.rsqrt(vv.square().mean(-1, keepdim=True) + 1e-6))
            .reshape(v.shape)
            .to(dtype)
        )
        source.copy_(work)
    capacity = tokens + 17
    seed = torch.randn(capacity, 1, kh, dim, device=device, dtype=dtype)
    kc, vc = seed.clone(), seed.clone()
    positions = (torch.arange(tokens, device=device, dtype=torch.int64) + 1235) % 8192
    locations = torch.arange(1, tokens + 1, device=device, dtype=torch.int64)
    if tokens > 1:
        locations[-1] = 0
    if tokens > 2:
        locations[-2] = -1
    mapping = (
        torch.arange(capacity, device=device, dtype=torch.int64) if mapped else None
    )
    if mapped:
        mapping[1:] = mapping[1:].flip(0).clone()
        mapping[0] = capacity - 1
    qo, ko = (
        torch.empty(q.shape, device=device, dtype=dtype),
        torch.empty(k.shape, device=device, dtype=dtype),
    )

    def run():
        if cpu:
            if tokens:
                _launch(
                    q,
                    k,
                    v,
                    rotary.cos_sin_cache,
                    positions,
                    locations,
                    qo,
                    ko,
                    kc,
                    vc,
                    qh,
                    kh,
                    dim,
                    rotary.rotary_dim,
                    4,
                    mapping,
                )
            return qo, ko
        return npu_gemma4_rope_kv(
            q,
            k,
            v,
            rotary.cos_sin_cache,
            positions,
            locations,
            kc,
            vc,
            qh,
            kh,
            dim,
            rotary.rotary_dim,
            mapping,
        )

    def native_rope():
        if not tokens:
            return qo, ko
        if cpu:
            angles = rotary.cos_sin_cache[positions].float()
            cos, sin = angles.chunk(2, -1)

            def rotate(x, heads):
                x1, x2 = x.reshape(tokens, heads, dim).float().chunk(2, -1)
                return (
                    torch.cat(
                        (
                            x1 * cos[:, None] - x2 * sin[:, None],
                            x2 * cos[:, None] + x1 * sin[:, None],
                        ),
                        -1,
                    )
                    .to(dtype)
                    .reshape(x.shape)
                )

            qr, kr = rotate(q, qh), rotate(k, kh)
        elif cache_fp32:
            qr, kr = rotary.forward_npu(
                positions, q.reshape(tokens, qh, dim), k.reshape(tokens, kh, dim)
            )
            qr, kr = qr.reshape(q.shape), kr.reshape(k.shape)
        else:
            qr, kr = rotary.forward_npu(positions, q, k)
        return qr, kr

    def native(kbuf, vbuf):
        qr, kr = native_rope()
        slot = locations.clone()
        positive = slot > 0
        if mapped:
            slot[positive] = mapping[slot[positive]]
        valid = positive & (slot > 0) & (slot < capacity)
        slot, kr_valid, v_valid = slot[valid], kr[valid], v[valid]
        if cpu:
            kbuf[:, 0].index_copy_(0, slot, kr_valid.reshape(-1, kh, dim))
            vbuf[:, 0].index_copy_(0, slot, v_valid.reshape(-1, kh, dim))
        else:
            torch_npu.npu_scatter_nd_update_(
                kbuf.view(capacity, kh, dim),
                slot.contiguous().view(-1, 1),
                kr_valid.contiguous().view(-1, kh, dim),
            )
            torch_npu.npu_scatter_nd_update_(
                vbuf.view(capacity, kh, dim),
                slot.contiguous().view(-1, 1),
                v_valid.contiguous().view(-1, kh, dim),
            )
        return qr, kr

    def reset():
        work.copy_(source)
        kc.copy_(seed)
        vc.copy_(seed)
        if not cpu:
            # Flush queued Torch writes before the raw Triton launch/capture.
            torch.npu.synchronize()

    def check(actual, stage):
        expected_k, expected_v = seed.clone(), seed.clone()
        expected = native(expected_k, expected_v)
        if not cpu:
            torch.npu.synchronize()
        tol = 2e-2 if dtype == torch.bfloat16 else 3e-3
        label = f"stage={stage} tokens={tokens} dim={dim} dtype={dtype} mapped={mapped} cache_fp32={cache_fp32}"
        label += f" cache_shape={tuple(kc.shape)} cache_stride={kc.stride()}"
        if not cpu:
            label += f" npu_format={torch_npu.get_npu_format(kc)}"
        for name, a, e in zip(("q", "k"), actual, expected):
            torch.testing.assert_close(
                a,
                e,
                atol=tol,
                rtol=tol,
                msg=lambda detail, name=name: f"{label} component={name}\n{detail}",
            )
        # Cross-check the native scatter oracle without NPU views or queued writes.
        slots = locations.cpu().long()
        live = slots > 0
        if mapped:
            slots[live] = mapping.cpu()[slots[live]]
        live &= (slots > 0) & (slots < capacity)
        cpu_k, cpu_v = seed.cpu().clone(), seed.cpu().clone()
        cpu_k[:, 0].index_copy_(
            0, slots[live], expected[1].cpu().reshape(tokens, kh, dim)[live]
        )
        cpu_v[:, 0].index_copy_(0, slots[live], v.cpu().reshape(tokens, kh, dim)[live])
        for name, a, e in (
            ("native_k_scatter", expected_k.cpu(), cpu_k),
            ("native_v_scatter", expected_v.cpu(), cpu_v),
            ("candidate_k_cache", kc.cpu(), cpu_k),
            ("candidate_v_cache", vc.cpu(), cpu_v),
        ):
            tolerance = 0 if "v_" in name else tol
            changed = (a != e).reshape(capacity, -1).any(1).nonzero().flatten().tolist()
            torch.testing.assert_close(
                a,
                e,
                atol=tolerance,
                rtol=tolerance,
                msg=lambda detail, name=name, changed=changed: (
                    f"{label} component={name} changed_rows={changed[:16]} locations={slots[:8].tolist()}\n{detail}"
                ),
            )
        torch.testing.assert_close(work, source, atol=0, rtol=0)
        torch.testing.assert_close(sentinel, source[:, -17:], atol=0, rtol=0)

    reset()
    actual = run()
    if not cpu:
        torch.npu.synchronize()
    check(actual, "eager")
    if not cpu:
        compiled = compile_case(run)
        reset()
        actual = compiled()
        torch.npu.synchronize()
        check(actual, "fullgraph")
        for mode in (0, 1):
            torch.npu.synchronize()
            torch_npu.npu.set_task_queue_enable(mode)
            reset()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                actual = run()
            for _ in range(2):
                source.normal_()
                positions.copy_(positions.roll(1))
                locations.copy_(locations.roll(1))
                if mapped:
                    mapping[1:].copy_(mapping[1:].flip(0))
                reset()
                graph.replay()
                torch.npu.synchronize()
                check(actual, f"replay_queue{mode}")

    def measure(fn):
        wall, device_times = [], []
        for sample in range(samples + 2):
            reset()
            if not cpu:
                torch.npu.synchronize()
                begin, end = (
                    torch.npu.Event(enable_timing=True),
                    torch.npu.Event(enable_timing=True),
                )
                begin.record()
            start = time.perf_counter_ns()
            fn()
            if not cpu:
                end.record()
                end.synchronize()
            if sample >= 2:
                wall.append((time.perf_counter_ns() - start) / 1000)
                if not cpu:
                    device_times.append(begin.elapsed_time(end) * 1000)
        return {
            "wall_median_us": statistics.median(wall),
            "device_median_us": statistics.median(device_times)
            if device_times
            else None,
        }

    # Time the original primitives with valid slots; prepare batch metadata outside timing.
    locations.copy_(torch.arange(1, tokens + 1, device=device, dtype=torch.int64))
    native_slots = mapping[locations] if mapped else locations

    def native_benchmark():
        qr, kr = native_rope()
        if tokens:
            if cpu:
                kc[:, 0].index_copy_(0, native_slots, kr.reshape(tokens, kh, dim))
                vc[:, 0].index_copy_(0, native_slots, v.reshape(tokens, kh, dim))
            else:
                torch_npu.npu_scatter_nd_update_(
                    kc.view(capacity, kh, dim),
                    native_slots.contiguous().view(-1, 1),
                    kr.contiguous().reshape(tokens, kh, dim),
                )
                torch_npu.npu_scatter_nd_update_(
                    vc.view(capacity, kh, dim),
                    native_slots.contiguous().view(-1, 1),
                    v.contiguous().reshape(tokens, kh, dim),
                )
        return qr, kr

    return {"candidate": measure(run), "native": measure(native_benchmark)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokens", type=int, nargs="+", default=list(ROPE_TOKENS))
    p.add_argument("--samples", type=int, default=20)
    p.add_argument(
        "--head-dims",
        type=int,
        nargs="+",
        choices=[128, 256, 512],
        default=[128, 256, 512],
    )
    p.add_argument(
        "--dtypes",
        nargs="+",
        choices=["float16", "bfloat16"],
        default=["float16", "bfloat16"],
    )
    p.add_argument("--mapped", choices=["all", "yes", "no"], default="all")
    p.add_argument("--cache-fp32", choices=["all", "yes", "no"], default="all")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.samples < 1 or any(t < 0 for t in args.tokens) or args.output.exists():
        p.error("Use nonnegative token counts, positive samples and a new output file")
    if args.cpu:
        torch.set_num_threads(1)
        report = {"kind": "cpu_diagnostic", "complete": False, "cases": []}
    else:
        import torch_npu

        report = new_report("rope_kv", torch, torch_npu)
    torch.manual_seed(48)
    try:
        for dtype_name in args.dtypes:
            dtype = getattr(torch, dtype_name)
            for dim in args.head_dims:
                for tokens in args.tokens:
                    for mapped in (
                        (False, True)
                        if args.mapped == "all"
                        else (args.mapped == "yes",)
                    ):
                        for cache_fp32 in (
                            (False, True)
                            if args.cache_fp32 == "all"
                            else (args.cache_fp32 == "yes",)
                        ):
                            report["current_case"] = {
                                "tokens": tokens,
                                "head_dim": dim,
                                "dtype": dtype_name,
                                "mapped": mapped,
                                "cache_fp32": cache_fp32,
                            }
                            print(f"BEGIN {report['current_case']}", flush=True)
                            timing = validate(
                                tokens,
                                dim,
                                dtype,
                                mapped,
                                cache_fp32,
                                args.samples,
                                cpu=args.cpu,
                            )
                            report["cases"].append(
                                {
                                    "candidate": "rope_kv",
                                    "dtype": str(dtype),
                                    "head_dim": dim,
                                    "tokens": tokens,
                                    "mapped": mapped,
                                    "cache_fp32": cache_fp32,
                                    "eager_correct": True,
                                    "fullgraph_trace_correct": not args.cpu,
                                    "graph_replay_modes": [] if args.cpu else [0, 1],
                                    "timing": timing,
                                }
                            )
        report["complete"] = not args.cpu
        report.pop("current_case", None)
    except Exception as error:
        report["failure"] = {"case": report.get("current_case"), "error": str(error)}
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
    print(f"Validation: {args.output}; CPU diagnostics are not NPU acceptance")


if __name__ == "__main__":
    main()
