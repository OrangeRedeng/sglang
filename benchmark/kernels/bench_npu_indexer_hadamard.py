"""Profile BF16 Hadamard-128 + MXFP8; probe an existing A5 PTO FHT binary.

--inputs accepts a torch.save'd BF16 q or k tensor immediately before Hadamard.
Without it, synthetic inputs provide kernel diagnostics only. The FP16 PTO
candidate is never timed unless quantized bytes and E8M0 scales match exactly.
This probe does not enable FHT in serving or validate indexer top-k/quality.
"""

import argparse
import ctypes
import hashlib
import json
from pathlib import Path

PTO_REFERENCE = (
    "https://github.com/huawei-csl/pto-kernels/tree/"
    "52236e7fc7147329eef9e0118cdad789e553690f/examples/jit_cpp/fast_hadamard_a5"
)


def load_pto_fht(path):
    lib = ctypes.CDLL(str(path.resolve()))
    rows_for = lib.hadamard_rows_for
    rows_for.argtypes = [ctypes.c_uint32]
    rows_for.restype = ctypes.c_uint32
    tile_rows = int(rows_for(128))
    if tile_rows <= 0:
        raise ValueError("PTO binary has no N=128 instantiation")
    kernel = lib.call_hadamard
    kernel.argtypes = [
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
    ]
    kernel.restype = None
    return kernel, tile_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--tensor-name", choices=("q", "k"), default="q")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dst-type", choices=("e4m3fn", "e5m2"), default="e4m3fn")
    parser.add_argument("--pto-so", type=Path)
    parser.add_argument("--block-dim", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        min(args.tokens, args.heads, args.block_dim, args.iterations) <= 0
        or args.warmup < 0
    ):
        parser.error(
            "dimensions, block-dim and iterations must be positive; warmup >= 0"
        )

    import torch
    import torch_npu
    from npu_tp_bench_utils import timing

    from sglang.srt.environ import envs
    from sglang.srt.layers.attention.dsa.dsa_npu_indexer import (
        _quantize_npu_indexer_activation,
        create_npu_hadamard_128,
    )

    torch.npu.set_device(args.device)
    torch.manual_seed(0)
    x = (
        torch.load(args.inputs, map_location="cpu", weights_only=True).to(args.device)
        if args.inputs
        else torch.randn(args.tokens, args.heads, 128, device=args.device).bfloat16()
    )
    if x.dtype != torch.bfloat16 or x.ndim < 2 or x.shape[-1] != 128 or not x.numel():
        parser.error("Expected nonempty BF16 [...,128] input")
    hadamard = create_npu_hadamard_128(128, x.device)
    dst_type = getattr(torch, f"float8_{args.dst_type}")
    report = {
        "input": str(args.inputs) if args.inputs else "synthetic",
        "shape": list(x.shape),
        "stride": list(x.stride()),
        "dst_type": str(dst_type),
        "device": torch.npu.get_device_name(),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "scope": "isolated kernel timings; no TTFT or route/quality acceptance",
    }
    runs = {
        "matmul": lambda: x @ hadamard,
        "matmul_mx_quant": lambda: _quantize_npu_indexer_activation(
            x, hadamard, dst_type, tensor_name=args.tensor_name
        ),
    }
    rejected = False
    with (
        torch.inference_mode(),
        envs.SGLANG_NPU_DSA_INDEXER_HADAMARD_MODE.override("matmul"),
    ):
        if args.pto_so:
            if "950" not in report["device"]:
                parser.error("The PTO candidate requires Ascend 950/A5")
            if not torch.isfinite(x.to(torch.float16)).all().item():
                parser.error("Input is not finite/representable in FP16")
            kernel, tile_rows = load_pto_fht(args.pto_so)
            coefficient = hadamard[0, 0].float()
            rows = x.reshape(-1, 128)
            padded_rows = -(-rows.shape[0] // tile_rows) * tile_rows

            def fht():
                # Pad outside the upstream wrapper to avoid its device-wide wait.
                buf = torch.empty(
                    (padded_rows, 128), device=x.device, dtype=torch.float16
                )
                buf[: rows.shape[0]].copy_(rows)
                buf[rows.shape[0] :].zero_()
                stream = torch.npu.current_stream()
                kernel(
                    args.block_dim,
                    stream._as_parameter_,
                    buf.data_ptr(),
                    padded_rows,
                    128,
                )
                buf.record_stream(stream)
                # Match the stored BF16 matrix coefficient, not exact 1/sqrt(128).
                return (
                    (buf[: rows.shape[0]].float() * coefficient)
                    .bfloat16()
                    .reshape(x.shape)
                )

            def fht_quant():
                rotated = fht()
                quantized, scale = torch.ops.npu.npu_dynamic_mx_quant(
                    rotated.reshape(-1, 128), dst_type=dst_type, axis=-1
                )
                return quantized.reshape(x.shape), scale.reshape(x.shape[:-1] + (2, 2))

            expected = runs["matmul_mx_quant"]()
            actual = fht_quant()
            mismatches = [
                int(
                    (
                        a.contiguous().view(torch.uint8)
                        != b.contiguous().view(torch.uint8)
                    )
                    .sum()
                    .item()
                )
                for a, b in zip(expected, actual)
            ]
            report["pto"] = {
                "upstream_reference": PTO_REFERENCE,
                "binary_sha256": hashlib.sha256(args.pto_so.read_bytes()).hexdigest(),
                "tile_rows": tile_rows,
                "block_dim": args.block_dim,
                "quantized_byte_mismatches": mismatches[0],
                "scale_byte_mismatches": mismatches[1],
                "status": "REJECTED"
                if any(mismatches)
                else "BYTES_MATCH_ON_THIS_INPUT",
            }
            rejected = any(mismatches)
            if not rejected:
                runs.update(pto_fht=fht, pto_fht_mx_quant=fht_quant)

        with envs.SGLANG_NPU_RESOURCE_SCHED_DIAGNOSTICS.override(False):
            report["timings"] = {
                name: timing(run, args.warmup, args.iterations)
                for name, run in runs.items()
            }
        if args.profile_dir:
            with envs.SGLANG_NPU_RESOURCE_SCHED_DIAGNOSTICS.override(True):
                with torch_npu.profiler.profile(
                    activities=[
                        torch_npu.profiler.ProfilerActivity.CPU,
                        torch_npu.profiler.ProfilerActivity.NPU,
                    ],
                    schedule=torch_npu.profiler.schedule(
                        wait=0, warmup=1, active=3, repeat=1
                    ),
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                        str(args.profile_dir)
                    ),
                    record_shapes=True,
                ) as prof:
                    for _ in range(4):
                        for name, run in runs.items():
                            with torch.profiler.record_function(
                                f"indexer_hadamard_probe.{name}"
                            ):
                                run()
                        prof.step()
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if rejected:
        raise SystemExit(
            "PTO FP16 FHT changed MXFP8 bytes/scales; candidate timing skipped"
        )


if __name__ == "__main__":
    main()
