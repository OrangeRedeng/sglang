"""Validate opt-in Gemma 4 QKV norm against CANN, including graph replay.

Run with PYTHONPATH=python python3 test/manual/kernels/validate_gemma4_qkv_norm_npu.py
"""

import statistics
import sys
import time
from pathlib import Path

import torch
import torch_npu
from sglang.srt.hardware_backend.npu.gemma4_ops import npu_gemma4_qkv_rmsnorm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gemma4.candidate_validation import compile_case


def validate(
    tokens, head_dim, dtype, shared_kv, *, norm=npu_gemma4_qkv_rmsnorm, samples=0
):
    q_heads, kv_heads = (16, 8) if head_dim == 256 else (16, 2)
    widths = [q_heads * head_dim, kv_heads * head_dim, kv_heads * head_dim, 17]
    source = torch.randn(tokens, sum(widths), device="npu", dtype=dtype)
    work = source.clone()
    q, k, v, sentinel = work.split(widths, dim=-1)
    q_weight = torch.randn(head_dim, device="npu", dtype=dtype)
    k_weight = torch.randn_like(q_weight)
    ones = torch.ones_like(q_weight)

    def reference():
        expected = []
        for index, (x, weight) in enumerate(
            zip(source.split(widths, dim=-1)[:3], [q_weight, k_weight, ones])
        ):
            if shared_kv and index > 0:
                expected.append(x)
                continue
            shape = x.shape
            expected.append(
                torch_npu.npu_rms_norm(x.reshape(tokens, -1, head_dim), weight, 1e-6)[
                    0
                ].reshape(shape)
            )
        return expected

    def run():
        norm(
            q,
            None if shared_kv else k,
            None if shared_kv else v,
            q_weight,
            None if shared_kv else k_weight,
            q_heads,
            kv_heads,
            head_dim,
        )

    def check():
        expected = reference()
        tolerance = 2e-2 if dtype == torch.bfloat16 else 3e-3
        torch.testing.assert_close(q, expected[0], atol=tolerance, rtol=tolerance)
        for index, actual in enumerate([k, v], 1):
            target = (
                source.split(widths, dim=-1)[index] if shared_kv else expected[index]
            )
            torch.testing.assert_close(actual, target, atol=tolerance, rtol=tolerance)
        torch.testing.assert_close(sentinel, source[:, -17:], atol=0, rtol=0)

    torch.npu.synchronize()
    torch_npu.npu.set_task_queue_enable(2)
    run()
    torch.npu.synchronize()
    check()
    compiled = compile_case(run)
    work.copy_(source)
    torch.npu.synchronize()
    compiled()
    torch.npu.synchronize()
    check()
    for mode in (0, 1):
        torch.npu.synchronize()
        torch_npu.npu.set_task_queue_enable(mode)
        graph = torch.npu.NPUGraph()
        work.copy_(source)
        torch.npu.synchronize()
        with torch.npu.graph(graph):
            run()
        for _ in range(2):
            source.normal_()
            work.copy_(source)
            torch.npu.synchronize()
            graph.replay()
            torch.npu.synchronize()
            check()
    print(
        f"PASS tokens={tokens} head_dim={head_dim} dtype={dtype} shared_kv={shared_kv}"
    )

    def measure(step):
        device_us, wall_us = [], []
        for index in range(samples + 2):
            work.copy_(source)
            torch.npu.synchronize()
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
            begin = time.perf_counter_ns()
            result = step()
            end.record()
            end.synchronize()
            elapsed = (time.perf_counter_ns() - begin) / 1000.0
            if index >= 2:
                device_us.append(start.elapsed_time(end) * 1000.0)
                wall_us.append(elapsed)
            del result
        return {
            "device_median_us": statistics.median(device_us),
            "wall_median_us": statistics.median(wall_us),
        }

    return {"candidate": measure(run), "native": measure(reference)} if samples else {}


if __name__ == "__main__":
    torch.manual_seed(0)
    for dtype in (torch.float16, torch.bfloat16):
        for head_dim in (128, 256, 512):
            for tokens in (1, 15, 1235, 8192):
                for shared_kv in (False, True):
                    validate(tokens, head_dim, dtype, shared_kv)
