"""Standalone Ascend FIA graph reproducer; requires only torch and torch_npu.

Run on a free NPU after stopping the server. No model, distributed setup,
profiler, files or production code changes. Each layout runs in a subprocess
so a stalled device synchronization cannot block the other layout forever.
The synthetic cache, outputs and workspace keep strong owners throughout.
"""

import argparse
import os
import subprocess
import sys
import time


def log(message):
    print(message, flush=True)


def run_case(args):
    import torch
    import torch_npu

    torch.npu.set_device(args.device)
    torch.manual_seed(42)
    device = f"npu:{args.device}"
    dtype = torch.bfloat16
    swa = args.layout == "swa"
    dim, kv_heads, cache_pages = (256, 8, 473) if swa else (512, 2, 591)
    query_shape = (1, 16, dim) if swa else (1, 1, 16 * dim)
    query = torch.randn(query_shape, device=device, dtype=dtype) * 0.1
    key = torch.randn(
        (cache_pages, 128, kv_heads * dim), device=device, dtype=dtype
    ) * 0.1
    value = torch.randn_like(key) * 0.1
    key[0].zero_()
    value[0].zero_()
    table = torch.zeros((1, 65), device=device, dtype=torch.int32)
    base_table = torch.arange(1, 66, dtype=torch.int32).reshape(1, 65)
    operation = (
        torch_npu.npu_fused_infer_attention_score_v2
        if swa else torch_npu.npu_fused_infer_attention_score
    )
    get_workspace = (
        torch_npu._npu_fused_infer_attention_score_v2_get_max_workspace
        if swa else torch_npu._npu_fused_infer_attention_score_get_max_workspace
    )
    length_name = "actual_seq_kvlen" if swa else "actual_seq_lengths_kv"
    params = dict(
        query=query, key=key, value=value, block_table=table, block_size=128,
        num_key_value_heads=kv_heads,
    )
    if swa:
        mask = torch.triu(
            torch.ones((2048, 2048), dtype=torch.int8, device=device), diagonal=1
        )
        params.update(
            num_query_heads=16, input_layout="TND", actual_seq_qlen=[1],
            atten_mask=mask, sparse_mode=4, pre_tokens=1023,
            next_tokens=2147483647, softmax_scale=dim ** -0.5,
        )
    else:
        params.update(num_heads=16, input_layout="BSH", sparse_mode=0,
                      scale=dim ** -0.5)

    def set_table(length):
        row = base_table.clone()
        if swa:
            row[:, :max(0, (length - 1024) // 128)] = 0
        table.copy_(row)

    def sync(label):
        log(f"{args.layout}: {label}: synchronize begin")
        torch.npu.synchronize()
        log(f"{args.layout}: {label}: synchronize complete")

    references = {}
    for length in (275, 2313):
        set_table(length)
        log(f"{args.layout}: eager kv={length} begin")
        result = operation(**{**params, length_name: [length]})[0]
        sync(f"eager kv={length}")
        references[length] = result.detach().cpu()
        assert torch.isfinite(references[length]).all(), "non-finite eager result"
        del result

    # Same capture-time lengths as production's zero-filled graph inputs by
    # default. --capture-length 2313 is the control for length-dependent tiling.
    set_table(args.capture_length)
    capture_params = {**params, length_name: [args.capture_length]}
    workspace = get_workspace(**capture_params)
    output = torch.empty_like(query)
    lse = query.new_empty(1)
    capture_params.update(workspace=workspace, out=[output, lse])
    log(f"{args.layout}: capture kv={args.capture_length} "
        f"workspace_bytes={workspace.numel() * workspace.element_size()}")
    compute = torch.npu.current_stream()
    capture_stream = torch.npu.Stream(device=args.device)
    update_stream = torch.npu.Stream(device=args.device)
    gate = torch.npu.ExternalEvent()
    graph = torch.npu.NPUGraph()
    sync("before capture")
    with torch.npu.graph(graph, stream=capture_stream, auto_dispatch_capture=False):
        stream = torch.npu.current_stream()
        gate.wait(stream)
        gate.reset(stream)
        torch.npu.graph_task_group_begin(stream)
        operation.out(**capture_params)
        handle = torch.npu.graph_task_group_end(stream)
    log(f"{args.layout}: capture complete")

    for length in (275, 2313):
        set_table(length)
        rebound = {**capture_params, length_name: [length]}
        if swa:
            rebound["actual_seq_qlen"] = [1]
        rebound["out"] = [output, lse]
        update_stream.wait_stream(compute)
        log(f"{args.layout}: replay kv={length} begin")
        started = time.monotonic()
        graph.replay()
        with torch.npu.stream(update_stream):
            torch.npu.graph_task_update_begin(update_stream, handle)
            operation.out(**rebound)
            torch.npu.graph_task_update_end(update_stream)
            gate.record(update_stream)
        log(f"{args.layout}: replay kv={length} update submitted")
        sync(f"replay kv={length}")
        actual = output.detach().cpu()
        torch.testing.assert_close(actual, references[length], rtol=0.03, atol=0.03)
        log(f"{args.layout}: PASS kv={length}, eager parity, "
            f"elapsed_ms={(time.monotonic() - started) * 1000:.1f}")
    log(f"{args.layout}: CASE PASS capture_kv={args.capture_length}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=45)
    parser.add_argument("--capture-length", type=int, choices=(0, 275, 2313), default=0)
    parser.add_argument("--layout", choices=("all", "swa", "full"), default="all")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.child:
        log(f"CASE START layout={args.layout} capture_kv={args.capture_length}")
        run_case(args)
        return 0

    layouts = ("swa", "full") if args.layout == "all" else (args.layout,)
    failed = False
    for layout in layouts:
        command = [sys.executable, "-u", os.path.abspath(__file__), "--child",
                   "--layout", layout, "--device", str(args.device),
                   "--capture-length", str(args.capture_length)]
        log(f"START {layout}, hard timeout={args.timeout}s (includes NPU initialization)")
        process = subprocess.Popen(command)
        try:
            code = process.wait(timeout=args.timeout)
            log(f"RESULT {layout}: {'PASS' if code == 0 else 'FAIL'} exit={code}")
            failed |= code != 0
        except subprocess.TimeoutExpired:
            process.kill()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                log("Child did not exit after kill; stopping remaining cases")
                return 1
            log(f"RESULT {layout}: TIMEOUT; last printed stage identifies the stall")
            failed = True
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
