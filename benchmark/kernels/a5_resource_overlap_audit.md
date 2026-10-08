# A5 resource scheduling / indexer Hadamard audit — 2026-10-08

The runner and target eligibility described below are the original audit snapshot.
For the current NeoX/CP/DCP patch and commands, see [the target guide](a5_resource_overlap_target.md).

Audited the existing uncommitted patch in `sglang-a5-merge`, based on
`731252c7d07a56255c7369b00158e751804ec72f`, against the supplied handoff.
Phase 1 has substantial implementation, but the full shared-expert schedule and
hardware acceptance are incomplete. All serving experiments remain opt-in.

| Control | Default | Implementation / status |
|---|---|---|
| `SGLANG_NPU_DSA_OVERLAP_QPROJ_KVNORM` | `False` | Separate q RMSNorm, q projection side stream, KV norm on caller, event join before q split. Non-NeoX eager extend; excludes capture and qlora gather. |
| `SGLANG_NPU_DSA_OVERLAP_QPROJ_KVNORM_MIN_TOKENS` | `0` | Independent threshold; negative values rejected. |
| `SGLANG_NPU_DSA_INDEXER_STREAM_MODE` | `legacy` | `inline` serializes weights projection; `resource` serializes projections and uses one side stream for cast/split/K norm/paired RoPE. Paired RoPE waits for K, so q RoPE does not overlap wk independently. CP/DCP, capture, NeoX and qlora-gather cases fall back. |
| `SGLANG_NPU_TP_MOE_SHARED_PIPELINE` | `legacy` | **Partial:** grouped-fused shared GateUp/activation/quant overlaps TopK/init routing; routed GMM1 waits for shared quant. Shared Down follows routed GMM2. No independent activation boundary in the fused operators. Context resets on exceptions; fused-GMM2-finalize and unsupported topology are excluded. |
| `SGLANG_NPU_DSA_INDEXER_HADAMARD_MODE` | `matmul` | Dense BF16 path retained. `fht` intentionally rejects until an appropriate primitive passes target accuracy. |
| `SGLANG_NPU_RESOURCE_SCHED_DIAGNOSTICS` | `False` | Mode diagnostics and, after this audit, separate q/k Hadamard and MX-quant profiler scopes. |
| Existing `SGLANG_NPU_TP_MOE_MXFP8_GATE_TOPK_LAYOUT` | `default` | Added `materialized_nd`: contiguous clone then format cast 2. Auto candidates unchanged. |

The first six controls were already present in the draft. This audit adds no
environment variables. Existing qnope/RoPE overlap, eager indexer, query gather
and native TP fusion controls remain in place.

The runner includes baseline anchors, HCCL/TASKQ sweep, DCP engagement check,
single cases, combinations and diagnostic profiles. Fixed its Stage C eligibility:
TTFT alone no longer makes a candidate eligible for combination without a
successful correctness command. Profiles do not rank TTFT. The analyzer classifies
Cube/Vector from task/core metadata and reports unknown/mixed coverage explicitly;
its pair-overlap values exclude unknown/mixed kernels.

Still pending: fresh-server A5 runs, captured-input route/output correctness,
actual resource classification and overlap measurements. The full three-stage
shared schedule is not implemented. Optional MXFP4 physical-layout A/B, gate token
pipeline and MC2 research remain Phase 2 work; the optional weight-layout env is
not declared. No end-to-end speedup was measured here.

## Existing kernel found

[Upstream A5 FHT](https://github.com/huawei-csl/pto-kernels/tree/52236e7fc7147329eef9e0118cdad789e553690f/examples/jit_cpp/fast_hadamard_a5)
has a Vector implementation for N=128 and other powers of two on Ascend 950.
Verified source contract at the pinned revision:

- `fast_hadamard_a5.cpp` uses `half` / `vector_f16`, transforms in place and is unnormalized.
- `jit_util_a5.load_lib()` explicitly rejects BF16 and noncontiguous inputs.
- Its padding branch invokes `torch.npu.synchronize()`.
- No fused MXFP8/E8M0 path is supplied by this example; separate quant examples use INT4/INT8.

It is a concrete experimental candidate, not a BF16 replacement. The installed
torch_npu 2.10/CANN 9.2 ABI cannot be established on this host. The local
`sgl-kernel-npu` checkout contains no Hadamard implementation; the SGLang JIT FHT
uses CUDA. The inspected torch_npu source tree had no Hadamard-named entry.

`bench_npu_indexer_hadamard.py` probes the existing A5 binary without adding or
modifying kernel code. Its host adapter pads before launching, preserves the
input, uses the current stream and applies the stored BF16 matrix coefficient
after the FP16 transform. FP16 butterfly arithmetic can change quantization.
The probe checks FP8 bytes and E8M0 scale bytes and refuses candidate timing on
any mismatch. A pass applies only to that input; indexer top-k and model quality
still need real-capture checks. Synthetic data is explicitly marked diagnostic.

## Run on A5

Enable `SGLANG_NPU_RESOURCE_SCHED_DIAGNOSTICS=1` during a normal server profile.
The CPU scopes are `npu_dsa_indexer.q.hadamard.matmul`,
`npu_dsa_indexer.k.hadamard.matmul`, and the corresponding `.mx_quant` scopes.
Use their operator/launch correlations and device task/core metadata to identify
the exact device kernels; scope names alone do not establish a resource class.

First profile the current path on a captured tensor immediately before Hadamard
(q after query sharding, k before the cache write):

```bash
PYTHONPATH=python python3 benchmark/kernels/bench_npu_indexer_hadamard.py \
  --inputs /path/indexer_q_before_hadamard.pt --tensor-name q \
  --profile-dir /tmp/a5-hadamard-q-profile --output /tmp/a5-hadamard-q.json
```

Repeat for k with `--tensor-name k` and its captured tensor. Select `--dst-type`
to match the pool (`e4m3fn` is the probe default). To compile the existing upstream
example, use its pinned checkout, CANN environment and `jit_util_a5.compile_kernel`:

```bash
cd /path/pto-kernels/examples/jit_cpp/fast_hadamard_a5
python3 -c 'from jit_util_a5 import compile_kernel; print(compile_kernel(n=128))'
```

From the SGLang worktree, probe the resulting binary:

```bash
PYTHONPATH=python python3 benchmark/kernels/bench_npu_indexer_hadamard.py \
  --inputs /path/indexer_q_before_hadamard.pt --tensor-name q \
  --pto-so /path/pto-kernels/examples/jit_cpp/fast_hadamard_a5/build/fht_a5.so \
  --profile-dir /tmp/a5-fht-q-profile --output /tmp/a5-fht-q.json
```

The JSON records binary SHA256, input geometry, quant/scale mismatches and
unprofiled kernel timings. There is no serving `fht` switch activation from this probe.

Run the existing resource experiment runner with the real checkpoint and command files:

```bash
PYTHONPATH=python python3 benchmark/kernels/run_a5_resource_overlap_experiments.py \
  --model-path /path/GLM-5.2-W4A8C8 \
  --server-args-file /path/server-args.txt \
  --benchmark-command-file /path/benchmark.txt \
  --correctness-command-file /path/check-routes-and-output.txt \
  --profile-command-file /path/profile.txt --profile-mode best \
  --stages A B C --runs 3 --output-dir /tmp/a5-resource-20261008
```

The server args must explicitly select the checkpoint's FP8 KV dtype. Geometry
is TP4/EP1, chunk/max-prefill 16384, page 128. Benchmark files must emit the
17-request / 3,731,608-input-token / 17-output-token signature. Correctness checks
must compare captured router logits, top-k IDs/weights and final output.

## Local validation

36 CPU tests passed: 10 new scheduling/runner tests and 26 existing autotune tests.
Two existing torch-dependent tests skipped. `py_compile`, Ruff checks on the
touched files, CLI help and `git diff --check` passed. CPU tests exercise production
decision bodies and context cleanup; they do not validate NPU events, tensor
lifetime, kernel accuracy, HCCL or performance. This host has no torch/torch_npu,
CANN installation or NPU device.
