# A3 baseline recovery and EP16 residual experiment

The comparison baseline is the previously measured **24.5888492 s cached TTFT** recipe, not the ordinary unoptimized control. The historical number remains a reference until reproduced on the current build; every candidate is compared with a fresh matched baseline.

## Recovered Saturday run

Source: Codex chat `Finish A3 merge and push 8-NPU tests (2)`, session `01a18cf9-72ea-71a0-b3dc-ea8acd6a2f78`; actual server attachment `d27f29ba-b75a-461b-b350-27249e8a425e`, client metrics attachment `b9e53b8a-f8d7-4e70-97da-fbece4fec956` (also `50820fb6-2a47-4bfc-a097-c152a76a062f`).

- Full GLM5.2 ModelSlim W4A8 weights; eager TP16/DCP16, EP1.
- Chunk 32768; running-request cap 1; prefill cap 1; memory fraction 0.85.
- Local KV rows 67584 (1081344 logical), context limit 1048576; INT8 SFA KV (~5.98 GiB/rank).
- Token-sharded residuals, full-width Q, streamed o_proj, local top-k, packed read, gather prefetch, SFA overlap, decode Q preissue, INT8 indexer and multistream enabled. Shared-expert fusion left enabled.
- HCCL buffer 1000, DeepEP buffer 2500, loopback interfaces, AIV, expandable segments, CPU affinity and 32 streams; chat encode pool/memo/raw enabled.
- AISBench: nominal 1000000 input, one output token, 16 requests/concurrency 16, seed 4, 99% cached-prefix workload, request rate 0. Chat framing produced mean 1000012.1875 input tokens.
- Cached TTFT mean 24.5888492 s, P90 41.6045252 s; E2E wall 46.1064093 s; 0.347 requests/s. Cold prefix preparation took 182.6229746 s. TPOT is undefined for one output token.

The later 1M result (39.50060 s control, 34.76681 s MegaMoE) used chunk 16384, memory fraction 0.88, separate shared experts and different attention/quantization flags. All 1000000 tokens were retained, with 989184 cached. It does not establish a regression against the recovered recipe. The 500k results were 18.94556 s and 15.91719 s. Both MegaMoE runs failed parity (13767/20803 values, max absolute error 14.0983634); their timings remain diagnostic.

## Implementation and limits

The TP residual path gathers quantized tokens before expert GEMMs and reduce-scatters the result. EP16 instead dispatches each rank's local token rows, combines routed expert results back to their owner, adds the replicated shared expert locally, and skips that TP reduction. The existing input quantize/dequantize step is preserved. Empty ranks still participate in dispatch. Only FuseEP mode 3 with MoE TP1 and matching TP16/DCP16/EP16 rank order is admitted. Speculation, incompatible parallel layouts and prefill graph restrictions remain enforced.

This is an experimental runtime repair, not A3 qualification. Local row-ownership tests cover dense TP and local EP, ragged/padded token grids, final logits and input-logprob gathering. Strict A3 parity and activation evidence remain required before MegaMoE timing is admitted.

The historical 32768-token chunk gives 2048 local rows with token sharding. MegaMoE dispatch capacity is therefore 2048, propagated to both qualification and timed launches. It is not the global scheduler chunk. Compared with capacity 4096, this reduces the known fixed workspace bound from ~18 GiB to ~9 GiB; actual peak memory must be read from the A3 logs. No dependencies are upgraded.

## Runs

Run `start_a3_retest.sh historical-repro` for one matched 1M/C16 baseline repetition, or `start_a3_retest.sh overnight-repair` for qualification followed by one matched repetition per admitted variant.

The nightly profile keeps the complete historical baseline stack for all candidates. It covers TP MoE scaling/finalize/prequant/shared/activation, MegaMoE, stream/HCCL tuning, combinations and an FP16-indexer ablation. INT8 indexer is already enabled in baseline; the duplicate INT8 case is reported as blocked coverage. Prefill graphs remain blocked with this DCP baseline rather than silently changing its geometry.

Budget: baseline serving qualification previously took ~700 s. About 15 executable serving candidates plus reference reuse suggest ~3–4 hours for qualification. Allow ~8–15 minutes per admitted 1M benchmark (load plus cold prefix preparation and cached C16 requests), ~2–3.5 additional hours for 13 variants. Plan roughly 5–8 hours, with a 9.5-hour global deadline. Failing/inactive candidates are skipped; the driver does not start another timed case with less than 15 minutes remaining. Timeouts also respect the remaining budget. This is an estimate, not a guarantee that every case will finish.

Full host/client/container artifacts and resolved launch settings are retained by the existing runner. Never relabel smoke, inactive, failed parity or budget-blocked coverage as a pass.
