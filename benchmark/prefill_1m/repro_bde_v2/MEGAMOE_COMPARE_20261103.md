# Two-launch historical baseline / MegaMoE comparison

Run on the A3 HOST after updating `glm52-a3-optimizations`:

```bash
REPO=/home/swx1199799/megamoe_test/sglang
docker cp "sglang-a3:$REPO/benchmark/prefill_1m/start_a3_retest.sh" /tmp/start_a3_retest.sh
screen -dmS "a3-megamoe-compare-$(date +%Y%m%d-%H%M%S)" bash /tmp/start_a3_retest.sh megamoe-compare
```

Exactly two sequential server launches: historical BDE baseline, then that recipe plus MegaMoE EP16. No reference-server or serving-qualification launches. Both use full GLM5.2 ModelSlim W4A8 weights, eager TP16/DCP16, context limit 1048576, nominal 1000000 input tokens, chunk 32768, running/prefill caps 1, local KV rows 67584, memory fraction .85, INT8 KV/indexer and the historical overlap flags. MegaMoE uses FuseEP3/MoE TP1, dispatch capacity 2048, and separate replicated shared experts as required by that backend. No baseline optimization flags are disabled to accommodate MegaMoE.

AISBench: 16 client requests/concurrency 16, 99% shared prefix, seed 4, request rate 0, one output token, one repetition. Each invocation prepares its own prefix before measuring cached requests. The old 24.5888492 s is a historical comparison target, not a promised result. TPOT is undefined.

The runner skips numerical qualification entirely. It still validates actual topology, real/full weights, KV capacity and activation from the timed server logs. Successful measurements are `diagnostic-unqualified`, quality `not_checked`; exit 2 denotes those incomplete-quality results, while exit 1 indicates an actual failure. Do not count a missing or inactive MegaMoE path as a speed result.

Results: `/home/tbaydasov/aisbench/overnight-results/retest-megamoe-compare-*/results/SUMMARY.md`. Per-case folders retain raw AISBench logs/CSV, workload, activation and aggregates; container artifacts retain launch settings, server info and server logs. Archive and empty-file removal behavior are unchanged. The profile has a two-hour budget.

## Repairs prompted by the previous run

- Baseline launch used the composed historical DCP16 recipe, but validation looked up the plain DCP1 scenario. Validation now uses the same composition and checks the API's `ep_size` field for MegaMoE.
- The observed MegaMoE watchdog stack stopped at a blocking host-to-NPU construction of packed KV length metadata in sparse attention. That metadata is now prepared before gather issue, using a device fill and a buffer reused while row count/device match. A changed prefix length refreshes the buffer. This removes the confirmed synchronous copy; only another A3 run can establish whether the distributed hang is resolved.

Local verification: focused runner and prefix-gather regressions, profile dry-run, shell syntax and Python lint checks. No target-device run or dependency upgrade was performed in this workspace.
