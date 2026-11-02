# Cached-prefix and MegaMoE repair check

Based on the completed `retest-overnight-20261102-052839` at `1b7b76129b`.
All NPU unit groups passed, including fused GMM2. Cold baseline mean TTFT was
287.797 s; the cached phase was rejected by the harness before saving timing.
The earlier 24.589 s cached mean used 16 concurrent requests; this repair check
uses one request at concurrency one and does not assume the same mean.

## Confirmed fixes

- Cache validation uses resolved `page_size * dcp_size`, excluding the final
  sampling input token before page alignment. DCP16 accepts the observed
  989184 reusable tokens for a 990000-token prefix; an actually missing aligned
  page still fails. Numerical parity tolerances are unchanged.
- Native client timings are written **before** validation. Invalid samples
  retain their values and an explicit failed validation status.
- A later timed-phase failure returns completed benchmark points to the host
  summary and preserves the enclosing case failure with phase `benchmark`.
- The DCP activation probe keeps full-prefix logprobs on the prime but starts
  follow-up logprobs at the suffix: the scheduler caps prefix reuse at
  `logprob_start_len`, so the old zero value forced zero cache hits.
  It primes more than one full chunk and supplies a
  nontrivial suffix. It records and checks cache reuse. At chunk=16384 and
  DCP16/page128 it uses 18432 prefix + 1024 suffix tokens and requires at least
  16384 reused tokens. With chunk=4096 it uses 6144 + 1024 and requires 4096.
- The **GLM5.2 A3 MegaMoE test configuration** sets
  `DEEPEP_HCCL_BUFFSIZE=768` MiB before process-group creation. Its observed
  native tiling minimum was 603 MiB (200 MiB had been supplied). Existing
  process-group code applies this to MoE; the DCP group continues to use
  `HCCL_BUFFSIZE`. This is an explicit configuration for this model/EP16/cap4096
  test, not a general runtime auto-sizing formula or a dependency upgrade.

Real A3 validation is still required. Remaining routed-scale/shared/prequant/
shared-activation/INT8-indexer serving parity defects are not resolved here.
The new MegaMoE run may expose another native prerequisite or parity failure;
passing CPU contracts does not establish its device result.

## Short verification profiles

Run on the A3 HOST after updating the container checkout and recopying the host
script. Both profiles retain the experiment lock and archive all host/container
logs and resolved arguments.

```bash
REPO=/home/swx1199799/megamoe_test/sglang
docker cp "sglang-a3:$REPO/benchmark/prefill_1m/start_a3_retest.sh" /tmp/start_a3_retest.sh
screen -dmS "a3-fix-$(date +%Y%m%d-%H%M%S)" bash /tmp/start_a3_retest.sh fix-check
```

`fix-check` performs baseline, MegaMoE and the three I/O feature qualifications,
then a **cached-only 1M baseline**: one untimed warmup, three fresh prefix primes
and three measured requests, concurrency one, one output token. No cold timing
sweep or multistream timing is repeated. Expect roughly **60–90 minutes** from
previous startup/qualification costs and the current 288 s cold prefill used
for warmup/priming. A two-hour execution budget may defer cases if they cannot
fit; cleanup follows. Inspect qualification and cached measurements separately.

For only baseline/MegaMoE serving qualification (roughly 20–30 minutes, one-hour
budget):

```bash
screen -dmS "a3-megamoe-$(date +%Y%m%d-%H%M%S)" bash /tmp/start_a3_retest.sh megamoe
```

Run one profile at a time. MegaMoE uses TP16/EP16; cached DCP baseline uses
TP16/DCP16. A missing activation, capacity prerequisite or numerical mismatch
remains a failure/incomplete qualification; none is silently converted to pass.

Artifacts appear under `/home/tbaydasov/aisbench/overnight-results/retest-fix-check-*/`
(or `retest-megamoe-*/`). Read `results/SUMMARY.md` after each stage; inspect
`container-artifacts/long/feature.log` during the long stage. Raw cached samples
are `*prefix99-samples.json`, with client TTFT and validation status for every
completed request, including a failed check.
