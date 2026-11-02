# MegaMoE parity and matched TTFT retest

The supplied fix-check run at 0b21b037e5 confirms that the 768 MB HCCL buffer
fix reaches MegaMoE on all 16 ranks. MegaMoE still fails the unchanged
atol=rtol=0.01 teacher-forced comparison: 98/1349 logprobs fail, maximum absolute
error 0.5454654694. No numerical runtime repair is established by these logs.
The three DCP I/O features and the cached 1M baseline passed this run.

The current TP16 reference and EP16 candidate change expert partitioning as well
as fusion. W4A8 dynamically quantizes the W2 input in each expert partition;
TP16 and EP16 therefore need an unfused EP16 control to distinguish partitioning
from MegaMoE-specific numerical differences. This is a diagnostic hypothesis,
not a demonstrated explanation of all 98 failures. Do not change tolerances,
drop compensation biases, or declare the kernel repaired without A3 evidence.

## Short experiment

Run the updated host launcher:

```bash
REPO=/home/swx1199799/megamoe_test/sglang
docker cp "sglang-a3:$REPO/benchmark/prefill_1m/start_a3_retest.sh" /tmp/start_a3_retest.sh
screen -dmS "a3-megamoe-$(date +%Y%m%d-%H%M%S)" bash /tmp/start_a3_retest.sh megamoe-measure
```

This profile runs three variants using the same full GLM5.2 W4A8 checkpoint:

- `moe-baseline`: ordinary TP16, separate shared expert.
- `moe-ep-baseline`: TP16/EP16, unfused DeepEP normal dispatch and separate shared expert.
- `megamoe`: TP16/EP16, FuseEP mode 3, separate shared expert, 768 MB HCCL buffer.

Each variant retains the ordinary TP16 parity check. The EP control also saves
its complete responses, allowing a direct EP-control/MegaMoE comparison after
the run. A failure to launch the EP control is recorded independently.

Timing settings: native streaming client, exact 1024 input tokens, one output
token, DCP1, context limit 8192, local KV cap 8192, chunk 4096, memory fraction
0.88, two requests at concurrency 1 and 2. Each point has three repetitions,
with separate cold and 99%-prefix regimes and one warmup per regime. The cache
is flushed and its prefix primed before each cached repetition. Prefix preparation
is excluded from TTFT. TPOT is undefined for one output token. These are short
MoE timings, not 1M-context timings and not AISBench measurements.

Allow approximately 45-90 minutes including full-weight reloads, with a two-hour
budget. This is a planning estimate, not verified completion time. One runner
may own `/tmp/sglang-bde-overnight.lock`; do not delete the lock to bypass it.

`--diagnostic-megamoe` explicitly permits timing only an activated mode-3
candidate after parity failure. Its case remains `fail`, quality remains
`unverified`, and timing rows/files say `unqualified-parity`; runner exit 1 is
expected while parity remains unresolved. Ordinary benchmark qualification gates
remain unchanged. Other failed/inactive candidates do not receive this exception.

## Results and logs

Read `results/SUMMARY.md` for individual serving cases and exact-token TTFT mean,
P90, E2E, throughput and cold/cached rows. Raw samples retain token counts and
cache-page validation. `parity.json` saves total comparisons/failures, maximum
error, and the first 32 failing positions and token identities. Original response
files retain all logprobs.

The result archive includes full supplied server logs, checkpoint metadata,
resolved server arguments, source diff, runtime identities and worker protocol
responses. Empty files are removed with a recorded path manifest. An empty
worker stderr stream no longer appears to be a missing worker result.

## Acceptance still pending

Local CPU tests establish reporting, argument wiring and preservation of failed
qualification. They do not establish A3 numerical correctness, performance,
HCCL stability or memory safety. Compare the unfused EP control with MegaMoE on
identical teacher-forced inputs, then isolate the first divergent MoE layer if
needed. A candidate must pass the existing TP16 quality policy before its timings
are admitted as a qualified optimization. Earlier unrelated failure families
from the full overnight matrix are not repaired by this reporting patch.
