# H07 night agent handoff

Use the published H07 server. Keep the approved topology, model and decoder
hashes, native monolithic request, seed and four-step sigma grid fixed within
each comparison. Run each optimization separately with the other two disabled.

## Selective FFN

Reuse the merged FlashGen exports from [H05_NIGHTLY.md](H05_NIGHTLY.md). Run B0
with `H3_FFN_ACTIVE_FRACTION=1.0`, then `0.75`, then `0.5` if parity and
budget permit. This launcher disables Cache-DiT. TP2 also excludes the AdaLN
memo. Change only the active fraction within this pair.

```bash
: "${H07_OUTPUT_PREFIX:?Set the shared result prefix}"
h07_launcher=test/manual/minimax_h3_realtime/research/run_a3_windowing.sh
H3_FFN_ACTIVE_FRACTION=1.0 bash "$h07_launcher" --inside "$PWD" B0 "$H07_OUTPUT_PREFIX-full"
H3_FFN_ACTIVE_FRACTION=0.75 bash "$h07_launcher" --inside "$PWD" B0 "$H07_OUTPUT_PREFIX-f75"
```

Use separate `H3_PROFILE=1` runs with `SGLANG_PROFILE_RECORD_SHAPES=1` for
`h3_dit_attention_layer_*` and `h3_ffn_fc1_rows_*`. Count real layer attention
entries and compare the executed FC1 shapes by rank and step. Include protected
rows and gather/scatter overhead. Review motion, detail and AV coherence before
trying quantized DiT under its own matched baseline.

## AdaLN memo

Start only if a resident eager TP1 deployment fits HBM. One possible eight-A3
layout is TP1/Ulysses8/Ring1 with Laser. Use direct serve with the saved
launch flags adjusted for that topology. Compare
`SGLANG_H3_NPU_ADALN_MEMO=0` against its default using clean server starts,
Cache-DiT disabled and full FFN. Use repeated identical schedules plus one
changed schedule. In separate profiles, look for `h3_adaln_memo_hit` and
`h3_adaln_memo_miss` and record HBM. A server warmup may populate the memo
before the first measured request. Record eligibility failures instead of
inferring misses from absent markers.

## Adaptive Cache-DiT

Use direct serve with full FFN, memo disabled, compile/graph disabled, and
the same native fixed-grid request. Omit the explicit `quality` field from
both control and candidate so the generic Cache-DiT policy can apply. Start
with `enable_cache_dit=false`, then use the following independent request
overrides after confirming the runtime CFG mode and cache context reset.

```json
{
  "enable_cache_dit": true,
  "cache_dit_params": {
    "Fn_compute_blocks": 1,
    "Bn_compute_blocks": 0,
    "max_warmup_steps": 2,
    "residual_diff_threshold": 0.12,
    "max_continuous_cached_steps": 1,
    "enable_taylorseer": false
  }
}
```

This is a screening configuration. Record resolved cache settings, actual
block/step execution, and library cache-hit and residual-delta data if the
installed build exposes it. Record missing counters explicitly. Keep profiling
and warmed latency separate. Preserve AV media and stop on parity, quality or
operator failure. Defer dynamic row freezing and combined modes until the
individual screens pass.
