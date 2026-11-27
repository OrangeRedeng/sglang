# H01 night agent handoff

Recover the original monolithic client result and six-chunk repeat 1. Use
`feat/minimax-h3-ascend-realtime-research` for the client. The server baseline is
`2b0d64467d8f5eca703dd402d26055057cdc7800` from the global/local-windowing branch.
Record both SHAs. Run inside the A3 container where the client and server share
output paths. Start with recovery, which only lists existing files.

```bash
h3_artifacts=/home/swx1199799
h3_nightly=test/manual/minimax_h3_realtime/research/h01_nightly.py
python3 "$h3_nightly" --artifacts "$h3_artifacts"
```

Read any recovered measured samples, including `samples.jsonl` if the aggregate
is missing. Keep warmup, client E2E and summed server times distinct. Inspect
server logs for each request and record the evidence path and its counting
scope. Preserve the original clips and results.

Only run a missing case on an approved 16-A3 allocation. Verify the server
commit, merged checkpoint and decoder hashes against the recorded deployment.
Use the merged four-step FlashGen checkpoint, LightVAE, TP2/Ulysses8/Ring1,
Laser with padded heads, rungs 1000/700/400/150 and cache disabled. Use the
original launch record, including a separate rendezvous port and no runtime
LoRA. If the config, match or allocation cannot be established, record the
blocker and stop. The client reads the existing three-chunk config for the
endpoint and request.

```bash
# Select exactly one missing case. b0 records one warmup and one measured sample.
# six-1 uses the already-warmed matched server and records one six-chunk result.
h3_case=b0
h3_run=$(mktemp -d "$h3_artifacts/H01-$h3_case-$(TZ=Europe/Moscow date +%Y%m%d-%H%M%S)-XXXXXX")
git rev-parse HEAD > "$h3_run/client-sha.txt"
set -o pipefail
python3 "$h3_nightly" --artifacts "$h3_artifacts" --run "$h3_case" \
  --output "$h3_run/result" 2>&1 | tee "$h3_run/client.log"
```

Store the server launch manifest and logs with the result. Report raw timings,
artifact paths, counted execution evidence, and AV quality status. Geometry
checks leave perceptual quality pending. Do not start another handoff.
