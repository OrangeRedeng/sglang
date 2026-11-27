# H09 night agent handoff

Use the published H09 checkout and the environment that runs SGLang. This
controller runs one preapproved launcher case at a time. The first smoke is
A3-C4 B0, followed by W0R only if authorized and ready. Preserve H01–H08
evidence and keep already accepted runs out of the manifest. A5 execution
requires a separate adapter. No schedule is installed.

## Prepare and inspect

1. Copy [campaign.example.json](campaign.example.json) outside the checkout.
   Fill the approved source SHA, actual hostname, allocated devices and SoC,
   model, decoder and runtime hashes, prompt and budgets. Keep `approved=false`
   until the operator approves the filled file. Use materialized checkpoint
   files and provide `H09.weights.sha256` in the model directory. This is a
   GNU sha256sum-style manifest covering every file except the inventory
   itself. Hash the inventory and pin that hash in the campaign manifest.
2. Keep the device and port lock paths on a shared mount if multiple
   controllers use the same host. The locks coordinate this controller's
   processes. Confirm the reservation with the hardware owner and check for
   unrelated NPU workloads before launch.
3. Inspect the plan and run the focused host checks before a device smoke.

```bash
h09_research=test/manual/minimax_h3_realtime/research
python3 "$h09_research/experiment_controller.py" plan --manifest "$H09_MANIFEST"
python3 "$h09_research/experiment_controller.py" run --manifest "$H09_MANIFEST" --dry-run
python3 -m pytest -q "$h09_research/test_experiment_controller.py"
```

W1 and W2 require a preapproved `h05_repack_parity` receipt. It contains
`{"result":"pass","contract_hash":"..."}`. Compute the hash using the same
`runner.digest` function on the approved contract. Record the receipt's
absolute path and SHA-256 under `capabilities.h05_repack_parity`. A new
approval uses a distinct campaign manifest and output directory.

## Run and inspect

After operator approval, set `approved=true` in the local manifest and
`H09_RUN` to an unused campaign directory outside the checkout.

```bash
python3 "$h09_research/experiment_controller.py" run \
  --manifest "$H09_MANIFEST" --output "$H09_RUN" --no-model
python3 "$h09_research/experiment_controller.py" status --campaign-dir "$H09_RUN"
python3 "$h09_research/experiment_controller.py" report --campaign-dir "$H09_RUN"
```

To use the optional supervisor, select `--supervisor-config` with a local
copy of [supervisor.example.json](supervisor.example.json). Run the supervisor
on a CPU or separate allocation. It receives a compact case and evidence
snapshot. The controller pauses on invalid or unavailable model output.
Deterministic mode remains available without a model.

Use `stop --campaign-dir "$H09_RUN"` from another terminal to request
cancellation. Use `resume --campaign-dir "$H09_RUN" --no-model` to continue a
paused, unstopped campaign. Resume preserves all charged budget and completed
results. An interrupted or unverified cleanup requires operator
reconciliation before starting another campaign. A failed case is retained
and is not retried automatically.

Keep the SQLite ledger, per-case launcher and server logs, heartbeats, device
preflight, request-correlated denoise records, MP4 hashes and report. Report
latency separately from diagnostic profiles and review AV quality manually.
