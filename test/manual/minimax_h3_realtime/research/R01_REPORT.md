# R01 MiniMax-H3 merged baseline

## Integration base

R01 descends from clean H09 at **29bfac48b73c80ec2e1e874d6669d84661941d8c**. Live source checks found Allor-maker at `1d222dac2b9a08e2a69cce3c243fef59d52bde69` and upstream main at `87a99e7717b725ffa754c46d4f2ea6aad4bbdb44` (61 commits past the handoff snapshot). The upstream merge base with H09 is `38ae340221b4b65859c68e85d9e8ffb6b9c6f167`.

| Path | Decision |
|---|---|
| Encoder callback | Allor-maker's `1f5313a593` gates the shared callback. H04 already excludes the H3 NPU callback and traces decode phases. Keep its narrower gate. |
| AdaLN | Apply the preparation-failure cleanup from upstream `5db474a006` (#44806). The patch covers release of cache slots after a failed embedding or TP setup and carries its original regression cases. |
| Graph | Upstream #52522 concerns single-GPU breakable graphs. The R01 launch pins eager DiT execution, matching the A3 baseline. |
| Laser and MindIE-SD | Keep H09 dispatch, BF16 range compensation and dense fallback. Carry forward the optional crop-order and tensor-fingerprint portions of the interrupted H09 patch. H08 ERR00100 still requires dispatch isolation. |

## Night evidence

| Case | Status for R01 |
|---|---|
| H02-CROP | The operator confirmed merged FlashGen plus runtime LoRA. Reject the 67.54 s comparison. H02 and H05 share the crop-order source blob. |
| H03 STEP3 and H04 VAE | Preserve the trace and decode evidence. The reports lack complete launch provenance. Establish checkpoint and adapter state from actual argv and logs before using either as a paired baseline. H03 host-time attribution cannot establish an allocator or HCCL cause. |
| T8 / DEV-T8 | The supplied records describe runtime Turbo LoRA with alpha 8. The 84.80 s result verifies the 8-step schedule, with base checkpoint identity unresolved. Exclude it from the offline-merged baseline. |
| H07 FFN | F100 matched B0 media in the supplied report. F75 changes output and the timing difference falls within sample variability. Verify the underlying launch records. |
| H09 B0/W0R | The deterministic controller reports equal media. Published controller source requests a merged FlashGen launch. Retrieve its launch receipts before treating the night run as verified against the new contract. |
| H05 B0/W0R and LLLL | Preserve byte-identical decoded video and PCM with matching step/forward counts for the pinned pair. Intermediate parity is pending. LLLL's visible seam disqualifies it for the accepted quality baseline. |

The raw launch records and device artifacts are on the operator's A3 host under `/home/aibot/h3-nightly-20261127-201536`. The audit above uses supplied summaries and available source.

## Offline checkpoint preparation

Use local immutable files and a clean original MiniMax-H3 FL2VA snapshot. Keep the original base, each merge output, and each verification receipt separate. Before merging, record full adapter revision and content hash, merge tool revision, alpha, rank and user scale. Inventory the reference first. Create an empty output root for each merge. Quantization comes after the merge.

The publisher's [FlashGen merge tool](https://huggingface.co/Beidouqixing/minimax-h3-4step-lora-flashgen/blob/main/merge_lora_ckpt.py) maps native DiffSynth PEFT `.default` pairs directly onto the raw checkpoint and fails on missing targets. Pin the reviewed script by content hash. For its native FlashGen rank-64 export, alpha 64 and user scale 1 give an effective scale of 1. Use local paths inside the A3 container or offline merge environment.

```bash
: "${H3_CLEAN_ROOT:?Set the immutable original checkpoint root}"
: "${H3_FLASHGEN_ADAPTER:?Set the pinned local FlashGen adapter}"
: "${H3_FLASHGEN_MERGER:?Set the pinned local publisher merge script}"
: "${H3_FLASHGEN_OUT:?Choose a new merged checkpoint root}"
mkdir "$H3_FLASHGEN_OUT"
python3 "$H3_FLASHGEN_MERGER" "$H3_CLEAN_ROOT/FL2VA" "$H3_FLASHGEN_ADAPTER" \
  "$H3_FLASHGEN_OUT" --scale 1 --base-schedule 1,0.7,0.4,0.15,0
```

The publisher links untouched components. Materialize those links into the new output before inventorying it. Confirm the result contains the original components and the merged transformer shards. The clean root and merge output both contain `FL2VA/model_index.json`.

Turbo8 requires a separate offline merge from that same clean base. Use the exact original FL2V 8-step release file and a reviewed merger that handles its specific PEFT layout and metadata alpha. The FlashGen merger requires native-layout `.default` keys and a single rank. Do not apply it directly to a differently packed Turbo file. Record each effective alpha/rank factor, the raw target mapping and per-target merge coverage. The night report's alpha 8 belongs to its specific adapter. Block Turbo8 until there is a verified merge and manifest. [LightX2V release](https://huggingface.co/lightx2v/Minimax-h3-Turbo)

Create an inventory outside the model directory. Use the resulting `files` and `checkpoint_sha256` in the contract below. For the clean reference, `clean_base_sha256` equals its `checkpoint_sha256`. Use that same clean digest in both merged manifests. The decoder is either `{"kind":"native"}` or `{"kind":"lightvae","path":"/absolute/path/to/lightvae.safetensors","sha256":"<file SHA-256>"}`. Use separate manifests for separate decoders.

```bash
research=test/manual/minimax_h3_realtime/research
python3 "$research/checkpoint_contract.py" inventory "$H3_FLASHGEN_OUT" > "$H3_FLASHGEN_OUT.inventory.json"
sha256sum "$H3_FLASHGEN_ADAPTER" "$H3_FLASHGEN_MERGER"
```

```json
{
  "version": 1,
  "family": "flashgen4",
  "model": "/absolute/path/to/flashgen-root",
  "variant": "fl2va",
  "dtype": "bf16",
  "files": {},
  "checkpoint_sha256": "<digest from inventory>",
  "clean_base_sha256": "<digest from native inventory>",
  "merge": {
    "adapter": "<release revision and filename>",
    "adapter_sha256": "<adapter content SHA-256>",
    "recipe": "<record exact command and rank/effective scaling>",
    "tool_revision": "<pinned revision and script SHA-256>",
    "alpha": 64,
    "scale": 1
  },
  "decoder": {"kind": "native"},
  "schedule": {"steps": 4, "rungs": [1000, 700, 400, 150], "video_shift": 12, "audio_shift": 3}
}
```

For `native50`, use `merge: null`, the native decoder and schedule `{"steps":50,"rungs":[],"video_shift":12,"audio_shift":3}`. For `turbo8`, use 8 steps, rungs `[1000,875,750,625,500,375,250,125]` and shifts 6/3. Verify on the host/container filesystem that will run the launcher. Verification reads each file's content and records file identity, size and timestamps. Subsequent launches check the metadata against the receipt. Keep weights and receipts immutable. Reverify after moving or remounting the checkpoint if the file identity changes.

```bash
export H3_CHECKPOINT_MANIFEST=/absolute/path/to/flashgen4.json
export H3_CHECKPOINT_RECEIPT=/absolute/path/to/flashgen4.receipt.json
python3 "$research/checkpoint_contract.py" verify "$H3_CHECKPOINT_MANIFEST" "$H3_CHECKPOINT_RECEIPT"
```

## Registered cases

Use the launcher inside the allocated A3 container. Pin the prompt, seed, device IDs and output prefix. For the 16-A3 FlashGen pair, set `H3_NUM_GPUS=16 H3_LASER16=1 H3_PIN_CPU_MEMORY=false`.

| Controller adapter | Shell case | Purpose |
|---|---|---|
| `R01.CLEAN4.B0`, `R01.CLEAN4.W0R` | `B0`, `W0R` | Timing and AV checks, FlashGen4. |
| `R01.PARITY.B0`, `R01.PARITY.W0R` | `B0`, `W0R` with `H3_WINDOW_PARITY=1` | Separate tensor fingerprints and decoded video/PCM parity. |
| `R01.CROP.BEFORE`, `R01.CROP.AFTER` | `B0` with `H3_LASER_CROP_BEFORE_CAST=1` or `0` | Same source and checkpoint, BF16 range compensation preserved and kernel format to be confirmed on device. |
| `R01.QUALITY.NATIVE`, `R01.QUALITY.FLASHGEN`, `R01.QUALITY.TURBO` | `B0` with the corresponding manifest | Native full-step and separate few-step quality references. Run paired decoder manifests and require exact input and latent hashes before rating decoder quality. |
| `R01.HIST.OFF`, `R01.HIST.ON` | `HIST-OFF`, `HIST-ON` | Native50, 5 s / 124 frames / 8× A3, historical Cache-DiT controls. |

```bash
launcher="$research/run_a3_windowing.sh"
H3_WINDOW_PARITY=1 H3_REPEATS=1 bash "$launcher" --inside "$PWD" B0 /absolute/output/r01
H3_WINDOW_PARITY=1 H3_REPEATS=1 bash "$launcher" --inside "$PWD" W0R /absolute/output/r01
python3 "$research/runner.py" parity /absolute/output/printed-B0-dir \
  --candidate /absolute/output/printed-W0R-dir
```

For the decoder pair, run B0 twice with tensor parity enabled. Use two FlashGen4
manifests pointing to the same checkpoint, with a different decoder in each.
Run the parity command with the two B0 artifact paths and `--decoder-only`.
The two denoising runs must produce identical latent hashes at every step.
Review both decoded outputs independently.

Compare ordinary timing launches with the same prompt, seed, source, device placement and output policy. Use `H3_PROFILE=1` for stage, output and allocator evidence. The profiled request has separate artifacts and is excluded from the timing median. The allocator snapshot describes the output rank's PyTorch allocator, with peaks since its last reset. The before/after `npu-smi` files describe device-wide snapshots. Per-request downloads are recorded separately in both modes.

## Historical regression

The published eight-NPU sequence is 55.07 → 59.69 → 55.21 s. The first gap is 4.62 s. These are 50-step reports with historical server timing. The local Git range `61b67316d828..88151fa62c61` contains the following candidates. `3400b40b343e` is the subsequent #49353 merge.

| Candidate | What to isolate on the same stack |
|---|---|
| `11a5f34c7c` #49826 | Cache-DiT input preservation and actual block executions, cache off/on. |
| `800f9b3051` #44753 | RoPE layout, concrete NPU dispatch and copies. |
| `40c2faa94d` #50391 | Residual/norm fallback and precision with the same tensors. |
| `f92ff40b13` #49762 | Resolved quality/cache schedule. HIST omits explicit quality to enable the historical generic policy. |

Use the same clean weights, decoder, stack and prompt in the old revision worktrees. Start with the August and pre-#49353 endpoints, then the #49353 merge. Historical launchers predate the R01 controls. Use the H01 regression runbook from the Round 1 handoff for those snapshots and retain original argv and cache evidence. R01 HIST cases reproduce the requested settings on the new source and record rank-0 CFG forwards and each physical block-stack execution. They do not establish the historical causal range. #50117 precision and #51818/#53106 stack skew require independent controls. An actual rerun and fixed-stack compatibility are pending operator access.

Historical source reports: [#33569](https://github.com/sgl-project/sglang/pull/33569) and [#49353](https://github.com/sgl-project/sglang/pull/49353).

## Validation

The 35 CPU research tests passed, including synthetic FlashGen4, Turbo8 and
native launch argv, receipt drift, schedule rejection, endpoint adapter checks,
and parity evidence rejection. The sparse benchmark dry run, controller plan,
Python compilation, repo-selected Ruff rules and Bash syntax checks passed.
Device timing, physical Cache-DiT counts, latent and media parity, and quality
acceptance require the operator's weights and A3 allocation.

### Phase 2.1

Turbo8 manifests now require a references object with native50 and flashgen4 entries, each containing absolute manifest and receipt paths. Verify the Native50 and FlashGen4 files before verifying Turbo8. The three content inventories must be distinct, both adapted releases must name the verified Native50 base, and their adapter hashes must differ. The offline merge recipe remains the operator's provenance record. Preserve the native FP32 islands during any subsequent online quantization.

Timing adapters must explicitly report measured scheduler_updates, dit_evaluations, profiled=false and diagnostic=false. The launcher and controller exclude tensor parity and physical block-count instrumentation from unprofiled aggregates. The standalone warm-server sweep requires a distinct, unprofiled server execution record per client invocation. Intermediate parity and runtime LoRA guards remain active.
