# Native H3 chunk generation

Launch each endpoint through [the R01 verified launcher](R01_REPORT.md) with
`H3_SERVE_ONLY=1`. Use the manifest digest printed by verification in both
deployment records and the request model ID. The client checks the served
identity and empty runtime adapter state before submitting work. For the
two four-device anchor servers, launch `B3` with distinct `H3_PORT` and
`H3_B3_DEVICES` values.

`h3_chunks.py` performs actual `/v1/videos` requests against existing H3 servers.
It implements **RGB endpoint-conditioned continuation and parallel generation**.
It does not implement AV latent-tail continuation, global temporal RoPE, or an
exact factorization of bidirectional H3. FlashGen was trained for T2VA; its FL2VA
transfer and chunk boundaries need quality validation on the unchanged adapter.

## Sequential continuation

Start the existing verified H3 FL2VA-partition server with your current FlashGen,
attention, MXFP8 and LightVAE configuration. T2VA and endpoint-conditioned FL2VA
share that partition. Supply its actual immutable revisions in this JSON:

```json
{
  "mode": "continuation",
  "duration_s": 15,
  "window_frames": 124,
  "overlap_frames": 1,
  "deployment": {
    "family": "flashgen4",
    "manifest_sha256": "REPLACE_WITH_VERIFIED_MANIFEST_SHA256"
  },
  "endpoints": [{
    "url": "http://127.0.0.1:30010",
    "devices": [0,1,2,3,4,5,6,7],
    "deployment": {
      "family": "flashgen4",
      "manifest_sha256": "REPLACE_WITH_VERIFIED_MANIFEST_SHA256"
    }
  }],
  "request": {
    "model": "h3-research-REPLACE_WITH_VERIFIED_MANIFEST_SHA256",
    "prompt": "An uninterrupted shot of a sailboat crossing a calm lake, with continuous natural sound.",
    "num_inference_steps": 4,
    "seed": 1101,
    "flow_shift": 12.0,
    "audio_flow_shift": 3.0,
    "quality": "lossless",
    "enable_cache_dit": false,
    "target": {"short_edge": 768, "aspect_ratio": "16:9"}
  }
}
```

Keep the deployed FlashGen sigma schedule; a four-step request alone does not
install it. The orchestrator preserves all request fields other than the
window geometry, prompt override, seed, task, conditions, and output count.
For a 16-NPU Laser endpoint, list device IDs 0–15 in the continuation config.
Start the server with `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15`,
`SGLANG_H3_LASER_SP8_PAD_HEADS=1`, and
`--num-gpus 16 --tp-size 2 --sp-degree 8 --ulysses-degree 8 --ring-degree 1 --attention-backend laser_attn`.

```bash
python test/manual/minimax_h3_realtime/research/h3_chunks.py \
  /tmp/chunks.json --output /tmp/h3-chunks --plan-only
python test/manual/minimax_h3_realtime/research/h3_chunks.py \
  /tmp/chunks.json --output /tmp/h3-chunks
```

The output directory must be new; ffmpeg/libx264/FFV1 and ffprobe are required.
`expected_canvas` defaults to `[1344, 768]`; change it explicitly for a different
canvas. Outputs that follow a mismatched anchor aspect ratio are rejected.
Server and orchestrator must share the output filesystem because native
conditioning receives `file://` images. Endpoints are externally managed.

For the default 15-second request, the planner commits exactly 362 unique frame
positions at 24 FPS. Each request follows native `17n+5` frame geometry and
4–15-second bounds. A short final suffix is generated inside a longer legal
window and trimmed. The first window is T2VA unless `first_frame` is supplied.
Each subsequent window receives the previous generated frame at its global
start as its first-frame condition. The sampled overlap is discarded. Default
one-frame overlap shares the previous final frame and drops the next first
frame. Larger overlaps are available for sequential experiments.

Audio is trimmed on the **same time intervals** as video, then concatenated.
No audio crossfade or slow-motion metadata is used. Audio is generated separately
in each window; uninterrupted speech/music, AV sync and motion across seams
remain approximate and require listening/viewing checks.

## Parallel anchor-conditioned chunks

Set `mode` to `anchor_parallel`, keep `overlap_frames: 1`, and provide:

```json
"anchor_source": "user",
"anchors": ["/shared/frame0.png", "/shared/frame1.png", "/shared/frame2.png", "/shared/frame3.png"]
```

Provide one anchor for each boundary, including the beginning/end (four for the
three-window default). Intermediate windows receive native first/last-frame
conditions; neighboring chunks share the same boundary image and commit it once.
A padded final request omits the last-frame condition because that sampled
endpoint lies outside the committed output interval. The final supplied anchor
is consequently unused in that case.

Start multiple **separate** H3 servers on disjoint NPU groups; for example two
servers with four visible NPUs and Ulysses degree four each. Preserve your
verified launch settings and assign separate ports. Replace `endpoints` with
their URLs and physical device lists, e.g. `[0,1,2,3]` and `[4,5,6,7]`, each with
identical deployment metadata. The scheduler dispatches at most one active chunk
per endpoint/group and assembles results in timeline order. It rejects overlapping
device lists and budgets above eight NPUs. Declared assignments cannot prove
how an external server was actually launched: verify its topology and residency.

Use `anchor_source: "oracle"` for anchors extracted from a pre-generated video;
results are labeled oracle. Precomputed model-generated draft anchors are also
oracle for timing purposes. Automatic timed anchor generation is not implemented
and `timed_draft` is rejected. There is no claim that independent anchors provide
coherent dialogue or motion. Anchor preparation by a model must be added to the
critical path before a production system speedup can be claimed.

Optional `continuation_prompts` contains one prompt per window. Seeds are
`base_seed + chunk_index`; this is deterministic per window, **not** global noise
or exact overlap consistency. No unrelated-request throughput is reported as
single-video acceleration.

## Artifacts and validation

The directory retains `config.json`, `windows.json`, every request, server job
response, raw AV chunk, generated boundary PNG, committed lossless intermediate,
`output.mp4`, and `result.json`. Client E2E includes requests, downloads, boundary
extraction, trimming and final encoding/mux. Warm your servers separately.
Native executed NFE is explicitly `unverified` because the HTTP response does
not expose that counter; requested steps and server responses are retained.
Do not supply the requested value as measured NFE to the stricter campaign runner.

Requests are polled to completion with a timeout; cancellation/failure triggers
best-effort DELETE for outstanding jobs. Chunk exceptions cancel dependent work.
The CLI preserves a failure record and partial artifacts. Output checks require
AV streams, exact native frame counts, 24 FPS, consistent canvases, and final AV
durations within 0.1 seconds. They do not establish perceptual continuity.

```bash
python -m pytest -q test/manual/minimax_h3_realtime/research/test_h3_chunks.py
```

Host tests mock H3/media execution and verify native request payloads, previous
frame reuse, parallel dispatch, boundary trimming, ordering, and cancellation.
Real H3 inference and ffmpeg mux correctness require an environment with the
model and media tools; no Ascend speedup has been measured here.
