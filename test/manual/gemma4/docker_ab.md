# Three launch tests: SGLang and a separate vLLM container

Run each server separately and stop it before starting the next. Use host
networking for both containers with URL `http://127.0.0.1:30014`. With a shared
bridge network, set `GEMMA_BENCH_URL=http://<sglang-container-name>:30014` in the
benchmark container. Mount the tokenizer and `bench_ab.sh` there.

In the SGLang repository directory, set:

```bash
export GEMMA_ROOT="$PWD"
export GEMMA_ASSISTANT_MODEL=/home/test1/models/gemma-4-26B-A4B-it-assistant
```

| Test | Serving command |
| --- | --- |
| 1. Current reference | `bash test/manual/gemma4/serve_ab.sh metadata` |
| 2. Matched eager MTP control | `bash test/manual/gemma4/serve_ab.sh mtp-baseline` |
| 3. Serial QKV + MTP | `bash test/manual/gemma4/serve_ab.sh mtp` |

All three use serial QKV. Experimental tiled QKV and RoPE/KV paths were removed
after target-NPU launch crashes. No tiled validation report or row flag is needed.

Test1 keeps NPUs2/3, TP1/DP2, queue1, FIA, context8192, max-running64, token
budgets8192, memory0.92, supplied graph buckets, and delayer0.5/50ms/4/cap6.
Tests2/3 both disable graphs, overlap and the prefill delayer and use memory0.82.
The delayer requires overlap scheduling. Compare2 against3
to isolate MTP. Test3 additionally uses the matching assistant with NEXTN,
steps3, top-k1 and draft width4. Validate fixed-prompt greedy outputs/logprobs,
long-prefix SWA and shared KV before accepting MTP performance.

Append `--print` to inspect a server command without launching. Capture logs
with `2>&1 | tee test1.server.log`, changing the filename per test.

Run the matching command in the vLLM container while that server is active:

```bash
bash test/manual/gemma4/bench_ab.sh metadata 1
bash test/manual/gemma4/bench_ab.sh mtp-baseline 1
bash test/manual/gemma4/bench_ab.sh mtp 1
```

The helper preserves random prefix250/input1100/output100, C24, RPS87,
1600 prompts, temperature0, ignore-EOS and the requested latency percentiles.
Change the final argument to2/3 for repetitions without restarting serving.
It saves raw results and before/after server snapshots. MTP acceptance statistics
are cumulative where exposed. This screen does not establish the eight
recommendation/RAG targets; retain the full acceptance requirements in README.md.
