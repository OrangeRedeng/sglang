# H02 night agent handoff

Run the crop-before-cast candidate against the pinned Laser baseline
`2b0d64467d8f5eca703dd402d26055057cdc7800` on the same A3 allocation. Recover the
H01 evidence first. Keep checkpoint and LightVAE hashes, runtime, topology,
prompt, seed and four-step schedule matched. Check output parity, padded
geometry, warmed client and server latency, and peak HBM. Record both SHAs and
the artifact paths. Leave perceptual AV quality pending until reviewed.

## MindIE-SD fallback smoke

If Laser cannot load, record the Laser comparison as blocked and use the
separate MindIE-SD smoke case below. **Install MindIE-SD first** in the Python
environment that runs SGLang on A3. Use the compatible 3.2.0 wheel and preserve
the installed Torch/TorchNPU stack. If its wheel or native extension is
incompatible, stop this case and record the stack versions and import error.
[MindIE-SD installation guide](https://mindie-sd.readthedocs.io/en/stable/en/installation.html)

```bash
python3 -m pip install --no-deps --only-binary=:all: 'mindiesd==3.2.0'
python3 -c 'from importlib.metadata import version; from mindiesd import sparse_attention; assert callable(sparse_attention); print(version("mindiesd"))'
```

Use the matched model, decoder, and schedule from the original server launch.
Replace the Laser and parallelism flags with the following on an approved
eight-A3 allocation. Set `ASCEND_RT_VISIBLE_DEVICES` to its eight assigned IDs.
TP2/Ulysses4 divides the 28 local heads without the Laser-only SP8 padding.

```bash
--num-gpus 8 --tp-size 2 --sp-degree 4 --ulysses-degree 4 --ring-degree 1 \
--attention-backend ascend_sparse_attn \
--attention-backend-config '{"sparse_type":"rf_v2","precision":"bf16","sparsity":0.6,"skip_first_steps":1,"skip_last_steps":0,"dense_kwargs":{"precision":"bf16"}}'
```

Run one monolithic T2VA smoke with the saved request. Expect the first step to
use the dense executor and later steps to call MindIE-SD RFv2. When Laser cannot
import, the branch logs that Ascend FA serves the dense steps. Record the
observed dispatch and output geometry. If sparse execution fails, preserve the
failure log. A dense-only control can use `--attention-backend fa` with
`{"precision":"bf16"}` on the same eight-card topology. Keep these observations
separate from the 16-A3 Laser latency comparison.
