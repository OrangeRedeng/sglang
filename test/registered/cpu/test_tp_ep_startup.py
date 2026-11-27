"""Exercise real argument assembly and the experimental post-load hook on CPU."""

import argparse
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from sglang.srt.hardware_backend.npu import besttp12
from sglang.srt.server_args import ServerArgs


def test_shared_fusion_payload_prep_runs_after_weight_load(monkeypatch):
    """r8: with SGLANG_NPU_EP_SHARED_FUSION on, load_model must prepare each
    DeepseekV2MoE's payload after weight processing (and must NOT raise the
    r7 fail-closed error)."""
    path = (
        Path(__file__).resolve().parents[3]
        / "python/sglang/srt/model_executor/model_runner.py"
    )
    source = path.read_text()
    assert "validate_shared_fusion_disabled" not in source
    tree = ast.parse(source)
    runner = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ModelRunner"
    )
    load = next(
        node for node in runner.body
        if isinstance(node, ast.FunctionDef) and node.name == "load_model"
    )
    load_body_text = ast.unparse(load)
    assert "prepare_fuseep_shared_payload" in load_body_text
    # The prep walks model.modules() (per-MoE), not a single global binding.
    assert "self.model.modules()" in load_body_text
    # Prep must run after the weights are loaded/processed.
    prep_index = load_body_text.index("prepare_fuseep_shared_payload")
    load_index = load_body_text.index("load_model_with_memory_saver")
    assert prep_index > load_index


@pytest.fixture(scope="module")
def parser():
    result = argparse.ArgumentParser()
    ServerArgs.add_cli_args(result)
    return result


@pytest.mark.parametrize(
    "spec", [None, "gate:w8a8_dynamic", "gate:w8a8_dynamic,shared_expert:w8a8_dynamic"]
)
def test_real_server_args_register_online_quantization(parser, spec):
    flags = ["--model-path", "unused-checkpoint", "--device", "npu"]
    if spec is not None:
        flags += ["--online-quantization", spec]
    args = ServerArgs.from_cli_args(parser.parse_args(flags))
    assert args.online_quantization == spec


@pytest.mark.parametrize("spec", [None, "gate:w8a8_dynamic"])
def test_post_load_hook_handles_control_and_requested_conversion(
    parser, monkeypatch, spec
):
    flags = ["--model-path", "unused-checkpoint", "--device", "npu"]
    if spec is not None:
        flags += ["--online-quantization", spec]
    args = ServerArgs.from_cli_args(parser.parse_args(flags))
    path = (
        Path(__file__).resolve().parents[3]
        / "python/sglang/srt/model_executor/model_runner.py"
    )
    tree = ast.parse(path.read_text())
    runner = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ModelRunner"
    )
    load = next(
        n
        for n in runner.body
        if isinstance(n, ast.FunctionDef) and n.name == "load_model"
    )
    hook = next(
        n
        for n in load.body
        if isinstance(n, ast.If) and "online_quantization" in ast.unparse(n.test)
    )
    index = load.body.index(hook)
    nodes = [hook]
    if index and isinstance(load.body[index - 1], ast.Assign):
        previous = load.body[index - 1]
        if any(
            isinstance(target, ast.Name) and target.id == "online_quantization"
            for target in previous.targets
        ):
            nodes.insert(0, previous)
    calls = []
    monkeypatch.setattr(
        besttp12,
        "convert_online",
        lambda model, requested: calls.append((model, requested)),
    )
    model = object()
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"),
        {
            "self": SimpleNamespace(server_args=args, model=model, device="npu"),
            "get_model": lambda: args,
        },
    )
    assert calls == ([] if spec is None else [(model, spec)])
