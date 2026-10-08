"""CPU unit test for the chat render+encode off-loop pool (branch E, E1).

The /v1/chat/completions path renders and encodes the full prompt
synchronously on the asyncio loop: serving_chat
._render_and_encode_chat_template runs a blocking apply_chat_template +
tokenizer.encode (~3.6 s per ~1M-token prompt on the AISBench box) before the
request ever reaches the scheduler. With 16 concurrent phase-2 requests the
encodes serialize on the one event loop, so the scheduler sees one tokenized
request per ~3.6 s while each forward takes ~1.4 s -- the NPU idles ~2.2 s
per request (15 idle gaps of 2.1-2.6 s, period ~3.6 s, in the phase-2 msprof
comb). E1 moves the
whole _convert_to_internal_request call off the loop, onto a
ThreadPoolExecutor owned by the TokenizerManager, at the single async
boundary OpenAIServingBase.handle_request; the flag-off path is byte-for-byte
the base call.

The value this test guards is the one a future refactor can break silently:
the offload must actually happen (the conversion runs on a pool thread, not
the loop thread) AND the loop must stay responsive while it runs (a heartbeat
keeps ticking) -- either property alone is worthless. The flag-off path must
still run the conversion inline on the loop, exactly as base. The real
wiring is pinned by source checks on the three touched files (flag, pool
creation with its gate, the hop in handle_request); the behavior is proven by
running the hop shape against a blocking conversion and checking thread
affinity and loop liveness, in the style of the D1 reordering test.

Usage:
    python -m pytest test_chat_encode_pool.py -v
    python test_chat_encode_pool.py
"""

import ast
import asyncio
import importlib.util
import os
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

try:
    from sglang.test.ci.ci_register import register_cpu_ci
    from sglang.test.test_utils import CustomTestCase

    register_cpu_ci(est_time=10, suite="base-a-test-cpu")
    _BaseCase = CustomTestCase
except ImportError:  # standalone dev-box run without the installed package
    _BaseCase = unittest.TestCase

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)

ENVIRON = "python/sglang/srt/environ.py"
TOKENIZER_MANAGER = "python/sglang/srt/managers/tokenizer_manager.py"
SERVING_BASE = "python/sglang/srt/entrypoints/openai/serving_base.py"

POOL_PREFIX = "chat-encode"


def _function_source(src, name):
    """Source text of the named method (first match in the file)."""
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            seg = ast.get_source_segment(src, node)
            if seg is not None:
                return seg
    raise AssertionError(f"function {name!r} not found")


def _load_environ_module():
    """Load environ.py straight from the file. It is stdlib-only by design,
    but importing it through the sglang package drags in the whole runtime
    (torchvision et al.), which a CPU test box need not have."""
    path = os.path.join(REPO_ROOT, ENVIRON)
    spec = importlib.util.spec_from_file_location("_e1_environ_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeTokenizerManager:
    def __init__(self, pool):
        self.chat_encode_executor = pool


class _FakeServing:
    """The E1 hop shape from OpenAIServingBase.handle_request. The real
    method's wiring is pinned by TestChatEncodePoolWiring; this copy exists
    so the behavior (thread affinity + loop liveness) is executable on a CPU
    box without a server, tokenizer or NPU."""

    def __init__(self, pool, convert_fn):
        self.tokenizer_manager = _FakeTokenizerManager(pool)
        self._convert = convert_fn

    def _convert_to_internal_request(self, request, raw_request):
        return self._convert(request, raw_request)

    async def handle_request(self, request, raw_request):
        chat_encode_pool = getattr(
            self.tokenizer_manager, "chat_encode_executor", None
        )
        if chat_encode_pool is not None:
            return await asyncio.get_running_loop().run_in_executor(
                chat_encode_pool,
                self._convert_to_internal_request,
                request,
                raw_request,
            )
        return self._convert_to_internal_request(request, raw_request)


class TestChatEncodePoolWiring(_BaseCase):
    def _source(self, relpath):
        with open(os.path.join(REPO_ROOT, relpath), encoding="utf-8") as f:
            return f.read()

    def test_flag_defined_default_on(self):
        self.assertRegex(
            self._source(ENVIRON),
            r"SGLANG_ENABLE_CHAT_ENCODE_POOL\s*=\s*EnvBool\(True\)",
        )

    def test_flag_runtime_default_on(self):
        module = _load_environ_module()
        saved = os.environ.pop("SGLANG_ENABLE_CHAT_ENCODE_POOL", None)
        try:
            self.assertTrue(module.envs.SGLANG_ENABLE_CHAT_ENCODE_POOL.get())
        finally:
            if saved is not None:
                os.environ["SGLANG_ENABLE_CHAT_ENCODE_POOL"] = saved

    def test_pool_created_gated_in_tokenizer_manager(self):
        seg = _function_source(
            self._source(TOKENIZER_MANAGER), "init_tokenizer_and_processor"
        )
        self.assertIn("self.chat_encode_executor = None", seg)
        self.assertRegex(seg, r"envs\.SGLANG_ENABLE_CHAT_ENCODE_POOL\.get\(\)")
        self.assertIn("self.tokenizer is not None", seg)
        # Text-only models only: the multimodal chat path mutates the
        # chat-template cache, which is not confined to one thread here.
        self.assertIn("not self.model_config.is_multimodal", seg)
        self.assertIn("ThreadPoolExecutor(", seg)
        self.assertIn(f'thread_name_prefix="{POOL_PREFIX}"', seg)
        self.assertIn("[chat-encode] pool engaged (branch E E1)", seg)
        # The pool reads self.tokenizer, so it must be built after the
        # tokenizer is assigned, not before it.
        self.assertGreater(
            seg.index("self.chat_encode_executor = None"),
            seg.index("self.tokenizer = get_tokenizer("),
        )

    def test_worker_count_formula(self):
        seg = _function_source(
            self._source(TOKENIZER_MANAGER), "init_tokenizer_and_processor"
        )
        self.assertIn("max(1, min(8, (os.cpu_count() or 2) // 2))", seg)

    def test_handle_request_offloads_conversion_when_pool_engaged(self):
        src = self._source(SERVING_BASE)
        self.assertIn("import asyncio", src)
        seg = _function_source(src, "handle_request")
        self.assertRegex(
            seg,
            r"chat_encode_pool\s*=\s*getattr\(\s*"
            r"self\.tokenizer_manager,\s*\"chat_encode_executor\",\s*None\s*\)",
        )
        calls = [
            n
            for n in ast.walk(ast.parse(seg))
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "run_in_executor"
        ]
        self.assertEqual(
            len(calls),
            1,
            "exactly one run_in_executor hop in handle_request",
        )
        call = calls[0]
        # hop args: (pool, self._convert_to_internal_request, request, raw_request)
        self.assertIsInstance(call.args[0], ast.Name)
        self.assertEqual(call.args[0].id, "chat_encode_pool")
        self.assertIsInstance(call.args[1], ast.Attribute)
        self.assertEqual(call.args[1].attr, "_convert_to_internal_request")
        self.assertEqual(
            [a.id for a in call.args[2:4]],
            ["request", "raw_request"],
            "the offloaded call must keep the base (request, raw_request) signature",
        )
        # Flag-off fallback: the plain inline call must still exist, so
        # =0 restores the exact base path.
        inline = [
            n
            for n in ast.walk(ast.parse(seg))
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "_convert_to_internal_request"
        ]
        self.assertGreaterEqual(
            len(inline),
            1,
            "the flag-off inline fallback call must remain in handle_request",
        )

    def test_fallback_runs_before_dispatch(self):
        # The hop must sit where the base conversion used to sit: after the
        # request logging, before the GenerateReqInput timing branch.
        seg = _function_source(self._source(SERVING_BASE), "handle_request")
        log = seg.index("log_openai_received_request")
        hop = seg.index("run_in_executor")
        dispatch = seg.index("isinstance(adapted_request")
        self.assertLess(log, hop)
        self.assertLess(hop, dispatch)


class TestChatEncodePoolBehavior(_BaseCase):
    def _blocking_convert(self, records, seconds):
        def convert(request, raw_request):
            records.append((request, threading.current_thread().name))
            time.sleep(seconds)
            return (request, "ids")

        return convert

    def test_offload_runs_on_pool_thread_and_keeps_loop_responsive(self):
        # 4 concurrent blocking conversions with 4 workers: all run at once
        # on pool threads while the loop stays alive. If the conversion ran
        # on the loop (the base bug), the heartbeat would stall and the
        # thread names would not carry the pool prefix.
        pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix=POOL_PREFIX)
        records = []
        serving = _FakeServing(pool, self._blocking_convert(records, 0.5))

        async def scenario():
            ticks = 0

            async def heartbeat():
                nonlocal ticks
                while True:
                    ticks += 1
                    await asyncio.sleep(0.02)

            hb = asyncio.ensure_future(heartbeat())
            loop_thread = threading.current_thread().name
            results = await asyncio.gather(
                *(serving.handle_request(i, None) for i in range(4))
            )
            hb.cancel()
            return results, ticks, loop_thread

        try:
            results, ticks, loop_thread = asyncio.run(scenario())
        finally:
            pool.shutdown(wait=True)

        self.assertEqual(results, [(i, "ids") for i in range(4)])
        for request, thread in records:
            self.assertTrue(
                thread.startswith(POOL_PREFIX),
                f"conversion for {request} ran on {thread!r}, not a pool thread",
            )
            self.assertNotEqual(thread, loop_thread)
        self.assertGreaterEqual(
            ticks,
            5,
            "the event loop must stay responsive while the conversions run",
        )

    def test_fallback_runs_on_loop_thread_and_blocks_it(self):
        # Pool not engaged (=0 / multimodal): the conversion runs inline on
        # the loop thread and blocks it -- exactly the base behavior, which
        # is what the flag must restore.
        records = []
        serving = _FakeServing(None, self._blocking_convert(records, 0.3))

        async def scenario():
            ticks = 0

            async def heartbeat():
                nonlocal ticks
                while True:
                    ticks += 1
                    await asyncio.sleep(0.02)

            hb = asyncio.ensure_future(heartbeat())
            loop_thread = threading.current_thread().name
            result = await serving.handle_request(0, None)
            hb.cancel()
            return result, ticks, loop_thread

        result, ticks, loop_thread = asyncio.run(scenario())

        self.assertEqual(result, (0, "ids"))
        self.assertEqual(records[0][1], loop_thread)
        self.assertLessEqual(
            ticks,
            4,
            "the flag-off path must run the conversion on the loop (base behavior)",
        )


if __name__ == "__main__":
    unittest.main()
