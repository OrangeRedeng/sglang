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
import random
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
SERVING_CHAT = "python/sglang/srt/entrypoints/openai/serving_chat.py"
CHAT_ENCODE_MEMO = "python/sglang/srt/managers/chat_encode_memo.py"

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


def _load_memo_module():
    """Load chat_encode_memo.py straight from the file. It is stdlib-only by
    design (threading + collections + time), so it is executable on a CPU
    box without the sglang runtime."""
    path = os.path.join(REPO_ROOT, CHAT_ENCODE_MEMO)
    spec = importlib.util.spec_from_file_location("_e2a_chat_encode_memo", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeTokenizerManager:
    def __init__(self, pool, memo=None):
        self.chat_encode_executor = pool
        self.chat_encode_memo = memo


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


class TestChatEncodeWiring(_BaseCase):
    """E2a wiring: the env flag, the memo hook, the memo module contract."""

    def _source(self, relpath):
        with open(os.path.join(REPO_ROOT, relpath), encoding="utf-8") as f:
            return f.read()

    def test_e2a_memo_env_defined_default_on(self):
        self.assertRegex(
            self._source(ENVIRON),
            r"SGLANG_CHAT_ENCODE_MEMO\s*=\s*EnvBool\(True\)",
        )
        module = _load_environ_module()
        saved = os.environ.pop("SGLANG_CHAT_ENCODE_MEMO", None)
        try:
            self.assertTrue(module.envs.SGLANG_CHAT_ENCODE_MEMO.get())
        finally:
            if saved is not None:
                os.environ["SGLANG_CHAT_ENCODE_MEMO"] = saved

    def test_e2a_memo_created_gated_after_pool(self):
        seg = _function_source(
            self._source(TOKENIZER_MANAGER), "init_tokenizer_and_processor"
        )
        self.assertIn("self.chat_encode_memo = None", seg)
        self.assertRegex(seg, r"envs\.SGLANG_CHAT_ENCODE_MEMO\.get\(\)")
        self.assertIn("ChatEncodeMemo(", seg)
        self.assertIn("[chat-encode] memo engaged (branch E E2a)", seg)
        self.assertGreater(
            seg.index("envs.SGLANG_CHAT_ENCODE_MEMO.get()"),
            seg.index("self.chat_encode_executor = None"),
        )

    def test_e2a_memo_hook_in_serving_chat(self):
        src = self._source(SERVING_CHAT)
        seg = _function_source(src, "_render_and_encode_chat_template")
        self.assertIn(
            "getattr(self.tokenizer_manager, \"chat_encode_memo\", None)", seg
        )
        self.assertIn("self._memo_render_and_encode(", seg)
        mseg = _function_source(src, "_memo_render_and_encode")
        for call in (
            "memo.get_full(",
            "memo.begin_assembly(",
            "memo.claim_creation(shape_key)",
            "memo.wait_family(shape_key, _MEMO_BOOTSTRAP_TIMEOUT_S)",
            "memo.prefix_ids(",
            "memo.commit_full(",
            "memo.store_full(",
        ):
            self.assertIn(call, mseg)
        self.assertIn("_MEMO_BOOTSTRAP_TIMEOUT_S = 30.0", src)
        anchor_seg = _function_source(src, "_memo_anchor")
        self.assertIn("MIN_PREFIX_CHARS", anchor_seg)
        shape_seg = _function_source(src, "_family_shape_key")
        self.assertIn("MIN_PREFIX_CHARS", shape_seg)
        self.assertIn("apply_chat_template", mseg)
        self.assertIn("self._memo_anchor(messages)", mseg)

    def test_e2a_memo_module_has_contract(self):
        src = self._source(CHAT_ENCODE_MEMO)
        self.assertIn("MIN_PREFIX_CHARS = 4096", src)
        self.assertIn("MAX_FULL_ENTRIES = 16", src)
        module = _load_memo_module()
        for name in ("ChatEncodeMemo", "lcp_len"):
            self.assertTrue(hasattr(module, name), f"missing {name}")
        memo = module.ChatEncodeMemo()
        for method in ("claim_creation", "wait_family", "begin_assembly",
                       "prefix_ids", "commit_full", "store_full", "get_full"):
            self.assertTrue(callable(getattr(memo, method, None)), f"missing {method}")


class _MemoCase(_BaseCase):
    @classmethod
    def setUpClass(cls):
        cls.memo_module = _load_memo_module()

    def make_memo(self, **kw):
        return self.memo_module.ChatEncodeMemo(**kw)

    def fake_ids(self, text):
        return [ord(c) % 256 for c in text]


class TestChatEncodeMemoBehavior(_MemoCase):
    """The memo flow mirrors _memo_render_and_encode: begin_assembly first
    (assembled from the family, no render), else the bootstrap
    (claim_creation / wait_family), else a full render followed by
    commit_full (which creates the family on the first long prompt)."""

    PRELUDE = "<|system|>sys<|user|>"
    POSTLUDE = "<|assistant|>"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.SHARED = "P" * 100

    def encode(self, memo, text):
        return self.fake_ids(text)

    def make_family(self, memo, content, shape_key="sk"):
        rendered = self.PRELUDE + content + self.POSTLUDE
        ids, result = memo.commit_full(
            shape_key, content, rendered, "kw", lambda t: self.fake_ids(t)
        )
        return rendered, ids, result

    def test_full_hit_returns_stored_ids(self):
        memo = self.make_memo(min_prefix_chars=16)
        memo.store_full("k", "ABCD" * 10, [1, 2, 3])
        self.assertEqual(memo.get_full("k"), ("ABCD" * 10, (1, 2, 3)))
        self.assertIsNone(memo.get_full("other"))

    def test_family_create_prefix_ids_plus_postlude(self):
        memo = self.make_memo(min_prefix_chars=16)
        content1 = self.SHARED + "TAIL-ONE"
        rendered1, ids1, result = self.make_family(memo, content1)
        self.assertEqual(result, "family")
        self.assertEqual(
            ids1,
            self.fake_ids(self.PRELUDE + content1) + self.fake_ids(self.POSTLUDE),
        )

    def test_sibling_assembles_without_render(self):
        memo = self.make_memo(min_prefix_chars=16)
        content1 = self.SHARED + "ONE"
        self.make_family(memo, content1)
        content2 = self.SHARED + "TWO"
        assembled = memo.begin_assembly("sk", content2)
        self.assertIsNotNone(assembled)
        rendered_prefix, cut, postlude, version = assembled
        self.assertEqual(rendered_prefix, self.PRELUDE + self.SHARED)
        self.assertEqual(cut, len(self.SHARED))
        self.assertEqual(postlude, self.POSTLUDE)
        self.assertEqual(memo.stats()["family_prefix_chars"], len(self.SHARED))
        rendered2 = rendered_prefix + content2[cut:] + postlude
        prefix_ids = memo.prefix_ids(
            version, "kw", lambda: self.fake_ids(rendered_prefix)
        )
        self.assertEqual(prefix_ids, tuple(self.fake_ids(self.PRELUDE + self.SHARED)))
        content3 = self.SHARED + "TAIL-THREE"
        assembled3 = memo.begin_assembly("sk", content3)
        self.assertIsNotNone(assembled3)
        self.assertEqual(assembled3[3], version)
        prefix_ids3 = memo.prefix_ids(
            version, "kw", lambda: self.fake_ids("WRONG")
        )
        self.assertEqual(prefix_ids3, prefix_ids)

    def test_exact_repeat_shortcuts_everything(self):
        memo = self.make_memo(min_prefix_chars=16)
        content1 = self.SHARED + "TAIL-ONE"
        rendered1, ids1, _ = self.make_family(memo, content1)
        memo.store_full("k1", rendered1, ids1)
        self.assertEqual(memo.get_full("k1"), (rendered1, tuple(ids1)))

    def test_shape_mismatch_never_assembles(self):
        memo = self.make_memo(min_prefix_chars=16)
        self.make_family(memo, self.SHARED + "ONE", shape_key="sk-a")
        self.assertIsNone(memo.begin_assembly("sk-b", self.SHARED + "TWO"))

    def test_short_content_no_family(self):
        memo = self.make_memo(min_prefix_chars=64)
        ids, result = memo.commit_full(
            "sk", "tiny", self.PRELUDE + "tiny" + self.POSTLUDE,
            "kw", lambda t: self.fake_ids(t),
        )
        self.assertEqual(result, "miss")
        self.assertIsNone(memo.begin_assembly("sk", "tiny" + "more"))
        self.assertFalse(memo.stats()["family"])

    def test_diverged_content_falls_back_to_full(self):
        memo = self.make_memo(min_prefix_chars=16)
        self.make_family(memo, self.SHARED + "ONE")
        diverged = "Q" * 100 + "TWO"
        self.assertIsNone(memo.begin_assembly("sk", diverged))
        rendered = self.PRELUDE + diverged + self.POSTLUDE
        ids, result = memo.commit_full(
            "sk", diverged, rendered, "kw", lambda t: self.fake_ids(t)
        )
        self.assertEqual(result, "miss")
        self.assertEqual(ids, self.fake_ids(rendered))
        self.assertEqual(
            memo.stats()["family_prefix_chars"], len(self.SHARED + "ONE")
        )
        assembled = memo.begin_assembly("sk", self.SHARED + "THREE")
        self.assertIsNotNone(assembled)
        self.assertEqual(memo.stats()["family_prefix_chars"], len(self.SHARED))

    def test_non_verbatim_template_never_forms_family(self):
        memo = self.make_memo(min_prefix_chars=16)
        content1 = self.SHARED + "TAIL-ONE   "
        rendered1 = self.PRELUDE + content1.rstrip() + self.POSTLUDE
        ids1, result = memo.commit_full(
            "sk", content1, rendered1, "kw", lambda t: self.fake_ids(t)
        )
        self.assertEqual(result, "miss")
        self.assertEqual(ids1, self.fake_ids(rendered1))
        self.assertFalse(memo.stats()["family"])
        self.assertIsNone(memo.begin_assembly("sk", self.SHARED + "TWO"))

    def test_stale_version_falls_back_to_full_encode(self):
        memo = self.make_memo(min_prefix_chars=16)
        content1 = self.SHARED + "ONE"
        self.make_family(memo, content1)
        assembled = memo.begin_assembly("sk", self.SHARED + "TWO")
        self.assertIsNotNone(assembled)
        stale_version = assembled[3]
        closer = self.SHARED[:50] + "CLOSER"
        assembled2 = memo.begin_assembly("sk", closer)
        self.assertIsNotNone(assembled2)
        self.assertNotEqual(assembled2[3], stale_version)
        self.assertIsNone(
            memo.prefix_ids(stale_version, "kw", lambda: self.fake_ids("STALE")),
        )
        fresh = memo.prefix_ids(
            assembled2[3], "kw", lambda: self.fake_ids(assembled2[0])
        )
        self.assertEqual(fresh, tuple(self.fake_ids(assembled2[0])))

    def test_lru_cap_evicts_oldest(self):
        memo = self.make_memo(min_prefix_chars=16, max_full_entries=2)
        memo.store_full("k1", "A" * 100, [1])
        memo.store_full("k2", "A" * 100 + "B", [2])
        memo.store_full("k3", "A" * 100 + "C", [3])
        self.assertIsNone(memo.get_full("k1"))
        self.assertIsNotNone(memo.get_full("k3"))
        self.assertEqual(memo.stats()["full_entries"], 2)

    def test_claim_serializes_same_shape(self):
        memo = self.make_memo(min_prefix_chars=16)
        self.assertTrue(memo.claim_creation("sk"))
        self.assertFalse(memo.claim_creation("sk"))
        self.assertTrue(memo.claim_creation("other"))
        self.make_family(memo, self.SHARED + "ONE")
        self.assertTrue(memo.claim_creation("sk"))

    def test_wait_family_returns_when_family_committed(self):
        memo = self.make_memo(min_prefix_chars=16)
        memo.claim_creation("sk")
        self.assertFalse(memo.claim_creation("sk"))

        def creator():
            time.sleep(0.3)
            self.make_family(memo, self.SHARED + "ONE")

        t = threading.Thread(target=creator)
        start = time.monotonic()
        t.start()
        self.assertTrue(memo.wait_family("sk", 5.0))
        t.join()
        self.assertLess(time.monotonic() - start, 2.5)

    def test_wait_family_times_out_without_creator(self):
        memo = self.make_memo(min_prefix_chars=16)
        memo.claim_creation("sk")
        start = time.monotonic()
        self.assertFalse(memo.wait_family("sk", 0.2))
        self.assertGreaterEqual(time.monotonic() - start, 0.15)

    def test_wait_family_other_shape_returns_immediately(self):
        memo = self.make_memo(min_prefix_chars=16)
        memo.claim_creation("sk-a")
        start = time.monotonic()
        self.assertFalse(memo.wait_family("sk-b", 1.0))
        self.assertLess(time.monotonic() - start, 0.5)

    def test_timeout_waiter_falls_back_and_creates_family(self):
        memo = self.make_memo(min_prefix_chars=16)
        memo.claim_creation("sk")
        self.assertFalse(memo.wait_family("sk", 0.2))
        content = self.SHARED + "WAITER"
        rendered = self.PRELUDE + content + self.POSTLUDE
        ids, result = memo.commit_full(
            "sk", content, rendered, "kw", lambda t: self.fake_ids(t)
        )
        self.assertEqual(result, "family")
        self.assertTrue(memo.stats()["family"])
        self.assertFalse(memo.stats()["family_claim"])

    def test_concurrent_smoke(self):
        memo = self.make_memo(min_prefix_chars=64, max_full_entries=32)
        fam_a = "A" * 80
        fam_b = "B" * 80
        errors = []

        def worker(wid):
            rnd = random.Random(wid)
            fam = fam_a if wid % 2 == 0 else fam_b
            sk = "sk-a" if wid % 2 == 0 else "sk-b"
            try:
                for i in range(40):
                    content = fam + "T" * (rnd.randint(0, 20) + wid) + str(i)
                    key = (wid, i)
                    if memo.get_full(key) is not None:
                        continue
                    assembled = memo.begin_assembly(sk, content)
                    if assembled is not None:
                        rp, cut, postlude, version = assembled
                        rendered = rp + content[cut:] + postlude
                        prefix_ids = memo.prefix_ids(
                            version, "kw", lambda: self.fake_ids(rp)
                        )
                        if prefix_ids is None:
                            ids = self.fake_ids(rendered)
                        else:
                            ids = list(prefix_ids) + self.fake_ids(
                                rendered[len(rp):]
                            )
                    else:
                        rendered = self.PRELUDE + content + self.POSTLUDE
                        ids, _ = memo.commit_full(
                            sk, content, rendered, "kw",
                            lambda t: self.fake_ids(t),
                        )
                    memo.store_full(key, rendered, ids)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(w,)) for w in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertLessEqual(memo.stats()["full_entries"], 32)
        self.assertTrue(memo.stats()["family"])

    def test_concurrent_bootstrap_serializes_full_renders(self):
        memo = self.make_memo(min_prefix_chars=64)
        fam = "F" * 80
        errors = []
        full_renders = 0
        renders_lock = threading.Lock()
        first_commit = {"done": False}

        def encode(text):
            if not first_commit["done"]:
                time.sleep(0.2)
            return self.fake_ids(text)

        def worker(wid):
            rnd = random.Random(wid)
            try:
                for i in range(10):
                    content = fam + "TAIL-" + str(wid) + "-" + str(i)
                    key = (wid, i)
                    if memo.get_full(key) is not None:
                        continue
                    assembled = memo.begin_assembly("sk", content)
                    if assembled is None:
                        if not memo.claim_creation("sk"):
                            if memo.wait_family("sk", 5.0):
                                assembled = memo.begin_assembly("sk", content)
                    if assembled is not None:
                        rp, cut, postlude, version = assembled
                        rendered = rp + content[cut:] + postlude
                        prefix_ids = memo.prefix_ids(
                            version, "kw", lambda: self.fake_ids(rp)
                        )
                        if prefix_ids is None:
                            ids = self.fake_ids(rendered)
                        else:
                            ids = list(prefix_ids) + self.fake_ids(
                                rendered[len(rp):]
                            )
                    else:
                        with renders_lock:
                            nonlocal full_renders
                            full_renders += 1
                            is_first = not first_commit["done"]
                        rendered = self.PRELUDE + content + self.POSTLUDE
                        ids, result = memo.commit_full(
                            "sk", content, rendered, "kw", encode
                        )
                        if is_first:
                            first_commit["done"] = True
                    memo.store_full(key, rendered, ids)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(w,)) for w in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertLessEqual(full_renders, 3)
        self.assertTrue(memo.stats()["family"])
        self.assertFalse(memo.stats()["family_claim"])

    def test_lcp_len(self):
        lcp = self.memo_module.lcp_len
        self.assertEqual(lcp("abcde", "abcfg"), 3)
        self.assertEqual(lcp("abc", "abc"), 3)
        self.assertEqual(lcp("abc", "ab"), 2)
        self.assertEqual(lcp("", "abc"), 0)
        self.assertEqual(lcp("x" * 100000, "x" * 99999 + "y"), 99999)


class TestChatEncodeFastpath(_BaseCase):
    """E2d: the gated raw-Rust encode. Default OFF; the flag and wiring
    are pinned, and the encode dispatch is proven with fake tokenizers."""

    def _source(self, relpath):
        with open(os.path.join(REPO_ROOT, relpath), encoding="utf-8") as f:
            return f.read()

    def test_e2d_raw_encode_env_defined_default_off(self):
        self.assertRegex(
            self._source(ENVIRON),
            r"SGLANG_CHAT_ENCODE_RAW\s*=\s*EnvBool\(False\)",
        )

    def test_e2d_encode_dispatch(self):
        module = _load_memo_module()

        class _RustEncoding:
            def __init__(self, ids):
                self.ids = ids

        class _RustTokenizer:
            def __init__(self, saw):
                self.saw = saw

            def encode(self, text, add_special_tokens=True):
                self.saw.append(("raw", add_special_tokens, len(text)))
                return _RustEncoding([9000, add_special_tokens * 10, len(text) % 100])

        class _WrapperTokenizer:
            def __init__(self):
                self.saw = []
                self._tokenizer = _RustTokenizer(self.saw)

            def encode(self, text, **kwargs):
                self.saw.append(("wrapper", kwargs))
                return [100, 200, 300]

        tok = _WrapperTokenizer()
        ids = module.encode_chat_text(tok, "abc", {"add_special_tokens": False}, True)
        self.assertEqual(ids, [9000, 0, 3])
        self.assertEqual(tok.saw, [("raw", False, 3)])
        ids = module.encode_chat_text(tok, "xy", {}, True)
        self.assertEqual(ids, [9000, 10, 2])
        self.assertEqual(tok.saw, [("raw", False, 3), ("raw", True, 2)])
        tok.saw.clear()
        ids = module.encode_chat_text(tok, "abc", {"add_special_tokens": False}, False)
        self.assertEqual(ids, [100, 200, 300])
        self.assertEqual(tok.saw, [("wrapper", {"add_special_tokens": False})])
        class _NoRust:
            def encode(self, text, **kwargs):
                return [7]
        self.assertEqual(module.encode_chat_text(_NoRust(), "x", {}, True), [7])

    def test_e2d_wired_in_serving_chat(self):
        src = self._source(SERVING_CHAT)
        self.assertIn("encode_chat_text", src)
        self.assertRegex(src, r"envs\.SGLANG_CHAT_ENCODE_RAW\.get\(\)")
        self.assertGreaterEqual(src.count("encode_chat_text("), 2)


if __name__ == "__main__":
    unittest.main()
