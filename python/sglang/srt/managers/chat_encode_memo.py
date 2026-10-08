"""E2a (branch E): render+encode memo for the chat ingest path.

The chat ingest path renders the full prompt with the Jinja template and
encodes it with the HF fast tokenizer (serving_chat.
_render_and_encode_chat_template). The after-E1 tokenizertime2 run
(16 concurrent 1M-token prompts sharing a 99% prefix, 8 pool workers)
measured this phase: 107.6 s of conversion CPU (5.5-11.4 s per request), a
17.1 s wall in 2 waves of 8 (wave 1: 8 concurrent conversions in 11.6 s),
against 3.88 s for the same conversion running solo. Two costs dominate:

- tokenizer.encode of the ~1M-token prompt, ~3.5 s per request, which the
  E1 pool parallelizes but re-runs 16 times on 99% identical input;
- the GIL-held render+message-processing part, ~1 s per request, which
  serializes across the pool and inflates the 8-wide first wave to
  5.5-11.4 s. The same GIL saturation starves the event loop thread: the
  first tokenized request was only handed to the scheduler ~12 s after its
  conversion finished (first Prefill batch with #queue-req: 0), idling the
  NPU while 8 workers held the GIL.

This memo attacks all three:

1. Full-prompt cache: an exact (template, messages, tools, kwargs) repeat
   returns the stored (rendered, ids) pair without rendering or encoding.
2. Family rendered-prefix assembly: the first long prompt defines a family
   by splitting its render around the anchored (long) message content into
   prelude + content + postlude (a unique substring find; a template that
   transforms the content fails the find and never forms a family). A
   sibling whose anchored content starts with the family's content prefix
   is assembled as rendered_prefix + tail + postlude -- no Jinja render at
   all (which is what removes the ~1 s GIL serialization) -- and encoded as
   prefix_ids + encode(tail + postlude), re-encoding only the ~10k-token
   tail instead of the ~1M-token prompt.
3. Serialized family bootstrap (claim_creation / wait_family): the family
   only exists once the first request's full render+encode has committed,
   so a burst of N concurrent first-family requests must not all start
   their own GIL-bound render (that is exactly the 8-wide wave above). The
   first request for a shape claims the creation and runs solo; the
   siblings block on a Condition (releasing the GIL) until the family
   exists, then assemble. One full render per burst, not N. The claim is a
   best-effort hint: a stalled creator simply times out the waiters, which
   fall back to the exact full render+encode (correctness never depends on
   the claim).

Correctness: prefix ids are a standalone encode of the prefix text, and the
returned ids are their concatenation with the tail encode. That is a valid
tokenization of the prompt text (the tokenizer is deterministic), but the
tokens around the prefix|tail cut can differ from a single full encode of
the same text by a few subword merges. Downstream only ever sees a
self-consistent token sequence, and the scheduler's radix cache shares the
true common token prefix, so generation semantics are unchanged; the shared
KV match with sibling prompts is at most a few boundary tokens shorter.
Assembly only engages at or above MIN_PREFIX_CHARS, where that skew is
negligible, and is gated by SGLANG_CHAT_ENCODE_MEMO. The family also assumes
the template passes the anchored content through verbatim (all standard
chat templates do); the unique-find at creation is the check, and any
mismatch simply disables the family for that template+shape.

The class is stdlib-only and thread-safe: the E1 chat-encode pool runs
concurrent requests through it. Encode work happens outside the lock; two
workers racing on the same shrink produce the same state (idempotent), and
a stale assembly (the family prefix shrank while a worker was between
begin_assembly and prefix_ids) is detected by the version counter and falls
back to a full encode of the assembled string. wait_family blocks on the
internal condition without holding the GIL in Python (Condition.wait
 releases it), so blocked siblings do not starve the event loop.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, Optional, Tuple

# Minimum anchored-content length (chars) for family formation. Below it,
# the re-encode savings are negligible and the BPE-boundary skew is not
# worth paying for.
MIN_PREFIX_CHARS = 4096

# LRU cap on distinct full (rendered, ids) entries. One 1M-token entry is
# ~4 MB of rendered text plus ~36 MB of int ids, so 16 entries bound the
# memo at ~640 MB worst case (plus one family).
MAX_FULL_ENTRIES = 16


def lcp_len(a: str, b: str) -> int:
    """Length of the longest common prefix, via binary-searched slice
    comparison (O(log n) C-level memcmps instead of O(n) char steps)."""
    n = min(len(a), len(b))
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if a[:mid] == b[:mid]:
            lo = mid
        else:
            hi = mid - 1
    return lo


class ChatEncodeMemo:
    """Render+encode memo: full-prompt dedup + family rendered-prefix
    assembly. See the module docstring for the cost model and the
    correctness argument."""

    def __init__(
        self,
        min_prefix_chars: int = MIN_PREFIX_CHARS,
        max_full_entries: int = MAX_FULL_ENTRIES,
    ):
        # A Condition (not a bare Lock): wait_family blocks on it while the
        # family creator runs.
        self._cond = threading.Condition()
        self._full: "OrderedDict[Any, Tuple[str, Tuple[int, ...]]]" = OrderedDict()
        self._min_prefix_chars = min_prefix_chars
        self._max_full_entries = max_full_entries
        # One family at a time: the first long prompt wins; requests with a
        # different shape key (or a diverged long content) simply do not
        # assemble and fall back to the exact render+encode.
        self._fam_shape_key: Optional[Any] = None
        self._fam_prelude: Optional[str] = None
        self._fam_postlude: Optional[str] = None
        self._fam_content_prefix: Optional[str] = None
        self._fam_rendered_prefix: Optional[str] = None
        self._fam_prefix_ids: Dict[str, Tuple[int, ...]] = {}
        self._fam_version = 0
        # E2a: shape_key whose full render+encode is in flight (the creator
        # claimed it via claim_creation). None once the family is committed.
        self._fam_claim: Optional[Any] = None

    # -- full-prompt cache -------------------------------------------------

    def get_full(self, key: Any) -> Optional[Tuple[str, Tuple[int, ...]]]:
        """LRU lookup of a full (rendered, ids) entry by cache key."""
        with self._cond:
            entry = self._full.get(key)
            if entry is None:
                return None
            self._full.move_to_end(key)
            return entry

    def store_full(self, key: Any, rendered: str, ids: Any) -> None:
        """Record a newly produced full (rendered, ids) pair (LRU)."""
        with self._cond:
            self._full[key] = (rendered, tuple(ids))
            self._full.move_to_end(key)
            while len(self._full) > self._max_full_entries:
                self._full.popitem(last=False)

    # -- family rendered-prefix assembly -----------------------------------

    def begin_assembly(
        self, shape_key: Any, content: Optional[str]
    ) -> Optional[Tuple[str, int, str, int]]:
        """Return (rendered_prefix, content_cut, postlude, version) when the
        request can be assembled from the family -- rendered =
        rendered_prefix + content[content_cut:] + postlude, with no Jinja
        render -- else None. When the content does not start with the
        family's content prefix but the LCP is long enough, the family
        prefix is first shrunk to that LCP (bumping the version, which
        invalidates any encoded prefix ids)."""
        with self._cond:
            if (
                self._fam_shape_key is None
                or self._fam_shape_key != shape_key
                or content is None
            ):
                return None
            prefix = self._fam_content_prefix
            if content.startswith(prefix):
                return (
                    self._fam_rendered_prefix,
                    len(prefix),
                    self._fam_postlude,
                    self._fam_version,
                )
            l = lcp_len(prefix, content)
            if l < self._min_prefix_chars:
                return None
            self._fam_content_prefix = prefix[:l]
            self._fam_rendered_prefix = self._fam_prelude + prefix[:l]
            self._fam_prefix_ids.clear()
            self._fam_version += 1
            return (
                self._fam_rendered_prefix,
                l,
                self._fam_postlude,
                self._fam_version,
            )

    def prefix_ids(
        self,
        version: int,
        encode_kwargs_key: str,
        make: Callable[[], Any],
    ) -> Optional[Tuple[int, ...]]:
        """The ids of the family prefix of `version`, encoding it via `make`
        on first use for `encode_kwargs_key`. None when the family changed
        (version bumped) after the caller matched, so the caller falls back
        to a full encode."""
        with self._cond:
            if version != self._fam_version:
                return None
            cached = self._fam_prefix_ids.get(encode_kwargs_key)
        if cached is not None:
            return cached
        ids = tuple(make())
        with self._cond:
            if version != self._fam_version:
                return None
            self._fam_prefix_ids.setdefault(encode_kwargs_key, ids)
            return self._fam_prefix_ids[encode_kwargs_key]

    def commit_full(
        self,
        shape_key: Any,
        content: Optional[str],
        rendered: str,
        encode_kwargs_key: str,
        encode: Callable[[str], Any],
    ) -> Tuple[List[int], str]:
        """After a full Jinja render: returns (prompt_ids, memo_result).
        Creates the family when the anchored content passes verbatim
        (unique find) and is long enough -- the creating request's ids are
        then prefix_ids + encode(postlude), a few-token BPE-boundary skew at
        the content|postlude cut (documented) that saves the second full
        encode. Without a family (or when another worker won the race with a
        different content): the exact full encode."""
        with self._cond:
            have_family = self._fam_shape_key is not None
        if have_family:
            return list(encode(rendered)), "miss"
        if content is None or len(content) < self._min_prefix_chars:
            return list(encode(rendered)), "miss"
        pos = rendered.find(content)
        if pos < 0 or rendered.find(content, pos + 1) >= 0:
            return list(encode(rendered)), "miss"
        prelude = rendered[:pos]
        postlude = rendered[pos + len(content):]
        prefix_ids = tuple(encode(prelude + content))
        with self._cond:
            if self._fam_shape_key is not None:
                # Another worker created the family (from a different
                # content) while we were encoding: exact encode for us.
                return list(encode(rendered)), "miss"
            self._fam_shape_key = shape_key
            self._fam_prelude = prelude
            self._fam_postlude = postlude
            self._fam_content_prefix = content
            self._fam_rendered_prefix = prelude + content
            self._fam_prefix_ids[encode_kwargs_key] = prefix_ids
            self._fam_version = 1
            self._fam_claim = None
            self._cond.notify_all()
        return list(prefix_ids) + list(encode(postlude)), "family"

    # -- serialized family bootstrap ---------------------------------------

    def claim_creation(self, shape_key: Any) -> bool:
        """E2a: True when this caller should run the full render+encode now
        (it becomes the family creator, or there is nothing to wait for --
        the family already exists, or another shape owns the slot). False
        when a same-shape creation is in flight: the caller should
        wait_family instead of starting its own GIL-bound render."""
        with self._cond:
            if self._fam_shape_key is not None:
                return True
            if self._fam_claim is None:
                self._fam_claim = shape_key
                return True
            return self._fam_claim != shape_key

    def wait_family(self, shape_key: Any, timeout_s: float) -> bool:
        """E2a: block (releasing the GIL) until the family for shape_key
        exists. True on success; False when the family went to another
        shape, there is no creation in flight for this shape, or the
        timeout expired (the caller then falls back to the exact full
        render+encode, which commit_full handles)."""
        deadline = time.monotonic() + timeout_s
        with self._cond:
            while True:
                if self._fam_shape_key is not None:
                    return self._fam_shape_key == shape_key
                if self._fam_claim != shape_key:
                    return False
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cond.wait(remaining)

    def stats(self) -> Dict[str, int]:
        with self._cond:
            return {
                "full_entries": len(self._full),
                "family": self._fam_shape_key is not None,
                "family_claim": self._fam_claim is not None,
                "family_prefix_chars": len(self._fam_content_prefix or ""),
            }

