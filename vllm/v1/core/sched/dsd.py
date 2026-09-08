"""Distributed speculative decoding: buffer and scheduling gate.

All DSD logic lives here so that scheduler.py takes a 6-line diff. schedule()
is churned constantly upstream, and a small insertion is a re-placement on
rebase rather than an archaeology session.

WHY A GATE IS NEEDED AT ALL
---------------------------
The obvious design is to have the proposer wait for the client's draft. It
deadlocks. propose() runs at the END of execute_model for step N: the token for
step N exists but has not been emitted yet — output leaves after execute_model
returns, via update_from_output -> ZMQ -> AsyncLLM -> gRPC. So the client is
still one token behind. If propose() blocked waiting for a draft conditioned on
the current length, the client could not produce one, because the token it
needs is stuck behind the block.

What is actually required is that step N+1 is not scheduled until the client
has seen step N's output. That is a scheduling decision, so it lives in
schedule(). While a request waits it stays in `running`, keeps its KV blocks,
and contributes zero scheduled tokens — so the GPU serves other requests
instead of idling.

THREADING
---------
submit() is called from EngineCore's input-queue drain and prepare() from
schedule(); both run on the EngineCore busy loop, same thread. No locking is
needed, and adding any would be misleading about where the concurrency is.
"""

from __future__ import annotations

import logging
import os
import time
from collections import Counter

logger = logging.getLogger(__name__)


class DSDGate:
    def __init__(self, vllm_config) -> None:
        spec = getattr(vllm_config, "speculative_config", None)
        self.max_spec = int(getattr(spec, "num_speculative_tokens", 0) or 0)
        self.timeout = float(os.environ.get("DSD_TIMEOUT_MS", "600")) / 1e3

        self.reqs: set[str] = set()             # enrolled request ids
        self.buf: dict[str, tuple[int, list[int]]] = {}   # one pending per request
        self.deadline: dict[str, float] = {}
        self.stats = Counter()
        self._warned_async = False

    # -- called from Scheduler.add_request -------------------------------- #

    def enroll(self, request) -> None:
        params = getattr(request, "sampling_params", None)
        extra = getattr(params, "extra_args", None) or {}
        logger.warning("DSD enroll: request_id=%s, extra=%s", request.request_id, extra)
        if extra.get("remote_spec"):
            self.reqs.add(request.request_id)
            self.stats["enrolled"] += 1

    # -- called from EngineCoreProc._handle_client_request ---------------- #

    def submit(self, msg) -> None:
        """Buffer a draft. Must not block; runs on the engine's busy loop.

        `msg` is whatever generic_decoder produced — a plain dict, since
        process_input_sockets decodes every non-ADD type with an untyped
        MsgpackDecoder. Validate rather than assume: a malformed message
        arrives as an arbitrary Python object, and an unguarded index here
        would take down the engine's busy loop.
        """
        try:
            rid = msg["request_id"]
            ctx_len = int(msg["ctx_len"])
            token_ids = list(msg["token_ids"])
        except (TypeError, KeyError, ValueError):
            logger.warning("DSD: malformed draft message: %r", msg)
            self.stats["malformed"] += 1
            return

        if rid not in self.reqs:
            # Finished, aborted, or never a DSD request.
            self.reqs.add(rid)  # enroll it so we don't log again
            self.stats["enrolled"] += 1

        # A newer draft always supersedes an older one: the client sends draft
        # N+1 only after receiving verification N, so at most one is in flight.
        self.buf[rid] = (ctx_len, token_ids)
        self.stats["received"] += 1

    # -- called from Scheduler.schedule ----------------------------------- #

    def prepare(self, request) -> bool:
        """True: schedule now. False: waiting on the edge drafter.

        On True with a draft available, request.spec_token_ids is set and the
        scheduler's existing speculative machinery does the rest.
        """
        rid = request.request_id
        if rid not in self.reqs:
            return True

        # Prefill is never gated: the client cannot draft until it has the
        # first sampled token, and prefill is what produces it.
        if request.num_computed_tokens < request.num_prompt_tokens:
            return True

        # ctx_len is compared against request.num_tokens, which assumes no
        # in-flight placeholder tokens. Async scheduling breaks that, and it
        # also routes drafts down a GPU-tensor path built for on-device drafts
        # (see _copy_draft_token_ids_to_cpu). Warn once rather than silently
        # producing a zero acceptance rate.
        if getattr(request, "num_output_placeholders", 0) and not self._warned_async:
            self._warned_async = True
            logger.error(
                "DSD: async scheduling appears to be enabled "
                "(num_output_placeholders=%d). Disable it in AsyncEngineArgs; "
                "drafts will otherwise never match.",
                request.num_output_placeholders,
            )

        entry = self.buf.get(rid)
        if entry is not None:
            ctx_len, token_ids = entry
            if ctx_len == request.num_tokens:
                del self.buf[rid]
                self.deadline.pop(rid, None)
                request.spec_token_ids = token_ids[: self.max_spec]
                self.stats["speculated"] += 1
                return True
            if ctx_len < request.num_tokens:
                # Conditioned on a prefix we have already moved past. Drop it:
                # the target would reject it anyway, costing a round trip.
                del self.buf[rid]
                self.stats["stale"] += 1
            # ctx_len > num_tokens: client ran ahead of us. Keep it and wait.

        now = time.monotonic()
        end = self.deadline.get(rid)
        if end is None:
            self.deadline[rid] = now + self.timeout
            self.stats["gated"] += 1
            return False
        if now < end:
            return False

        # Client is slow or gone. Decode normally rather than pinning KV blocks
        # indefinitely. The request stays enrolled, so speculation resumes if
        # drafts come back.
        del self.deadline[rid]
        self.stats["timeout"] += 1
        request.spec_token_ids = []
        return True

    # -- called from Scheduler._free_request ------------------------------ #

    def finish(self, request_id: str) -> None:
        # Terminal paths only. Preempted requests are still live and must keep
        # their DSD state.
        self.reqs.discard(request_id)
        self.buf.pop(request_id, None)
        self.deadline.pop(request_id, None)
        logger.warning(f"DSD stats: {self.stats}")

    def log_stats(self) -> None:
        logger.info("DSD %s inflight=%d enrolled=%d",
                    dict(self.stats), len(self.buf), len(self.reqs))
