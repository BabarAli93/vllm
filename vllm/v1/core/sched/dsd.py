"""Distributed speculative decoding: buffer and scheduling gate.

Install at:  vllm/v1/core/sched/dsd.py   (replaces the previous version)

CHANGES FROM THE PREVIOUS VERSION
---------------------------------
1. Request-id resolution. The engine suffixes the client's request_id
   (client "0386a834-...-751ca1a32a81" becomes "...-bff1a8b8" by the time it
   reaches Scheduler.add_request). enroll() saw the suffixed id and submit()
   the unsuffixed one, so buf was keyed under an id prepare() never looked up:
   drafts arrived, nothing was ever speculated, and every round fell through
   to the timeout.

2. Implicit enrolment removed. enroll() works, and enrolling from the wire id
   made the mismatch worse — it added a second, unusable entry to self.reqs
   that finish() never cleared, which is why `enrolled` climbed 2, 4, 6, 8
   across requests instead of resetting.

3. Per-request stats. The Counter lives on the scheduler and accumulated over
   the engine's whole lifetime, so every log line was a running total. Each
   request now reports its own.

WHY A GATE IS NEEDED AT ALL
---------------------------
propose() runs at the END of execute_model for step N: the token for step N
exists but has not been emitted yet — output leaves after execute_model
returns, via update_from_output -> ZMQ -> AsyncLLM -> gRPC. So the client is
still one token behind, and a proposer that blocked waiting for a draft
conditioned on the current length would deadlock, because the token the client
needs is stuck behind the block.

What is required instead is that step N+1 is not scheduled until the client has
seen step N's output. That is a scheduling decision, so it lives in schedule().
While a request waits it stays in `running`, keeps its KV blocks, and
contributes zero scheduled tokens — so the GPU serves other requests rather
than idling.

THREADING
---------
submit() is called from EngineCore's input-queue drain and prepare() from
schedule(); both run on the EngineCore busy loop, same thread. No locking
needed.
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
        self.timeout = float(os.environ.get("DSD_TIMEOUT_MS", "3000")) / 1e3

        self.reqs: set[str] = set()                      # engine-side request ids
        self.buf: dict[str, tuple[int, list[int]]] = {}  # one pending per request
        self.deadline: dict[str, float] = {}
        self.alias: dict[str, str] = {}                  # wire id -> engine id

        self.stats = Counter()                           # lifetime
        self.req_stats: dict[str, Counter] = {}          # per request
        self._warned_async = False
        self._warned_suffix = False

    # -- called from Scheduler.add_request -------------------------------- #

    def enroll(self, request) -> None:
        params = getattr(request, "sampling_params", None)
        extra = getattr(params, "extra_args", None) or {}
        if not extra.get("remote_spec"):
            return
        rid = request.request_id
        self.reqs.add(rid)
        self.req_stats[rid] = Counter()
        self._bump(rid, "enrolled")

    # -- called from EngineCoreProc._handle_client_request ---------------- #

    def submit(self, msg) -> None:
        """Buffer a draft. Must not block; runs on the engine's busy loop.

        `msg` is whatever generic_decoder produced — a plain dict, since
        process_input_sockets decodes every non-ADD type with an untyped
        MsgpackDecoder. Validate rather than assume: a malformed message
        arrives as an arbitrary Python object, and an unguarded index here
        would take down the busy loop.
        """
        try:
            wire_id = msg["request_id"]
            ctx_len = int(msg["ctx_len"])
            token_ids = list(msg["token_ids"])
        except (TypeError, KeyError, ValueError):
            logger.warning("DSD: malformed draft message: %r", msg)
            self.stats["malformed"] += 1
            return

        rid = self._resolve(wire_id)
        if rid is None:
            # Finished, aborted, or never a DSD request. NOT an invitation to
            # enrol: enrolment must key off the engine's request id, which only
            # add_request knows.
            self.stats["orphaned"] += 1
            return

        # A newer draft always supersedes an older one: the client sends draft
        # N+1 only after receiving verification N, so at most one is in flight.
        self.buf[rid] = (ctx_len, token_ids)
        self._bump(rid, "received")

    def _resolve(self, wire_id: str) -> str | None:
        """Map the client's request id to the engine's.

        The engine appends a suffix somewhere between AsyncLLM.generate() and
        Scheduler.add_request, so the id on the wire is a prefix of the id the
        scheduler holds. Resolution is cached, so the scan runs once per
        request.
        """
        rid = self.alias.get(wire_id)
        if rid is not None and rid in self.reqs:
            return rid
        if wire_id in self.reqs:
            self.alias[wire_id] = wire_id
            return wire_id
        for r in self.reqs:
            if r.startswith(wire_id):
                if not self._warned_suffix:
                    self._warned_suffix = True
                    logger.warning(
                        "DSD: engine suffixes request ids (wire=%s engine=%s); "
                        "resolving by prefix.", wire_id, r)
                self.alias[wire_id] = r
                return r
        return None

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
        # in-flight placeholder tokens. Async scheduling breaks that, and also
        # routes drafts down a GPU-tensor path built for on-device drafts
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
                self._bump(rid, "speculated")
                return True
            if ctx_len < request.num_tokens:
                # Conditioned on a prefix we have already moved past. Drop it:
                # the target would reject it anyway, costing a round trip.
                del self.buf[rid]
                self._bump(rid, "stale")
                if self.req_stats.get(rid, {}).get("stale", 0) in (1, 10, 100):
                    logger.warning(
                        "DSD stale draft rid=%s client_ctx=%d engine_ctx=%d "
                        "(prompt=%d computed=%d)",
                        rid[:8], ctx_len, request.num_tokens,
                        request.num_prompt_tokens, request.num_computed_tokens)
            # ctx_len > num_tokens: client ran ahead. Keep it and wait.

        now = time.monotonic()
        end = self.deadline.get(rid)
        if end is None:
            self.deadline[rid] = now + self.timeout
            self._bump(rid, "gated")
            return False
        if now < end:
            return False

        # Client is slow or gone. Decode normally rather than pinning KV blocks
        # indefinitely. The request stays enrolled, so speculation resumes if
        # drafts come back.
        del self.deadline[rid]
        self._bump(rid, "timeout")
        request.spec_token_ids = []
        return True

    # -- called from Scheduler._free_request ------------------------------ #

    def finish(self, request_id: str) -> None:
        # Terminal paths only. Preempted requests are still live and must keep
        # their DSD state.
        s = self.req_stats.pop(request_id, None)
        if s:
            logger.info(
                "DSD [%s] speculated=%d stale=%d timeout=%d gated=%d received=%d",
                request_id[:8], s["speculated"], s["stale"], s["timeout"],
                s["gated"], s["received"])
        self.reqs.discard(request_id)
        self.buf.pop(request_id, None)
        self.deadline.pop(request_id, None)
        for wire, engine in list(self.alias.items()):
            if engine == request_id:
                del self.alias[wire]

    def _bump(self, rid: str, key: str) -> None:
        self.stats[key] += 1
        s = self.req_stats.get(rid)
        if s is not None:
            s[key] += 1