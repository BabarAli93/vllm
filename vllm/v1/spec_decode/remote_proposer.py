"""Remote proposer.


There is no local drafting. The draft model is a real transformer on the
Jetson; drafts arrive over the gRPC stream and are attached to the request by
DSDGate. This class exists only to satisfy vLLM's proposer contract so that
speculative decoding is enabled on the engine.

The contract, from gpu_model_runner.py:5148, is positional — no req_ids are
passed, and _get_draft_token_ids_cpu() zips the return against
input_batch.req_ids by index:

    if isinstance(self._draft_token_ids, list):
        return self._draft_token_ids, self.input_batch.req_ids

A wrong-length return misattributes drafts to the wrong requests without
raising, so the length must equal len(sampled_token_ids).

"""

from typing import Any


class RemoteProposer:
    def __init__(self, vllm_config: Any) -> None:
        self.vllm_config = vllm_config

    def propose(
        self,
        sampled_token_ids: list[list[int]],
        num_tokens_no_spec: Any,
        token_ids_cpu: Any,
        slot_mappings: Any = None,
        **kwargs: Any,
    ) -> list[list[int]]:
        return [[] for _ in range(len(sampled_token_ids))]

    def load_model(self, *a: Any, **k: Any) -> None:
        return None

    def dummy_run(self, *a: Any, **k: Any) -> None:
        return None
