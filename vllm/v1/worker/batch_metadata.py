# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Owned CPU metadata for one live target forward, without tokens or tensors."""

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu_input_batch import InputBatch


@dataclass(frozen=True)
class RequestMetadata:
    """Scalar lengths and allocated KV block counts for one real request."""

    request_id: str
    prompt_tokens: int
    computed_tokens_cpu: int
    context_tokens_cpu: int
    query_tokens: int
    draft_tokens: int
    kv_blocks: tuple[int, ...]  # One allocated-block count per KV cache group.


@dataclass(frozen=True)
class BatchMetadata:
    """Real requests in runner order, with graph padding recorded separately."""

    step: int
    num_requests: int
    num_tokens: int
    num_requests_padded: int
    num_tokens_padded: int
    context_lengths_are_upper_bounds: bool  # True for async speculative decoding.
    microbatched: bool  # Internal forward slices are not recorded separately.
    requests: tuple[RequestMetadata, ...]

    @classmethod
    def from_batch(
        cls,
        batch: "InputBatch",
        scheduler: "SchedulerOutput",
        *,
        step: int,
        num_tokens_padded: int,
        num_reqs_padded: int,
        context_lengths_are_upper_bounds: bool,
        microbatched: bool,
    ) -> "BatchMetadata":
        """Read InputBatch CPU rows and SchedulerOutput counts into owned values."""
        requests = []
        for i, req_id in enumerate(batch.req_ids):
            computed = int(batch.num_computed_tokens_cpu[i])
            query = int(scheduler.num_scheduled_tokens[req_id])
            requests.append(
                RequestMetadata(
                    request_id=req_id,
                    prompt_tokens=int(batch.num_prompt_tokens[i]),
                    computed_tokens_cpu=computed,
                    context_tokens_cpu=computed + query,
                    query_tokens=query,
                    draft_tokens=len(
                        scheduler.scheduled_spec_decode_tokens.get(req_id, ())
                    ),
                    kv_blocks=tuple(
                        # Convert subdivided kernel blocks back to allocated KV blocks.
                        int(table.num_blocks_per_row[i]) // table.blocks_per_kv_block
                        for table in batch.block_table.block_tables
                    ),
                )
            )
        return BatchMetadata(
            step=step,
            num_requests=len(requests),
            num_tokens=sum(request.query_tokens for request in requests),
            num_requests_padded=int(num_reqs_padded),
            num_tokens_padded=int(num_tokens_padded),
            context_lengths_are_upper_bounds=context_lengths_are_upper_bounds,
            microbatched=microbatched,
            requests=tuple(requests),
        )

    def to_dict(self) -> dict[str, Any]:
        """Build the JSON record on the writer thread, preserving the log schema."""
        return {"type": "batch", **asdict(self)}
