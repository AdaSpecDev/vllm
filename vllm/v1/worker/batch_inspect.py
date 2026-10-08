# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU metadata for live target forwards; no tokens, tensors, or GPU timing.

Set VLLM_BATCH_INSPECT_DIR and VLLM_USE_V2_MODEL_RUNNER=0 to enable logging.
Each worker writes its own JSONL file. Each batch
contains real requests in runner order, excluding CUDA-graph padding. CPU context
lengths are upper bounds under async speculative decoding, as marked in records.
Microbatched forwards are marked; their internal slices are not logged separately.

Only small Python values are copied on the forward path. Serialization and disk
writes run in a daemon thread. A full queue drops the current record; step gaps
and the final summary expose losses. This still adds CPU work and GIL contention;
it is not a zero-overhead profiler. Abrupt exits can lose buffered records.
"""

import json
import logging
import os
import socket
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Thread
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu_input_batch import InputBatch

logger = logging.getLogger(__name__)


class BatchInspector:
    """One producer (the runner), one writer, and a bounded metadata queue."""

    def __init__(self, directory: str, queue_capacity: int = 1024) -> None:
        if queue_capacity < 1:
            raise ValueError("queue_capacity must be positive")
        self.path = Path(directory) / (
            f"batch-{socket.gethostname()}-{os.getpid()}-{uuid4().hex[:8]}.jsonl"
        )
        self._queue: Queue[dict[str, Any]] = Queue(maxsize=queue_capacity)
        self._closing = Event()
        self._failed = Event()
        self.steps = 0
        self.dropped = 0
        self._thread = Thread(target=self._write, name="batch-inspector", daemon=True)
        self._thread.start()

    def record(
        self,
        batch: "InputBatch",
        scheduler: "SchedulerOutput",
        *,
        num_tokens_padded: int,
        num_reqs_padded: int,
        context_lengths_are_upper_bounds: bool,
        microbatched: bool = False,
    ) -> None:
        self.steps += 1
        # Skip even the metadata construction while the writer is behind.
        if self._closing.is_set() or self._failed.is_set() or self._queue.full():
            self.dropped += 1
            return

        requests: list[dict[str, Any]] = []
        for i, req_id in enumerate(batch.req_ids):
            computed = int(batch.num_computed_tokens_cpu[i])
            query = int(scheduler.num_scheduled_tokens[req_id])
            requests.append(
                {
                    "request_id": req_id,
                    "prompt_tokens": int(batch.num_prompt_tokens[i]),
                    "computed_tokens_cpu": computed,
                    "context_tokens_cpu": computed + query,
                    "query_tokens": query,
                    "draft_tokens": len(
                        scheduler.scheduled_spec_decode_tokens.get(req_id, ())
                    ),
                    # Count allocated KV blocks, not subdivided kernel blocks.
                    "kv_blocks": [
                        int(table.num_blocks_per_row[i]) // table.blocks_per_kv_block
                        for table in batch.block_table.block_tables
                    ],
                }
            )

        record = {
            "type": "batch",
            "step": self.steps,
            "num_requests": len(requests),
            "num_tokens": sum(r["query_tokens"] for r in requests),
            "num_requests_padded": int(num_reqs_padded),
            "num_tokens_padded": int(num_tokens_padded),
            "context_lengths_are_upper_bounds": context_lengths_are_upper_bounds,
            "microbatched": microbatched,
            "requests": requests,
        }
        try:
            self._queue.put_nowait(record)
        except Full:
            self.dropped += 1

    def close(self, timeout: float = 2.0) -> bool:
        """Drain at shutdown, with a bounded wait. False means incomplete output."""
        self._closing.set()
        self._thread.join(timeout=timeout)
        return not self._thread.is_alive() and not self._failed.is_set()

    def _write(self) -> None:
        written = 0
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("x", encoding="utf-8") as stream:
                while not self._closing.is_set() or not self._queue.empty():
                    try:
                        record = self._queue.get(timeout=0.1)
                    except Empty:
                        continue
                    stream.write(json.dumps(record) + "\n")
                    stream.flush()
                    written += 1
                stream.write(
                    json.dumps(
                        {
                            "type": "summary",
                            "steps": self.steps,
                            "written": written,
                            "dropped": self.dropped,
                        }
                    )
                    + "\n"
                )
        except OSError:
            # Logging failure must not stop inference. Report once, off-path.
            self._failed.set()
            logger.warning(
                "Batch inspection disabled: cannot write %s", self.path, exc_info=True
            )
