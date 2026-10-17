# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Set VLLM_BATCH_INSPECT_DIR and VLLM_USE_V2_MODEL_RUNNER=0 to log live batches.

The runner supplies CPU metadata; a daemon thread serializes and writes JSONL.
Full queues drop records. CPU collection and the writer still add GIL overhead.
Abrupt exits can lose buffered records; a clean close writes a loss summary.
"""

import json
import logging
import os
import socket
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Thread
from typing import TYPE_CHECKING, Any, TextIO
from uuid import uuid4

from vllm.v1.worker.batch_metadata import BatchMetadata

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu_input_batch import InputBatch

logger = logging.getLogger(__name__)


class BatchInspector:
    """Shared collector: one runner producer, a bounded queue, and one writer."""

    def __init__(self, directory: str, queue_capacity: int = 1024) -> None:
        """Start a writer with a unique output file for this GPU worker process."""
        if queue_capacity < 1:
            raise ValueError("queue_capacity must be positive")
        self.path = Path(directory) / (
            f"batch-{socket.gethostname()}-{os.getpid()}-{uuid4().hex[:8]}.jsonl"
        )
        self._queue: Queue[BatchMetadata] = Queue(maxsize=queue_capacity)
        self._closing = Event()
        self._failed = Event()
        self.steps = 0
        self.dropped = 0
        self._thread = Thread(target=self._write, name="batch-inspector", daemon=True)
        self._thread.start()

    def enqueue_or_drop(
        self,
        batch: "InputBatch",
        scheduler: "SchedulerOutput",
        *,
        num_tokens_padded: int,
        num_reqs_padded: int,
        context_lengths_are_upper_bounds: bool,
        microbatched: bool = False,
    ) -> bool:
        """Capture CPU batch metadata and enqueue it without waiting for space."""
        self.steps += 1
        # Skip metadata capture too when the writer is full, closed, or failed.
        if self._closing.is_set() or self._failed.is_set() or self._queue.full():
            self.dropped += 1
            return False
        metadata = BatchMetadata.from_batch(
            batch,
            scheduler,
            step=self.steps,
            num_tokens_padded=num_tokens_padded,
            num_reqs_padded=num_reqs_padded,
            context_lengths_are_upper_bounds=context_lengths_are_upper_bounds,
            microbatched=microbatched,
        )
        try:
            self._queue.put_nowait(metadata)
            return True
        except Full:
            self.dropped += 1
            return False

    def close(self, timeout: float = 2.0) -> bool:
        """Drain queued metadata at shutdown; False means incomplete output."""
        self._closing.set()
        self._thread.join(timeout=timeout)
        return not self._thread.is_alive() and not self._failed.is_set()

    @staticmethod
    def _write_record(stream: TextIO, record: dict[str, Any]) -> None:
        """Serialize one plain record and flush it using the writer's file handle."""
        stream.write(json.dumps(record) + "\n")
        stream.flush()

    def _write(self) -> None:
        """Consume BatchMetadata objects until close, then write a loss summary."""
        written = 0
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("x", encoding="utf-8") as stream:
                while not self._closing.is_set() or not self._queue.empty():
                    try:
                        metadata = self._queue.get(timeout=0.1)
                    except Empty:
                        continue
                    self._write_record(stream, metadata.to_dict())
                    written += 1
                self._write_record(
                    stream,
                    {
                        "type": "summary",
                        "steps": self.steps,
                        "written": written,
                        "dropped": self.dropped,
                    },
                )
        except OSError:
            # Disk failures disable collection, without stopping model inference.
            self._failed.set()
            logger.warning(
                "Batch inspection disabled: cannot write %s", self.path, exc_info=True
            )
