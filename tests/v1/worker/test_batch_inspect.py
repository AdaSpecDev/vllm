# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only: run with python tests/v1/worker/test_batch_inspect.py -v."""

import json
import tempfile
import unittest
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace
from unittest.mock import patch

from vllm.v1.worker.batch_inspect import BatchInspector


def inputs():
    # Runner order differs from scheduler order; queued work isn't in this batch.
    batch = SimpleNamespace(
        req_ids=["long", "short"],
        num_prompt_tokens=[100, 10, 999],
        num_computed_tokens_cpu=[103, 10, 999],
        block_table=SimpleNamespace(
            block_tables=[
                SimpleNamespace(
                    num_blocks_per_row=[14, 2, 999],
                    blocks_per_kv_block=2,
                )
            ]
        ),
    )
    scheduler = SimpleNamespace(
        num_scheduled_tokens={"short": 1, "long": 4, "waiting": 30},
        scheduled_spec_decode_tokens={"long": [8, 9, 10]},
    )
    return batch, scheduler


def record(inspector, batch, scheduler, *, upper_bounds=False):
    inspector.record(
        batch,
        scheduler,
        num_tokens_padded=8,
        num_reqs_padded=4,
        context_lengths_are_upper_bounds=upper_bounds,
    )


def read(inspector):
    return [json.loads(line) for line in inspector.path.read_text().splitlines()]


class TestBatchInspector(unittest.TestCase):
    def test_live_order_and_owned_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            inspector = BatchInspector(directory)
            batch, scheduler = inputs()
            record(inspector, batch, scheduler)
            batch.req_ids.clear()
            batch.num_prompt_tokens[0] = 9999
            batch.num_computed_tokens_cpu[0] = 9999
            batch.block_table.block_tables[0].num_blocks_per_row[0] = 9999
            scheduler.scheduled_spec_decode_tokens["long"].clear()
            self.assertTrue(inspector.close())
            row, summary = read(inspector)
            self.assertEqual(row["num_requests"], 2)
            self.assertEqual(row["num_tokens"], 5)
            self.assertEqual(row["num_requests_padded"], 4)
            self.assertEqual(row["num_tokens_padded"], 8)
            self.assertEqual(
                [r["request_id"] for r in row["requests"]], ["long", "short"]
            )
            self.assertEqual(
                row["requests"][0],
                {
                    "request_id": "long",
                    "prompt_tokens": 100,
                    "computed_tokens_cpu": 103,
                    "context_tokens_cpu": 107,
                    "query_tokens": 4,
                    "draft_tokens": 3,
                    "kv_blocks": [7],
                },
            )
            self.assertFalse(row["context_lengths_are_upper_bounds"])
            self.assertEqual(
                summary,
                {
                    "type": "summary",
                    "steps": 1,
                    "written": 1,
                    "dropped": 0,
                },
            )
            self.assertTrue(inspector.close())

    def test_upper_bounds_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            inspector = BatchInspector(directory)
            record(inspector, *inputs(), upper_bounds=True)
            self.assertTrue(inspector.close())
            self.assertTrue(read(inspector)[0]["context_lengths_are_upper_bounds"])

    def test_full_queue_drops_without_waiting_and_recovers(self):
        entered, release, recovered = Event(), Event(), Event()
        dumps = json.dumps

        def slow_dump(value):
            if value.get("step") == 1:
                entered.set()
                release.wait(5)
            if value.get("step") == 2:
                recovered.set()
            return dumps(value)

        with (
            tempfile.TemporaryDirectory() as directory,
            patch(
                "vllm.v1.worker.batch_inspect.json.dumps",
                side_effect=slow_dump,
            ),
        ):
            inspector = BatchInspector(directory, queue_capacity=1)
            try:
                record(inspector, *inputs())
                self.assertTrue(entered.wait(2))
                record(inspector, *inputs())  # Occupies the only queue slot.
                producer = Thread(target=record, args=(inspector, *inputs()))
                producer.start()
                producer.join(timeout=2)
                self.assertFalse(producer.is_alive())  # Writer still blocked.
                self.assertEqual(inspector.dropped, 1)
                release.set()
                self.assertTrue(recovered.wait(2))
                record(inspector, *inputs())
            finally:
                release.set()
                self.assertTrue(inspector.close())
            rows = read(inspector)
            self.assertEqual([r["step"] for r in rows[:-1]], [1, 2, 4])
            self.assertEqual(
                rows[-1],
                {
                    "type": "summary",
                    "steps": 4,
                    "written": 3,
                    "dropped": 1,
                },
            )

    def test_writer_failure_does_not_raise_in_producer(self):
        with tempfile.TemporaryDirectory() as directory:
            parent_file = Path(directory) / "file"
            parent_file.write_text("not a directory")
            with self.assertLogs("vllm.v1.worker.batch_inspect", level="WARNING"):
                inspector = BatchInspector(str(parent_file))
                self.assertTrue(inspector._failed.wait(2))
            record(inspector, *inputs())
            self.assertEqual(inspector.dropped, 1)
            self.assertFalse(inspector.close())

    def test_shutdown_wait_is_bounded(self):
        entered, release = Event(), Event()
        dumps = json.dumps

        def slow_dump(value):
            if value.get("type") == "batch":
                entered.set()
                release.wait(5)
            return dumps(value)

        with (
            tempfile.TemporaryDirectory() as directory,
            patch(
                "vllm.v1.worker.batch_inspect.json.dumps",
                side_effect=slow_dump,
            ),
        ):
            inspector = BatchInspector(directory)
            try:
                record(inspector, *inputs())
                self.assertTrue(entered.wait(2))
                self.assertFalse(inspector.close(timeout=0.01))
            finally:
                release.set()
                self.assertTrue(inspector.close())

    def test_queue_must_be_bounded(self):
        with self.assertRaises(ValueError):
            BatchInspector("unused", queue_capacity=0)


if __name__ == "__main__":
    unittest.main()
