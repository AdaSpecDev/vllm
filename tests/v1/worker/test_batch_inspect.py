# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only: run with python tests/v1/worker/test_batch_inspect.py -v."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace
from unittest.mock import patch

from vllm.v1.worker.batch_inspect import BatchInspector
from vllm.v1.worker.batch_metadata import BatchMetadata


def inputs():
    """Fake CPU batch rows include padding and a scheduler-only waiting request."""
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


def record(inspector, batch, scheduler, *, upper_bounds=False, num_tokens_padded=8):
    """Supply graph padding and context semantics to the inspector entry point."""
    return inspector.enqueue_or_drop(
        batch,
        scheduler,
        num_tokens_padded=num_tokens_padded,
        num_reqs_padded=4,
        context_lengths_are_upper_bounds=upper_bounds,
    )


def read(inspector):
    """Read JSONL batch records and the final summary after the writer closes."""
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
            self.assertFalse(row["microbatched"])
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
                self.assertIsInstance(inspector._queue.queue[0], BatchMetadata)
                # A dropped record must skip even metadata construction.
                with patch.object(BatchMetadata, "from_batch") as capture:
                    self.assertFalse(record(inspector, *inputs()))
                    capture.assert_not_called()
                producer = Thread(target=record, args=(inspector, *inputs()))
                producer.start()
                producer.join(timeout=2)
                self.assertFalse(producer.is_alive())  # Writer still blocked.
                self.assertEqual(inspector.dropped, 2)
                release.set()
                self.assertTrue(recovered.wait(2))
                record(inspector, *inputs())
            finally:
                release.set()
                self.assertTrue(inspector.close())
            rows = read(inspector)
            self.assertEqual([r["step"] for r in rows[:-1]], [1, 2, 5])
            self.assertEqual(
                rows[-1],
                {
                    "type": "summary",
                    "steps": 5,
                    "written": 3,
                    "dropped": 2,
                },
            )

    def test_writer_failure_does_not_raise_in_producer(self):
        with tempfile.TemporaryDirectory() as directory:
            parent_file = Path(directory) / "file"
            parent_file.write_text("not a directory")
            with self.assertLogs("vllm.v1.worker.batch_inspect", level="WARNING"):
                inspector = BatchInspector(str(parent_file))
                self.assertTrue(inspector._failed.wait(2))
                self.assertFalse(inspector.close())  # Wait for the warning too.
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


def validate_live_batches(paths, expected):
    """Compare worker JSONL records with the example's RequestOutput manifest."""
    if not paths:
        raise AssertionError("No worker batch logs found")
    seen = set()
    live_decode_batch = False
    for path in paths:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert rows and rows[-1]["type"] == "summary", "Incomplete writer output"
        batches, summary = rows[:-1], rows[-1]
        assert summary["dropped"] == 0, "Records dropped; rerun this small check"
        assert summary["steps"] == summary["written"] == len(batches)
        assert [b["step"] for b in batches] == list(range(1, len(batches) + 1))
        previous_context: dict[str, int] = {}
        for batch in batches:
            requests = batch["requests"]
            assert not batch["context_lengths_are_upper_bounds"]
            assert not batch["microbatched"]
            assert batch["num_requests"] == len(requests)
            assert batch["num_requests_padded"] >= batch["num_requests"]
            assert batch["num_tokens_padded"] >= batch["num_tokens"]
            assert batch["num_tokens"] == sum(r["query_tokens"] for r in requests)
            for req in requests:
                req_id = req["request_id"]
                assert req["prompt_tokens"] == expected[req_id]
                assert req["query_tokens"] > 0 and req["draft_tokens"] == 0
                computed = req["computed_tokens_cpu"]
                assert computed == previous_context.get(req_id, 0)
                assert req["context_tokens_cpu"] == computed + req["query_tokens"]
                previous_context[req_id] = req["context_tokens_cpu"]
                seen.add(req_id)
            live_decode_batch |= len(requests) > 1 and all(
                r["computed_tokens_cpu"] >= r["prompt_tokens"] for r in requests
            )
    assert seen == set(expected), "Missing live request metadata"
    assert live_decode_batch, "Did not observe a multi-request decode batch"


class TestOfflineExample(unittest.TestCase):
    def test_validator_accepts_coherent_batches_and_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            inspector = BatchInspector(directory)
            batch, scheduler = inputs()
            batch.num_computed_tokens_cpu[:2] = [0, 0]
            scheduler.num_scheduled_tokens = {"long": 100, "short": 10}
            scheduler.scheduled_spec_decode_tokens = {}
            record(inspector, batch, scheduler, num_tokens_padded=128)
            batch.num_computed_tokens_cpu[:2] = [100, 10]
            scheduler.num_scheduled_tokens = {"long": 1, "short": 1}
            record(inspector, batch, scheduler)
            self.assertTrue(inspector.close())
            expected = {"long": 100, "short": 10}
            validate_live_batches([inspector.path], expected)
            rows = read(inspector)
            rows[1]["requests"][0]["computed_tokens_cpu"] = 999
            inspector.path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaises(AssertionError):
                validate_live_batches([inspector.path], expected)
            with self.assertRaises(AssertionError):
                validate_live_batches([], expected)

    @unittest.skipUnless(
        os.getenv("VLLM_BATCH_INSPECT_DIR"), "Run the GPU example first"
    )
    def test_logs_from_gpu_example(self):
        # Check a completed offline run without submitting another GPU job.
        directory = Path(os.environ["VLLM_BATCH_INSPECT_DIR"]).expanduser()
        expected = json.loads((directory / "requests.json").read_text())
        self.assertEqual(sorted(expected.values()), [16, 64, 128])
        request_records = [
            json.loads(line)
            for path in directory.glob("frontend-*.jsonl")
            for line in path.read_text().splitlines()
        ]
        internal_expected = {
            row["internal_request_id"]: expected[row["request_id"]]
            for row in request_records
            if row["type"] == "request"
        }
        self.assertEqual(len(internal_expected), len(expected))
        validate_live_batches(list(directory.glob("batch-*.jsonl")), internal_expected)


if __name__ == "__main__":
    unittest.main()
