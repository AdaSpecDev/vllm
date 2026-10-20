# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only checks: python tests/v1/worker/test_batch_statistics.py -v."""

import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from vllm.v1.engine import EngineCoreOutput, FinishReason
from vllm.v1.engine.output_processor import OutputProcessor
from vllm.v1.metrics.loggers import LoggingStatLogger
from vllm.v1.metrics.stats import IterationStats, RequestStateStats
from vllm.v1.spec_decode.metrics import SpecDecodingStats
from vllm.v1.worker.batch_inspect import BatchInspector


def records(writer):
    """Read plain statistics after the background writer has finished."""
    return [json.loads(line) for line in writer.path.read_text().splitlines()]


class TestNativeStatistics(unittest.TestCase):
    def test_native_request_measurements_and_id_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = BatchInspector(directory, role="frontend")
            processor = OutputProcessor.__new__(OutputProcessor)
            processor.inspection_logger = writer
            processor.lora_states = Mock()
            stats = RequestStateStats(arrival_time=10.0)
            stats.queued_ts, stats.scheduled_ts = 98.0, 99.0
            request = SimpleNamespace(
                request_id="request-randomsuffix",
                external_req_id="request",
                is_prefilling=True,
                stats=stats,
                lora_name=None,
                prompt_len=16,
                max_tokens_param=8,
                num_cached_tokens=0,
                parent_req=None,
            )
            iteration = IterationStats()
            iteration.iteration_timestamp = 14.0
            output = EngineCoreOutput(
                request_id=request.request_id, new_token_ids=[1], inspection_step_id=42
            )
            processor._update_stats_from_output(request, output, 100.0, iteration)
            request.is_prefilling = False
            iteration.iteration_timestamp = 17.0
            output.new_token_ids = [2, 3]
            processor._update_stats_from_output(request, output, 103.0, iteration)
            iteration.iteration_timestamp = 20.0
            processor._update_stats_from_finished(
                request, FinishReason.LENGTH, iteration
            )
            self.assertTrue(writer.close())
            first, following, finished, summary = records(writer)
            self.assertEqual(first["step"], 42)
            self.assertEqual(following["step"], 42)
            self.assertEqual(first["time_to_first_token_seconds"], 4.0)
            self.assertEqual(following["inter_output_latency_seconds"], 3.0)
            self.assertEqual(following["emitted_tokens"], 2)
            self.assertEqual(finished["request_id"], "request")
            self.assertEqual(finished["internal_request_id"], request.request_id)
            self.assertEqual(finished["request_latency_seconds"], 10.0)
            self.assertEqual(finished["queue_wait_seconds"], 1.0)
            self.assertEqual(finished["generated_tokens"], 3)
            self.assertEqual(finished["mean_time_per_output_token_seconds"], 1.5)
            self.assertEqual(summary["statistics_written"], 3)

    def test_throughput_keeps_its_native_reporting_window(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = BatchInspector(directory, role="frontend")
            logger = LoggingStatLogger.__new__(LoggingStatLogger)
            logger.inspection_logger = writer
            logger.engine_index = 0
            logger.last_log_time = 10.0
            logger.num_prompt_tokens = 16
            logger.num_generation_tokens = 12
            logger.last_prompt_throughput = logger.last_generation_throughput = 0.0
            with patch("vllm.v1.metrics.loggers.time.monotonic", return_value=14.0):
                logger._update_stats()
            self.assertTrue(writer.close())
            record = records(writer)[0]
            self.assertEqual(record["window_start_monotonic_seconds"], 10.0)
            self.assertEqual(record["window_end_monotonic_seconds"], 14.0)
            self.assertEqual(record["generated_tokens"], 12)
            self.assertEqual(record["generated_tokens_per_second"], 3.0)
            self.assertEqual(logger.num_generation_tokens, 0)

    def test_engine_counters_and_batch_share_step_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = BatchInspector(directory)
            batch = SimpleNamespace(
                req_ids=["request-internal"],
                num_computed_tokens_cpu=[16],
                num_prompt_tokens=[16],
                block_table=SimpleNamespace(block_tables=[]),
            )
            scheduler = SimpleNamespace(
                inspection_step_id=42,
                num_scheduled_tokens={"request-internal": 4},
                scheduled_spec_decode_tokens={"request-internal": [1, 2, 3]},
            )
            writer.enqueue_or_drop(
                batch,
                scheduler,
                step_id=42,
                num_tokens_padded=4,
                num_reqs_padded=1,
                context_lengths_are_upper_bounds=False,
            )
            speculative = SpecDecodingStats.new(3)
            speculative.observe_draft(num_draft_tokens=3, num_accepted_tokens=2)
            output = SimpleNamespace(
                scheduler_stats=SimpleNamespace(spec_decoding_stats=speculative),
                outputs=[
                    EngineCoreOutput(
                        request_id="request-internal", new_token_ids=[1, 2, 3]
                    )
                ],
            )
            writer.record_engine_step(scheduler, {0: output})
            speculative.num_accepted_tokens_per_pos.clear()
            self.assertTrue(writer.close())
            metadata, counters, summary = records(writer)
            self.assertEqual(metadata["step"], counters["step"])
            self.assertEqual(counters["step"], 42)
            self.assertEqual(output.outputs[0].inspection_step_id, 42)
            self.assertEqual(counters["speculative"]["num_draft_tokens"], 3)
            self.assertEqual(counters["speculative"]["num_accepted_tokens"], 2)
            self.assertEqual(
                counters["speculative"]["num_accepted_tokens_per_pos"], [1, 1, 0]
            )
            self.assertEqual(counters["requests"][0]["emitted_tokens"], 3)
            self.assertEqual(summary["dropped"], 0)


if __name__ == "__main__":
    unittest.main()
