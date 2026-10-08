# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-GPU logging check, without a server or GPU timing.

Run from the fork checkout with its environment activated:
    python examples/observability/batch_inspect.py --output-dir /path/to/new/run

Uses a small pinned model and three prompt lengths to check real prefill/decode
batches. This is a smoke check, not a controlled speculative-decoding experiment.
"""

import argparse
import json
import os
from pathlib import Path


def close_inspector(worker):
    return worker.model_runner.batch_inspector.close()


def validate(paths, outputs):
    expected = {out.request_id: len(out.prompt_token_ids) for out in outputs}
    seen = set()
    live_decode_batch = False
    for path in paths:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert rows and rows[-1]["type"] == "summary", "Incomplete writer output"
        batches, summary = rows[:-1], rows[-1]
        assert summary["dropped"] == 0, "Records dropped; rerun this small check"
        assert summary["steps"] == summary["written"] == len(batches)
        assert [b["step"] for b in batches] == list(range(1, len(batches) + 1))
        previous_context = {}
        for batch in batches:
            requests = batch["requests"]
            assert not batch["context_lengths_are_upper_bounds"]
            assert not batch["microbatched"]
            assert batch["num_requests"] == len(requests)
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
        print(f"Validated {len(batches)} live forwards: {path}")
    assert seen == set(expected), "Missing live request metadata"
    assert live_decode_batch, "Did not observe a multi-request decode batch"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    directory = args.output_dir.expanduser().resolve()
    if directory.exists() and any(directory.iterdir()):
        parser.error("Use an empty output directory")
    os.environ["VLLM_BATCH_INSPECT_DIR"] = str(directory)
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"

    from vllm import LLM, SamplingParams

    llm = LLM(
        model="Qwen/Qwen3-0.6B",
        revision="c1899de289a04d12100db370d81485cdf75e47ca",
        async_scheduling=False,
        enable_prefix_caching=False,
        enforce_eager=True,
        max_model_len=512,
        max_num_batched_tokens=256,
        max_num_seqs=3,
        seed=0,
    )
    tokenizer = llm.get_tokenizer()
    token = tokenizer.encode(" hello", add_special_tokens=False)[0]
    prompts = [{"prompt_token_ids": [token] * n} for n in (16, 64, 128)]
    outputs = llm.generate(
        prompts,
        SamplingParams(temperature=0, max_tokens=8, ignore_eos=True),
    )
    assert all(llm.collective_rpc(close_inspector)), "Writer did not finish cleanly"
    validate(list(directory.glob("batch-*.jsonl")), outputs)


if __name__ == "__main__":
    main()
