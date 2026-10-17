# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run on one GPU: python examples/observability/batch_inspect.py --output-dir DIR.

This logs three prompt lengths without a server; it does not measure GPU timing.
To check the logs, set VLLM_BATCH_INSPECT_DIR=DIR and run
python tests/v1/worker/test_batch_inspect.py -v.
"""

import argparse
import json
import os
from pathlib import Path


def close_inspector(worker):
    """RPC callback: GPUWorker owns the model runner and its BatchInspector."""
    return worker.model_runner.batch_inspector.close()


def main():
    """Configure worker logging, generate three requests, and drain the writers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    directory = args.output_dir.expanduser().resolve()
    if directory.exists() and any(directory.iterdir()):
        parser.error("Use an empty output directory")

    # These environment settings reach the GPU workers before vLLM starts them.
    os.environ["VLLM_BATCH_INSPECT_DIR"] = str(directory)
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    from vllm import LLM, SamplingParams

    # Synchronous scheduling gives exact CPU lengths; prefix reuse is disabled.
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
    # The model's tokenizer supplies one token; raw token IDs give exact lengths.
    token = llm.get_tokenizer().encode(" hello", add_special_tokens=False)[0]
    prompts = [{"prompt_token_ids": [token] * n} for n in (16, 64, 128)]
    outputs = llm.generate(
        prompts,
        SamplingParams(temperature=0, max_tokens=8, ignore_eos=True),
    )

    # collective_rpc invokes the callback on each worker, closing its local writer.
    if not all(llm.collective_rpc(close_inspector)):
        raise RuntimeError("Writer did not finish cleanly")
    # Each RequestOutput supplies an ID and prompt tokens for the live-log tests.
    expected = {out.request_id: len(out.prompt_token_ids) for out in outputs}
    (directory / "requests.json").write_text(json.dumps(expected), encoding="utf-8")
    print(f"Recorded {len(outputs)} requests in {directory}")


if __name__ == "__main__":
    main()
