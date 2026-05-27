import os
import random
import time

import numpy as np
import torch

from nanovllm import LLM, SamplingParams
# from vllm import LLM, SamplingParams

SEED = 0


def seed_everything(seed: int) -> None:
    """Pin every RNG that could influence benchmark inputs or model
    behavior so the same seed always produces the same input batch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_inputs(num_seqs: int, max_input_len: int, max_output_len: int):
    """Build the deterministic benchmark batch.

    Uses a *local* random.Random instance so the benchmark inputs are
    not affected by any other code in the process touching the global
    random state (e.g. LLM init, tokenizer init, dataloader workers).
    """
    rng = random.Random(SEED)
    prompt_token_ids = [
        [rng.randint(0, 10000) for _ in range(rng.randint(100, max_input_len))]
        for _ in range(num_seqs)
    ]
    max_tokens_per_seq = [
        rng.randint(100, max_output_len) for _ in range(num_seqs)
    ]
    sampling_params = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=mt)
        for mt in max_tokens_per_seq
    ]
    return prompt_token_ids, sampling_params


def main():
    # Seed every global RNG before anything else (covers torch sampling,
    # numpy ops, and any library that reads the global random state).
    seed_everything(SEED)

    num_seqs = 256
    max_input_len = 1024
    max_ouput_len = 1024

    # Build inputs BEFORE constructing the LLM so the input batch can't
    # be perturbed by anything LLM init does to the global RNG.
    prompt_token_ids, sampling_params = build_inputs(
        num_seqs, max_input_len, max_ouput_len
    )

    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    llm = LLM(path, enforce_eager=False, max_model_len=4096)
    # uncomment the following line for vllm
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    llm.generate(["Benchmark: "], SamplingParams())
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = (time.time() - t)
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")


if __name__ == "__main__":
    main()
