"""Shared helpers for example.py (v0) and example_v1.py (v1).

Keeps the two examples in lock-step so any throughput difference reflects the
engine, not the workload.
"""

from __future__ import annotations

import os
from time import time
from typing import Any

from nanovllm import SamplingParams
from transformers import AutoTokenizer


# A reasonably large passage. We slice it into many overlapping chunks so each
# prompt has non-trivial tokenization cost — this is where v1's pipelined
# tokenizer/detokenizer shows its advantage over v0's serial design.
PASSAGE = (
    "The history of computing hardware covers the developments from early "
    "simple devices to aid calculation, to modern day computers. Before the "
    "20th century, most calculations were done by humans. Early mechanical "
    "tools to help humans with digital calculations, such as the abacus, were "
    "referred to as calculating machines or calculators. The machine operator "
    "was called the computer. The first aids to computation were purely "
    "mechanical devices which required the operator to set up the initial "
    "values of an elementary arithmetic operation, then manipulate the device "
    "to obtain the result. Later, computers represented numbers in a "
    "continuous form, for instance distance along a scale, rotation of a "
    "shaft, or a voltage. Numbers could also be represented in the form of "
    "digits, automatically manipulated by a mechanism. Although this approach "
    "generally required more complex mechanisms, it greatly increased the "
    "precision of results. The development of transistor technology and then "
    "the integrated circuit chip led to a series of breakthroughs, starting "
    "with transistor computers and then integrated circuit computers, causing "
    "digital computers to largely replace analog computers. Metal-oxide "
    "semiconductor large-scale integration then enabled semiconductor memory "
    "and the microprocessor, leading to another key breakthrough, the "
    "miniaturized personal computer in the 1970s. The cost of computers "
    "gradually became so low that personal computers by the 1990s, and then "
    "mobile computers (smartphones and tablets) in the 2000s, became "
    "ubiquitous. "
)

DEFAULT_MODEL_PATH = "~/huggingface/Qwen3-0.6B/"
DEFAULT_NUM_PROMPTS = 128
DEFAULT_CHUNK_CHARS = 500
DEFAULT_MAX_TOKENS = 64
DEFAULT_TEMPERATURE = 0.6


def make_prompts(num_prompts: int, chunk_chars: int = DEFAULT_CHUNK_CHARS) -> list[str]:
    """Build many varied prompts by sliding a window over the passage."""
    base = (PASSAGE * 8).strip()
    prompts: list[str] = []
    for i in range(num_prompts):
        start = (i * 37) % max(1, len(base) - chunk_chars)
        chunk = base[start : start + chunk_chars]
        prompts.append(
            f"Read this passage and summarize it in two sentences:\n\n{chunk}"
        )
    return prompts


def build_inputs(
    model_path: str | None = None,
    num_prompts: int | None = None,
) -> tuple[Any, list[str], SamplingParams, str]:
    """Return (tokenizer, chat_prompts, sampling_params, resolved_model_path).

    Both example.py (v0) and example_v1.py (v1) call this so they receive
    *exactly* the same prompts and sampling params.
    """
    path = os.path.expanduser(model_path or DEFAULT_MODEL_PATH)
    n = num_prompts or int(
        os.environ.get("NANOVLLM_NUM_PROMPTS", str(DEFAULT_NUM_PROMPTS))
    )

    tokenizer = AutoTokenizer.from_pretrained(path)
    sampling_params = SamplingParams(
        temperature=DEFAULT_TEMPERATURE,
        max_tokens=DEFAULT_MAX_TOKENS,
        ignore_eos=True,
    )

    raw_prompts = make_prompts(n)
    chat_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in raw_prompts
    ]
    return tokenizer, chat_prompts, sampling_params, path


def run_benchmark(llm: Any, label: str, model_path: str | None = None) -> None:
    """Generate against `llm` with the shared workload and print a summary.

    `llm` must expose `.generate(prompts, sampling_params)` returning a list of
    {"text": str, "token_ids": list[int]} dicts. Both LLM and LLM_MP qualify.
    """
    tokenizer, prompts, sampling_params, _ = build_inputs(model_path)

    print(
        f"[{label}] Submitting {len(prompts)} prompts "
        f"(chunk ≈ {DEFAULT_CHUNK_CHARS} chars each)...",
        flush=True,
    )

    start = time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time() - start

    total_input_tokens = sum(len(tokenizer.encode(p)) for p in prompts)
    total_output_tokens = sum(len(o["token_ids"]) for o in outputs)

    # Print only first 2 completions to keep output readable.
    for i, (prompt, output) in enumerate(zip(prompts, outputs)):
        if i >= 2:
            break
        print()
        print(f"Prompt[{i}]: {prompt[:120]!r}...")
        print(f"Completion[{i}]: {output['text']!r}")

    print(f"\n========== {label} ==========")
    print(f"Num prompts:         {len(prompts)}")
    print(f"Total input tokens:  {total_input_tokens}")
    print(f"Total output tokens: {total_output_tokens}")
    print(f"End-to-end time:     {elapsed:.2f}s")
    print(f"Output throughput:   {total_output_tokens / elapsed:.2f} tok/s")
    print(
        f"Combined throughput: "
        f"{(total_input_tokens + total_output_tokens) / elapsed:.2f} tok/s"
    )
