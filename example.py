import os
from time import time

from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    start = time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time() - start

    total_tokens = sum(len(output["token_ids"]) for output in outputs)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")

    print("\n")
    print(f"Generation time: {elapsed:.2f}s")
    print(f"Total output tokens: {total_tokens}")
    print(f"Throughput: {total_tokens / elapsed:.2f} tokens/s")


if __name__ == "__main__":
    main()
