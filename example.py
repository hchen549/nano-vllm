from examples_common import build_inputs, run_benchmark
from nanovllm import LLM


def main():
    _, _, _, model_path = build_inputs()
    llm = LLM(model_path, enforce_eager=True, tensor_parallel_size=1)
    run_benchmark(llm, label="v0 (LLM, single-process)")


if __name__ == "__main__":
    main()
