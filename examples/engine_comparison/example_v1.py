from examples_common import build_inputs, run_benchmark
from nanovllm import LLM_MP


def main():
    _, _, _, model_path = build_inputs()
    llm = LLM_MP(model_path, enforce_eager=True, tensor_parallel_size=1)
    run_benchmark(llm, label="v1 (LLM_MP, subprocess + threaded tok/detok)")


if __name__ == "__main__":
    main()
