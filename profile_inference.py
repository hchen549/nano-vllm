"""Profile CUDA and CPU activity during nano-vllm inference, broken down
by *phase*: tokenization, scheduling, and GPU inference.

Outputs:
  - profile_trace.json : Chrome trace file. Open with chrome://tracing
    or https://ui.perfetto.dev to see a timeline of CPU and CUDA activity.
  - Console summary tables for top ops by CPU time and CUDA time.
  - Console phase breakdown showing wall-clock time spent in each phase.

------------------------------------------------------------------
HOW TO READ THE TRACE (chrome://tracing or https://ui.perfetto.dev)
------------------------------------------------------------------
The trace is split into rows ("tracks") grouped by process and thread.

CPU side (host, Python process):
  * "python" / main thread row
      - Python-level call stack: nanovllm scheduler, model_runner.run,
        model forward passes, sampler, etc. Each torch op (matmul,
        softmax, layer_norm, ...) appears as a colored block here.
      - This is where you see CPU time spent dispatching kernels,
        building tensors, running the scheduler, and Python overhead.
  * Custom phase labels added by this script (record_function blocks),
    visible as colored bars on the CPU track:
      - "warmup_generate"          : the warm-up generate() call
      - "profiled_generate"        : the entire profiled inference window
      - "prompt_<i>"               : per-prompt region
      - "tokenize_encode"          : tokenizer.encode (str -> ids)
                                     happens once per prompt before the
                                     scheduling loop starts
      - "tokenize_decode"          : tokenizer.decode (ids -> str)
                                     happens once per prompt at the end
      - "scheduler_schedule"       : Scheduler.schedule() — picks which
                                     sequences run this step (prefill
                                     vs decode batch construction)
      - "scheduler_postprocess"    : Scheduler.postprocess() — appends
                                     sampled tokens, frees finished seqs
      - "gpu_inference_step"       : ModelRunner.call("run", ...) —
                                     the actual forward + sampling step.
                                     Each one should line up with a burst
                                     of CUDA kernels on the GPU row below.
        Use these as anchors to find phases on the timeline.
  * Other CPU threads may show up for tokenizer / dataloader work.

GPU side (CUDA, one row per CUDA stream):
  * "stream 7" (or similar) row
      - Actual CUDA kernel executions: matmul / GEMM kernels,
        flash-attention, layernorm, sampling top-k, memcpy, etc.
      - Names typically look like
        "ampere_fp16_s16816gemm_...", "void at::native::...",
        "flash_fwd_kernel<...>", "Memcpy DtoH", etc.
      - These kernels execute *inside* "gpu_inference_step" regions on
        the CPU row. Click a CPU kernel-launch op to draw the flow
        arrow connecting to the actual GPU kernel.
  * Gaps on the GPU row = GPU is idle waiting on host (a sign your
    workload is CPU-bound or has bad batching). Gaps on the CPU row
    while the GPU is busy = host is waiting on the device.

Memory track (because profile_memory=True):
  * Allocations / frees show up as a memory-usage curve, broken down
    by device (CPU vs CUDA:0). Useful to spot KV-cache growth and
    activation memory peaks.

What to look for:
  * Long contiguous blocks on the GPU row inside "gpu_inference_step"
    = healthy kernel utilization. Many tiny blocks with gaps = launch
    overhead dominated, often fixable with CUDA graphs or larger batches.
  * Wide "scheduler_*" or "tokenize_*" blocks with no GPU activity
    below = CPU-side overhead is meaningful. Compare phase totals in
    the console output for the big picture.
  * Repeating pattern of small kernels = decode steps; one big burst
    at the start = prefill.
------------------------------------------------------------------
"""

import os
from collections import defaultdict
from time import perf_counter

import torch
from nanovllm import LLM, SamplingParams
from torch.profiler import profile, ProfilerActivity, record_function
from transformers import AutoTokenizer


# Wall-clock totals per phase, populated by instrument_engine().
PHASE_TOTALS: dict[str, float] = defaultdict(float)
PHASE_CALLS: dict[str, int] = defaultdict(int)


def _timed(name: str, fn):
    """Wrap fn so every call is labeled in the trace AND timed on the host."""

    def wrapped(*args, **kwargs):
        t = perf_counter()
        with record_function(name):
            try:
                return fn(*args, **kwargs)
            finally:
                # GPU work launched by fn may still be in flight; for the
                # phase totals we synchronize so the wall-clock reflects
                # actual completion, not just dispatch.
                if torch.cuda.is_available() and name == "gpu_inference_step":
                    torch.cuda.synchronize()
                PHASE_TOTALS[name] += perf_counter() - t
                PHASE_CALLS[name] += 1

    return wrapped


def instrument_engine(llm: LLM) -> None:
    """Monkey-patch the engine so tokenization / scheduling / GPU steps
    each appear as distinct labeled regions in the trace and get
    wall-clock totals."""

    # --- Tokenization ---
    orig_encode = llm.tokenizer.encode
    orig_decode = llm.tokenizer.decode
    llm.tokenizer.encode = _timed("tokenize_encode", orig_encode)
    llm.tokenizer.decode = _timed("tokenize_decode", orig_decode)

    # --- Scheduling ---
    orig_schedule = llm.scheduler.schedule
    orig_postprocess = llm.scheduler.postprocess
    llm.scheduler.schedule = _timed("scheduler_schedule", orig_schedule)
    llm.scheduler.postprocess = _timed("scheduler_postprocess", orig_postprocess)

    # --- GPU inference ---
    # ModelRunner.call("run", ...) is the entry point that runs the
    # forward pass + sampling on the GPU. Other call() invocations
    # (e.g. "exit") should NOT count as inference, so we wrap selectively.
    orig_call = llm.model_runner.call

    def call_wrapped(method, *args, **kwargs):
        if method == "run":
            return _timed("gpu_inference_step", orig_call)(method, *args, **kwargs)
        return orig_call(method, *args, **kwargs)

    llm.model_runner.call = call_wrapped


def reset_phase_totals() -> None:
    PHASE_TOTALS.clear()
    PHASE_CALLS.clear()


def print_phase_breakdown(total_wall: float) -> None:
    print("\n=== Phase breakdown (wall-clock, host-measured) ===")
    print(f"{'phase':<28} {'calls':>8} {'total_s':>10} {'%':>7}")
    print("-" * 56)
    accounted = 0.0
    for name in (
        "tokenize_encode",
        "scheduler_schedule",
        "gpu_inference_step",
        "scheduler_postprocess",
        "tokenize_decode",
    ):
        secs = PHASE_TOTALS.get(name, 0.0)
        calls = PHASE_CALLS.get(name, 0)
        pct = 100.0 * secs / total_wall if total_wall > 0 else 0.0
        accounted += secs
        print(f"{name:<28} {calls:>8d} {secs:>10.4f} {pct:>6.2f}%")
    other = max(0.0, total_wall - accounted)
    print(
        f"{'(other / overhead)':<28} {'-':>8} {other:>10.4f} "
        f"{100.0 * other / total_wall if total_wall > 0 else 0.0:>6.2f}%"
    )
    print("-" * 56)
    print(f"{'TOTAL':<28} {'':>8} {total_wall:>10.4f} {100.0:>6.2f}%")
    print(
        "\nNotes:\n"
        "  * gpu_inference_step is host-side wall time INCLUDING a "
        "torch.cuda.synchronize()\n"
        "    so it reflects actual GPU completion, not just dispatch.\n"
        "  * 'other / overhead' includes Python loop work, tqdm, "
        "list/dict bookkeeping,\n"
        "    and anything not explicitly wrapped.\n"
        "  * For per-kernel GPU time (matmul, attention, etc.), see the\n"
        "    'Top 20 ops by cuda_time_total' table above and the GPU row\n"
        "    inside gpu_inference_step regions in the Chrome trace.\n"
    )


def main():
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    # Add labeled regions + phase timers to the engine.
    instrument_engine(llm)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=64)
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

    # Warm-up run so kernels are compiled / cached before profiling.
    with record_function("warmup_generate"):
        llm.generate(prompts, sampling_params, use_tqdm=False)

    # Reset totals so the warm-up doesn't inflate the phase breakdown.
    reset_phase_totals()

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
        print("CUDA available, profiling CUDA kernels")

    trace_path = "profile_trace.json"

    wall_start = perf_counter()
    with profile(
        activities=activities,
        record_shapes=True,
        with_stack=False,
        profile_memory=True,
    ) as prof:
        # Top-level marker for the whole profiled inference window.
        with record_function("profiled_generate"):
            outputs = []
            for i, prompt in enumerate(prompts):
                with record_function(f"prompt_{i}"):
                    outputs.extend(
                        llm.generate([prompt], sampling_params, use_tqdm=False)
                    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_wall = perf_counter() - wall_start

    # Export Chrome trace for timeline visualization.
    prof.export_chrome_trace(trace_path)
    print(f"\nChrome trace written to: {os.path.abspath(trace_path)}")
    print("View it at: chrome://tracing  or  https://ui.perfetto.dev")
    print("See the docstring at the top of this file for a legend\n")

    # Per-op tables (kernel level).
    sort_key = "cuda_time_total" if torch.cuda.is_available() else "cpu_time_total"
    print(
        "=== Top 20 ops by",
        sort_key,
        "(GPU kernel time — what the device actually executed) ===",
    )
    print(prof.key_averages().table(sort_by=sort_key, row_limit=20))

    print(
        "\n=== Top 20 ops by cpu_time_total "
        "(host time — Python + dispatch + CPU work) ==="
    )
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

    # Phase breakdown (tokenization vs scheduling vs GPU inference).
    print_phase_breakdown(total_wall)

    for prompt, output in zip(prompts, outputs):
        print("\nPrompt:", repr(prompt))
        print("Completion:", repr(output["text"]))


if __name__ == "__main__":
    main()
