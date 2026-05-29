# Engine Comparison: v0 (`LLM`) vs v1 (`LLM_MP`)

This folder contains a side-by-side benchmark of nano-vllm's two engine
backends. The two example scripts share `examples_common.py` so the workload
(prompts, sampling params, max tokens) is **identical** between runs — any
throughput delta is the engine, not the input.

| File | Engine | Description |
|---|---|---|
| `example.py` | `LLM` (v0) | Single-process. Tokenize → step → step → … → detokenize. |
| `example_v1.py` | `LLM_MP` (v1) | Engine in subprocess via ZMQ. Tokenize/detokenize run in thread pools and pipeline with the GPU. |
| `examples_common.py` | — | Shared `make_prompts`, `build_inputs`, `run_benchmark`. |

## Architecture diff

```
v0 (single-process)                v1 (subprocess + thread pools)
────────────────────────────       ─────────────────────────────────────────
[ main thread                ]     [ main thread          ] [ engine subproc ]
   tokenize all 128 prompts          tokenize  ──┐               GPU step
   ↓                                              ▼ ZMQ ADD          ↓
   step(): GPU prefill              (8 thread workers)            scheduler
   step(): GPU decode               ↓                                ↓
   …                                ZMQ recv DONE ──┐             ZMQ DONE
   step(): GPU decode                                ▼               ↑
   detokenize all 128 outputs       (4 thread workers)              GPU
                                    decode text                     decode
                                    write into results dict           …
```

v0 work is serial. v1 overlaps tokenize → GPU → detokenize as a 3-stage
pipeline.

## Benchmark workload

Defined in `examples_common.py`:

- Prompts derived from a ~1.5 KB passage about the history of computing,
  sliced into 500-character overlapping chunks, wrapped in a chat template.
- `temperature=0.6`, `max_tokens=256`, `ignore_eos=False`.

`ignore_eos=False` is **load-bearing** for v1 — it makes sequences hit EOS at
varied decode steps, so DONE messages stream out throughout the run and the
detokenizer thread pool actually has GPU work to overlap with. With
`ignore_eos=True`, every sequence finishes in the same step and v1's
pipelining has nothing to hide behind.

## Results (Qwen3-0.6B, single GPU, `enforce_eager=True`)

| Workload (num prompts) | v0 e2e | v1 e2e | Δ time | v0 output tok/s | v1 output tok/s | v1 speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 5.12s | 5.15s | +0.6% | 1602 | 1590 | ~tie |
| 1,280 | 13.75s | 13.00s | −5.5% | 5957 | 6300 | **+5.7%** |
| 6,280 | 53.38s | 50.42s | −5.5% | 7529 | 7971 | **+5.9%** |

Combined throughput (input + output tokens / elapsed):

| Workload | v0 | v1 |
|---:|---:|---:|
| 128 | 4362 tok/s | 4331 tok/s |
| 1,280 | 16,186 tok/s | 17,117 tok/s |
| 6,280 | 20,454 tok/s | 21,657 tok/s |

## Reading the numbers

### Small workload (128 prompts) — v0 ≈ v1

v1's overhead (subprocess startup, ZMQ pickle round-trips per ADD/DONE,
and a "split first prefill" — see below) is amortized across only ~5s of GPU
work. Tokenize + detokenize for 128 short prompts is ~50–100ms total, so
even ideal pipelining can hide at most ~1–2% of wall-clock. Result is within
measurement noise.

### Medium workload (1,280 prompts) — v1 starts winning

Tokenize cost grows linearly with prompts; detok cost grows linearly with
**finished sequences × output length**. With 1,280 prompts × 256 max tokens,
we're decoding ~80k tokens. Detok for 80k tokens is hundreds of ms. v0 pays
this serially after the GPU is done; v1 spreads it across the run. Win:
**~5.5% lower e2e**, ~6% higher output throughput.

### Large workload (6,280 prompts) — v1 win is stable, not growing

The win plateaus at ~5–6% because at this scale the GPU is the bottleneck
and tokenize/detokenize is a fixed-fraction CPU tax that v1 eliminates. v1
buys back almost exactly that fraction.

## Why v1 doesn't win more dramatically here

1. **GPU dominates.** Qwen3-0.6B in eager mode at batch 128–6280 spends >90%
   of wall-clock in the model. There simply isn't more CPU-side cost for v1
   to hide.
2. **Split first prefill.** v1's engine wakes up as soon as the *first*
   request arrives (not all of them) — you'll see the trace start with a
   tiny prefill (e.g., `batch=2`) followed by a big one (`batch=126`). The
   small prefill is wasted forward-pass overhead that v0 avoids by enqueuing
   everything synchronously before the first step.
3. **ZMQ + pickle round-trips.** Every ADD and every DONE crosses a process
   boundary. Each round-trip is microseconds, but with thousands of requests
   they sum to single-digit ms.

Where v1 would win much more:
- Streaming / interactive serving where output tokens go back to the
  caller as they're produced (v0 has no streaming path at all).
- CUDA-graph mode with very large batches where CPU-side scheduler logic
  becomes a bottleneck — moving it off the GPU process matters more.
- Tokenizers that are slow Python (not HF fast/Rust) — the pool would
  outweigh the GIL serialization.

## Running

From the repo root:

```bash
uv run examples/engine_comparison/example.py        # v0
uv run examples/engine_comparison/example_v1.py     # v1
```

Override the workload size:

```bash
NANOVLLM_NUM_PROMPTS=1280 uv run examples/engine_comparison/example_v1.py
NANOVLLM_NUM_PROMPTS=6280 uv run examples/engine_comparison/example_v1.py
```

The model path defaults to `~/huggingface/Qwen3-0.6B/`. Edit
`DEFAULT_MODEL_PATH` in `examples_common.py` to point elsewhere.
