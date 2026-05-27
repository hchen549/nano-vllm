"""Two-process LLM engine: tokenizer/detokenizer in this process, model and
scheduler in a child process. Communicates via EngineCoreClient.

Drop-in replacement for nanovllm.engine.llm_engine.LLMEngine.
"""

from __future__ import annotations

import atexit
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields

from nanovllm.config import Config
from nanovllm.engine.engine_core_client import EngineCoreClient
from nanovllm.engine.ipc import MsgType
from nanovllm.sampling_params import SamplingParams
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase


class TokenizerWorker:
    """Parallel tokenization → client.add_request.

    HuggingFace fast tokenizers release the GIL in Rust, so threads scale.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        client: EngineCoreClient,
        max_workers: int = 8,
    ):
        self.tokenizer = tokenizer
        self.client = client
        self.pool = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="tokenize"
        )

    def submit(self, rid: int, prompt, sp: SamplingParams) -> None:
        self.pool.submit(self._tokenize_and_send, rid, prompt, sp)

    def _tokenize_and_send(self, rid: int, prompt, sp: SamplingParams) -> None:
        token_ids = self.tokenizer.encode(prompt) if isinstance(prompt, str) else prompt
        self.client.add_request(rid, token_ids, sp)

    def flush(self) -> None:
        self.pool.shutdown(wait=True)


class DetokenizerWorker:
    """Parallel detokenization off the recv loop."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_workers: int = 4):
        self.tokenizer = tokenizer
        self.pool = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="detokenize"
        )
        self.results: dict[int, dict] = {}

    def submit(self, rid: int, token_ids: list[int], on_done) -> None:
        self.pool.submit(self._decode, rid, token_ids, on_done)

    def _decode(self, rid: int, token_ids: list[int], on_done) -> None:
        self.results[rid] = {
            "text": self.tokenizer.decode(token_ids),
            "token_ids": token_ids,
        }
        on_done()

    def flush(self) -> None:
        self.pool.shutdown(wait=True)


class LLMEngineMP:
    """Drop-in replacement for the in-process LLMEngine.

    Public API matches LLMEngine.generate() so callers only need to swap
    the import.
    """

    def __init__(self, model: str, **kwargs):
        cfg_fields = {f.name for f in fields(Config)}
        cfg_kwargs: dict[str, Unknown] = {k: v for k, v in kwargs.items() if k in cfg_fields}
        self.config = Config(model, **cfg_kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model, use_fast=True)
        self.config.eos = self.tokenizer.eos_token_id

        # Single point of contact for the engine subprocess.
        self.client = EngineCoreClient(self.config)

        self.tok_worker = TokenizerWorker(self.tokenizer, self.client)
        self.detok_worker = DetokenizerWorker(self.tokenizer)

        self._closed = False
        atexit.register(self.exit)

    # ──────────────────────────── public API ────────────────────────────
    def add_request(self, rid: int, prompt, sampling_params: SamplingParams) -> None:
        self.tok_worker.submit(rid, prompt, sampling_params)

    def abort(self, rid: int) -> None:
        self.client.abort(rid)

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[dict]:
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        assert len(prompts) == len(sampling_params)

        # 1) Submit all (tokenization runs in threads, pipelined with engine).
        for rid, (p, sp) in enumerate(zip(prompts, sampling_params)):
            self.add_request(rid, p, sp)

        # 2) Drain DONE/STAT messages.
        pbar = tqdm(
            total=len(prompts),
            desc="Generating",
            dynamic_ncols=True,
            disable=not use_tqdm,
        )
        prefill_tps = decode_tps = 0.0
        pending = len(prompts)

        while pending > 0:
            msg = self.client.recv_output()
            if msg is None:
                continue
            if msg.op == MsgType.DONE:
                pending -= 1
                self.detok_worker.submit(msg.rid, msg.token_ids, pbar.update)
            elif msg.op == MsgType.STAT:
                prefill_tps = msg.prefill_tps or prefill_tps
                decode_tps = msg.decode_tps or decode_tps
                pbar.set_postfix(
                    {
                        "Prefill": f"{int(prefill_tps)}tok/s",
                        "Decode": f"{int(decode_tps)}tok/s",
                    }
                )

        self.detok_worker.flush()
        pbar.close()
        return [self.detok_worker.results[r] for r in sorted(self.detok_worker.results)]

    # ───────────────────────────── lifecycle ─────────────────────────────
    def exit(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.tok_worker.flush()
        finally:
            self.client.shutdown()
