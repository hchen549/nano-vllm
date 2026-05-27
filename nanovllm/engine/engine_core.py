"""Engine-side core: scheduler + model runner + ZMQ-driven step loop.

Runs in a dedicated subprocess spawned by the frontend (see
EngineCoreClient). Communicates over two ZMQ sockets defined in ipc.py.
"""

from __future__ import annotations

from time import perf_counter

import torch.multiprocessing as mp
import zmq
from nanovllm.config import Config
from nanovllm.engine.ipc import (
    ByeOutput,
    DoneOutput,
    IPCEndpoints,
    MsgType,
    StatOutput,
    ZmqChannel,
)
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams


class EngineCore:
    """Owns the model (ModelRunner + tensor-parallel workers), the Scheduler,
    and the ZMQ-driven step loop.

    Lifecycle:
        core = EngineCore(config, endpoints)
        core.serve_forever()       # blocks until ExitRequest received
        # serve_forever's finally clause calls shutdown()
    """

    IDLE_BLOCK_MS = 100  # how long to block on the inbox when scheduler is empty

    # ─────────────────────────────── init ───────────────────────────────
    def __init__(self, config: Config, endpoints: IPCEndpoints):
        self.config = config
        self.endpoints = endpoints

        # 1) Tensor-parallel worker fan-out (mirrors LLMEngine.__init__).
        self._tp_procs: list[mp.Process] = []
        self._tp_events: list = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            ev = ctx.Event()
            p = ctx.Process(target=ModelRunner, args=(config, i, ev))
            p.start()
            self._tp_procs.append(p)
            self._tp_events.append(ev)

        # 2) Rank-0 model runner + scheduler.
        self.runner = ModelRunner(config, 0, self._tp_events)
        Sequence.block_size = config.kvcache_block_size
        self.scheduler = Scheduler(config)

        # 3) IPC sockets — engine binds, frontend connects.
        self.req_chan = ZmqChannel(endpoints.req_addr, zmq.PULL, bind=True)
        self.out_chan = ZmqChannel(endpoints.out_addr, zmq.PUSH, bind=True)

        # 4) Bookkeeping for cross-process correlation.
        self._rid_by_seq_id: dict[int, int] = {}
        self._seq_by_rid: dict[int, Sequence] = {}
        self._exiting: bool = False
        self._shut_down: bool = False

    # ────────────────────────────── inbox ──────────────────────────────
    def _handle(self, msg) -> None:
        op = msg.op
        if op == MsgType.ADD:
            sp = SamplingParams(**msg.sp)
            seq = Sequence(msg.token_ids, sp)
            self.scheduler.add(seq)
            self._rid_by_seq_id[seq.seq_id] = msg.rid
            self._seq_by_rid[msg.rid] = seq
        elif op == MsgType.ABORT:
            self._abort(msg.rid)
        elif op == MsgType.EXIT:
            self._exiting = True

    def _abort(self, rid: int) -> None:
        seq = self._seq_by_rid.pop(rid, None)
        if seq is None:
            return
        self._rid_by_seq_id.pop(seq.seq_id, None)
        # Remove from whichever scheduler queue holds it; deallocate KV blocks.
        if seq in self.scheduler.waiting:
            self.scheduler.waiting.remove(seq)
        elif seq in self.scheduler.running:
            self.scheduler.running.remove(seq)
            self.scheduler.block_manager.deallocate(seq)
        seq.status = SequenceStatus.FINISHED

    def _drain_inbox(self, block: bool) -> None:
        if block:
            msg = self.req_chan.recv(timeout_ms=self.IDLE_BLOCK_MS)
            if msg is not None:
                self._handle(msg)
        while True:
            msg = self.req_chan.try_recv()
            if msg is None:
                return
            self._handle(msg)

    # ──────────────────────────── single step ────────────────────────────
    def _step_once(self) -> None:
        t0 = perf_counter()
        seqs, is_prefill = self.scheduler.schedule()
        print(
            f"[engine] step is_prefill={is_prefill} batch={len(seqs)} "
            f"tok={[s.num_scheduled_tokens for s in seqs]}",
            flush=True,
        )
        num_tokens = (
            sum(s.num_scheduled_tokens for s in seqs) if is_prefill else -len(seqs)
        )

        token_ids = self.runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        dt = perf_counter() - t0

        for s in seqs:
            if s.is_finished:
                rid = self._rid_by_seq_id.pop(s.seq_id, None)
                if rid is None:
                    continue  # was aborted
                self._seq_by_rid.pop(rid, None)
                self.out_chan.send(
                    DoneOutput(rid=rid, token_ids=s.completion_token_ids)
                )

        # if num_tokens > 0:
        #     self.out_chan.send(StatOutput(prefill_tps=num_tokens / dt, decode_tps=0.0))
        # elif num_tokens < 0:
        #     self.out_chan.send(StatOutput(prefill_tps=0.0, decode_tps=-num_tokens / dt))

    # ─────────────────────────────── loop ───────────────────────────────
    def serve_forever(self) -> None:
        try:
            while True:
                idle = self.scheduler.is_finished()
                self._drain_inbox(block=idle)
                if self._exiting and self.scheduler.is_finished():
                    break
                if not self.scheduler.is_finished():
                    self._step_once()
        finally:
            self.shutdown()

    # ───────────────────────────── teardown ─────────────────────────────
    def shutdown(self) -> None:
        if self._shut_down:
            return
        self._shut_down = True
        if getattr(self, "runner", None) is not None:
            try:
                self.runner.call("exit")
            except Exception:
                pass
            self.runner = None
            for p in self._tp_procs:
                p.join()
            self._tp_procs.clear()
        try:
            self.out_chan.send(ByeOutput())
        except Exception:
            pass
        self.req_chan.close()
        self.out_chan.close()


def engine_main(config: Config, endpoints: IPCEndpoints) -> None:
    """Entry-point for the engine subprocess."""
    EngineCore(config, endpoints).serve_forever()
