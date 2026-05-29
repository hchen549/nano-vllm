"""Frontend-side proxy for the EngineCore running in another process.

LLMEngineMP -> EngineCoreClient -> ZMQ -> EngineCore

This is the *only* place on the frontend side that knows about ZMQ.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Optional

import torch.multiprocessing as mp
import zmq
from nanovllm.config import Config
from nanovllm.engine.engine_core import engine_main
from nanovllm.engine.ipc import (
    AbortRequest,
    AddRequest,
    ExitRequest,
    IPCEndpoints,
    MsgType,
    ZmqChannel,
)
from nanovllm.sampling_params import SamplingParams


class EngineCoreClient:
    """Owns the engine subprocess and the two ZMQ sockets used to talk to it.

    Public API mirrors the wire protocol:
        client.add_request(rid, token_ids, sp)   # ADD
        client.abort(rid)                        # ABORT
        client.recv_output(timeout_ms=...)       # DONE / STAT / BYE
        client.shutdown()                        # EXIT + join
    """

    SHUTDOWN_TIMEOUT_S = 30.0
    READY_TIMEOUT_S = 600.0  # generous window for first-time model load + warmup

    # ──────────────────────────── construction ───────────────────────────
    def __init__(
        self,
        config: Config,
        endpoints: Optional[IPCEndpoints] = None,
        spawn_engine: bool = True,
    ):
        self.config = config
        self.endpoints = endpoints or IPCEndpoints.fresh()

        # 1) Optionally spawn the engine subprocess.
        self._engine_proc: Optional[mp.Process] = None
        if spawn_engine:
            ctx = mp.get_context("spawn")
            self._engine_proc = ctx.Process(
                target=engine_main,
                args=(config, self.endpoints),
                daemon=False,
            )
            self._engine_proc.start()

        # 2) Connect to the sockets the engine binds. ZMQ tolerates the
        #    bind-after-connect race for ipc:// — messages buffer locally.
        self._req_chan = ZmqChannel(self.endpoints.req_addr, zmq.PUSH, bind=False)
        self._out_chan = ZmqChannel(self.endpoints.out_addr, zmq.PULL, bind=False)

        self._closed = False

        # 3) Block until the engine signals it has finished loading the model
        #    and is ready to accept ADD requests. Keeps generate() timing fair.
        ready = self._out_chan.recv(timeout_ms=int(self.READY_TIMEOUT_S * 1000))
        if ready is None:
            raise RuntimeError(
                f"Engine did not signal READY within {self.READY_TIMEOUT_S}s"
            )
        if ready.op != MsgType.READY:
            raise RuntimeError(
                f"Expected READY as first engine message, got {ready.op}"
            )

    # ─────────────────────────────── send ──────────────────────────────
    def add_request(self, rid: int, token_ids: list[int], sp: SamplingParams) -> None:
        """Submit a tokenized prompt. Blocks if PUSH socket has hit HWM."""
        self._req_chan.send(
            AddRequest(rid=rid, token_ids=list(token_ids), sp=asdict(sp))
        )

    def abort(self, rid: int) -> None:
        self._req_chan.send(AbortRequest(rid=rid))

    # ─────────────────────────────── recv ──────────────────────────────
    def recv_output(self, timeout_ms: Optional[int] = None):
        """Return the next message from the engine, or None if a timeout was
        provided and elapsed without a message."""
        return self._out_chan.recv(timeout_ms=timeout_ms)

    def try_recv_output(self):
        """Non-blocking variant — returns None if the inbox is empty."""
        return self._out_chan.try_recv()

    # ────────────────────────────── lifecycle ─────────────────────────────
    def is_alive(self) -> bool:
        return self._engine_proc is None or self._engine_proc.is_alive()

    def shutdown(self) -> None:
        """Send EXIT, wait for BYE (or timeout), then join the engine.
        Idempotent."""
        if self._closed:
            return
        self._closed = True

        try:
            self._req_chan.send(ExitRequest())
        except Exception:
            pass  # engine may already be gone

        # Best-effort drain until BYE or timeout.
        deadline_ms = int(self.SHUTDOWN_TIMEOUT_S * 1000)
        while deadline_ms > 0:
            slice_ms = min(deadline_ms, 500)
            msg = self._out_chan.recv(timeout_ms=slice_ms)
            deadline_ms -= slice_ms
            if msg is None:
                continue
            if msg.op == MsgType.BYE:
                break

        if self._engine_proc is not None:
            self._engine_proc.join(timeout=self.SHUTDOWN_TIMEOUT_S)
            if self._engine_proc.is_alive():
                self._engine_proc.terminate()
                self._engine_proc.join(timeout=5)
            self._engine_proc = None

        self._req_chan.close()
        self._out_chan.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.shutdown()

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass
