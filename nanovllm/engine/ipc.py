"""ZeroMQ wire protocol + transport classes shared by the frontend and the
engine subprocess.

Two PUSH/PULL ipc:// sockets:
    - REQ_ADDR: Frontend(PUSH) -> Engine(PULL)   ADD / ABORT / EXIT
    - OUT_ADDR: Engine(PUSH)  -> Frontend(PULL)  DONE / STAT / BYE
"""

from __future__ import annotations

import os
import pickle
import tempfile
from dataclasses import dataclass
from enum import Enum
from typing import Any

import zmq


# ─────────────────────────── wire-protocol messages ──────────────────────────


class MsgType(str, Enum):
    ADD = "ADD"
    ABORT = "ABORT"
    EXIT = "EXIT"
    DONE = "DONE"
    STAT = "STAT"
    BYE = "BYE"
    READY = "READY"


@dataclass
class AddRequest:
    rid: int
    token_ids: list[int]
    sp: dict
    op: str = MsgType.ADD


@dataclass
class AbortRequest:
    rid: int
    op: str = MsgType.ABORT


@dataclass
class ExitRequest:
    op: str = MsgType.EXIT


@dataclass
class DoneOutput:
    rid: int
    token_ids: list[int]
    op: str = MsgType.DONE


@dataclass
class StatOutput:
    prefill_tps: float
    decode_tps: float
    op: str = MsgType.STAT


@dataclass
class ByeOutput:
    op: str = MsgType.BYE


@dataclass
class ReadyOutput:
    op: str = MsgType.READY


# ───────────────────────────── transport classes ─────────────────────────────


class ZmqChannel:
    """Thin wrapper around a single ZMQ socket — typed send/recv with pickle."""

    DEFAULT_HWM = 4096

    def __init__(self, addr: str, kind: int, bind: bool, hwm: int = DEFAULT_HWM):
        self.addr = addr
        self.kind = kind
        self.bind = bind
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(kind)
        self.sock.set_hwm(hwm)
        if bind:
            self.sock.bind(addr)
        else:
            self.sock.connect(addr)
        self._closed = False

    def send(self, msg: Any) -> None:
        self.sock.send(pickle.dumps(msg, protocol=5), copy=False)

    def recv(self, timeout_ms: int | None = None) -> Any | None:
        if timeout_ms is not None:
            if not self.sock.poll(timeout_ms):
                return None
        return pickle.loads(self.sock.recv())

    def try_recv(self) -> Any | None:
        try:
            return pickle.loads(self.sock.recv(flags=zmq.NOBLOCK))
        except zmq.Again:
            return None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.sock.close(linger=0)
        except Exception:
            pass
        # Best-effort cleanup of ipc socket file (only the binder owns it).
        if self.bind and self.addr.startswith("ipc://"):
            path = self.addr[len("ipc://") :]
            try:
                os.unlink(path)
            except OSError:
                pass


class IPCEndpoints:
    """Container for the two ipc:// addresses used by a single engine instance.

    Per-pid filenames so multiple engines on one host don't collide.
    """

    def __init__(self, req_addr: str, out_addr: str):
        self.req_addr = req_addr
        self.out_addr = out_addr

    @classmethod
    def fresh(cls, tag: str | None = None) -> "IPCEndpoints":
        tag = tag or f"{os.getpid()}"
        tmpdir = tempfile.gettempdir()
        return cls(
            req_addr=f"ipc://{tmpdir}/nanovllm-req-{tag}.sock",
            out_addr=f"ipc://{tmpdir}/nanovllm-out-{tag}.sock",
        )
