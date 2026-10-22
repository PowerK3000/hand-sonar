"""Full-duplex sonar audio: carrier out, mic in, with fail-safe tone shutdown.

Safety model for the carrier:
- `close()` aborts the output stream (drops buffered audio instead of draining it).
- A heartbeat watchdog in the output callback: if the owning loop stops calling
  `heartbeat()` (hang, deadlock, debugger pause) the callback emits silence.
- `atexit` closes any open instance; if the process dies outright, Windows tears
  down the WASAPI stream with it, so the tone cannot outlive the process.
"""

from __future__ import annotations

import atexit
import queue
import threading
import time
import weakref

import numpy as np
import sounddevice as sd

from .devices import Device
from .tone import ToneGenerator

HEARTBEAT_TIMEOUT_S = 2.0
_open_instances: "weakref.WeakSet[SonarIO]" = weakref.WeakSet()


@atexit.register
def _close_all() -> None:
    for inst in list(_open_instances):
        inst.close()


class SonarIO:
    """Plays per-channel carriers on `out_dev` and streams mono mic blocks from `in_dev`.

    Pass `out_dev=None` (or all carriers None) for listen-only mode.
    Consume audio with `read()`, which returns all samples captured since the last call.
    """

    def __init__(self, in_dev: Device, out_dev: Device | None, samplerate: int,
                 carriers: list[float | None] | None, amplitude: float,
                 blocksize: int = 512, max_queue_blocks: int = 400) -> None:
        self.samplerate = samplerate
        self._q: queue.Queue[np.ndarray] = queue.Queue(maxsize=max_queue_blocks)
        self.dropped_blocks = 0
        self.status_flags: list[str] = []
        self._last_beat = time.monotonic()
        self._lock = threading.Lock()
        self._closed = False
        self.tone: ToneGenerator | None = None
        self._out: sd.OutputStream | None = None

        self._in = sd.InputStream(device=in_dev.index, samplerate=samplerate,
                                  channels=1, dtype="float32", blocksize=blocksize,
                                  callback=self._in_cb)
        if out_dev is not None and carriers and any(c is not None for c in carriers):
            chans = list(carriers) + [None] * (out_dev.channels - len(carriers))
            self.tone = ToneGenerator(samplerate, chans[:out_dev.channels], amplitude)
            self._out = sd.OutputStream(device=out_dev.index, samplerate=samplerate,
                                        channels=out_dev.channels, dtype="float32",
                                        blocksize=blocksize, callback=self._out_cb)

    # -- callbacks (PortAudio thread) ----------------------------------------
    def _in_cb(self, indata, frames, t, status) -> None:
        if status:
            self.status_flags.append(str(status))
        try:
            self._q.put_nowait(indata[:, 0].copy())
        except queue.Full:
            self.dropped_blocks += 1

    def _out_cb(self, outdata, frames, t, status) -> None:
        if status:
            self.status_flags.append(str(status))
        if self._closed or time.monotonic() - self._last_beat > HEARTBEAT_TIMEOUT_S:
            outdata.fill(0)
            return
        outdata[:] = self.tone.render(frames)

    # -- control -------------------------------------------------------------
    def start(self) -> "SonarIO":
        self.heartbeat()
        self._in.start()
        if self._out is not None:
            self._out.start()
        _open_instances.add(self)
        return self

    def heartbeat(self) -> None:
        self._last_beat = time.monotonic()

    def read(self) -> np.ndarray:
        """All mic samples captured since the last call (may be empty). Also beats the watchdog."""
        self.heartbeat()
        blocks = []
        while True:
            try:
                blocks.append(self._q.get_nowait())
            except queue.Empty:
                break
        return np.concatenate(blocks) if blocks else np.zeros(0, dtype=np.float32)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        # Abort, not stop: stop() would drain queued carrier audio before going quiet.
        for s in (self._out, self._in):
            if s is None:
                continue
            try:
                s.abort()
                s.close()
            except sd.PortAudioError:
                pass  # stream already dead (device unplugged); nothing left to silence
        _open_instances.discard(self)

    def __enter__(self) -> "SonarIO":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.close()
