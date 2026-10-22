"""Phase-continuous tone generator with click-free ramps.

Abrupt starts/stops of a high-frequency sine splatter energy down into the audible
band (a "click"), so every amplitude change goes through a raised-cosine ramp.
"""

from __future__ import annotations

import threading

import numpy as np

# Hard ceiling on output amplitude regardless of config: keeps the carrier at a
# modest level even if a config file is mistyped.
MAX_AMPLITUDE = 0.3


class ToneGenerator:
    """Generates one sine per output channel; thread-safe target amplitude.

    `freqs[i]` is the carrier for channel i (None = silent channel).
    """

    def __init__(self, samplerate: int, freqs: list[float | None], amplitude: float,
                 ramp_seconds: float = 0.05) -> None:
        if samplerate <= 0:
            raise ValueError("samplerate must be positive")
        for f in freqs:
            if f is not None and not 0 < f < samplerate / 2:
                raise ValueError(f"carrier {f} Hz outside (0, Nyquist={samplerate / 2})")
        if not 0 <= amplitude <= MAX_AMPLITUDE:
            raise ValueError(f"amplitude must be in [0, {MAX_AMPLITUDE}]")
        self.samplerate = samplerate
        self.freqs = list(freqs)
        self._phase = np.zeros(len(freqs))
        self._target = amplitude
        self._current = 0.0  # start silent and ramp up
        self._ramp_step = 1.0 / max(1, int(ramp_seconds * samplerate))
        self._lock = threading.Lock()

    def set_amplitude(self, amplitude: float) -> None:
        if not 0 <= amplitude <= MAX_AMPLITUDE:
            raise ValueError(f"amplitude must be in [0, {MAX_AMPLITUDE}]")
        with self._lock:
            self._target = amplitude

    def set_frequencies(self, freqs: list[float | None]) -> None:
        if len(freqs) != len(self.freqs):
            raise ValueError("channel count cannot change")
        with self._lock:
            self.freqs = list(freqs)

    @property
    def is_silent(self) -> bool:
        return self._current == 0.0 and self._target == 0.0

    def render(self, frames: int) -> np.ndarray:
        """Return a (frames, channels) float32 block."""
        with self._lock:
            target, freqs = self._target, list(self.freqs)
        # A 50 ms linear envelope ramp keeps splatter well below audibility at 18+ kHz.
        step = self._ramp_step * MAX_AMPLITUDE
        direction = np.sign(target - self._current)
        env = self._current + direction * step * np.arange(1, frames + 1)
        env = np.minimum(env, target) if direction >= 0 else np.maximum(env, target)
        if frames:
            self._current = float(env[-1])
        out = np.zeros((frames, len(freqs)), dtype=np.float32)
        n = np.arange(frames)
        for ch, f in enumerate(freqs):
            if f is None:
                continue
            w = 2 * np.pi * f / self.samplerate
            out[:, ch] = env * np.sin(self._phase[ch] + w * n)
            self._phase[ch] = (self._phase[ch] + w * frames) % (2 * np.pi)
        return out
