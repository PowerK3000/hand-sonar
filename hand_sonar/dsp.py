"""Spectral helpers shared by the hardware test, visualizer and detector."""

from __future__ import annotations

import numpy as np


def power_spectrum(x: np.ndarray, samplerate: int) -> tuple[np.ndarray, np.ndarray]:
    """Hann-windowed power spectrum. Returns (freqs, power) with power in linear units."""
    if x.ndim != 1:
        raise ValueError("expected a mono signal")
    win = np.hanning(len(x))
    spec = np.fft.rfft(x * win)
    power = (np.abs(spec) ** 2) / np.sum(win ** 2)
    return np.fft.rfftfreq(len(x), 1 / samplerate), power


def band_power(freqs: np.ndarray, power: np.ndarray, lo: float, hi: float) -> float:
    mask = (freqs >= lo) & (freqs <= hi)
    if not mask.any():
        raise ValueError(f"no bins in [{lo}, {hi}] Hz")
    return float(power[mask].sum())


def db(x: float, floor: float = 1e-20) -> float:
    return 10 * np.log10(max(x, floor))


def doppler_hz(speed_mps: float, carrier_hz: float, c: float = 343.0) -> float:
    """Two-way Doppler shift of an echo from a reflector moving at `speed_mps` toward the mic."""
    return 2 * speed_mps * carrier_hz / c
