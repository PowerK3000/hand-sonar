"""Step 1 hardware test: which near-ultrasonic frequencies survive speaker -> room -> mic?

Plays a stepped tone sweep while recording, then reports, per frequency, the received
level and SNR against a silent baseline recorded in the same run.

    python -m hand_sonar.hwtest --out "Speakers (YU8)" --in "Microphone (YU8)"
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd

from .devices import DeviceError, find_device
from .dsp import band_power, db, power_spectrum
from .tone import MAX_AMPLITUDE

SAMPLERATE = 48000
SILENCE_S = 1.2
TONE_S = 0.8
GAP_S = 0.1
FADE_S = 0.03
# Only the steady middle of each tone is analysed, which absorbs device latency.
ANALYSIS_OFFSET_S = 0.3
ANALYSIS_LEN_S = 0.4
BAND_HALF_WIDTH_HZ = 15.0
USABLE_SNR_DB = 20.0


def build_sweep(freqs: list[float], amplitude: float) -> tuple[np.ndarray, list[int]]:
    """Mono sweep signal plus the sample index where each tone starts."""
    fade = int(FADE_S * SAMPLERATE)
    n_tone = int(TONE_S * SAMPLERATE)
    env = np.ones(n_tone)
    ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, fade)))
    env[:fade], env[-fade:] = ramp, ramp[::-1]
    t = np.arange(n_tone) / SAMPLERATE
    parts = [np.zeros(int(SILENCE_S * SAMPLERATE))]
    starts = []
    pos = len(parts[0])
    for f in freqs:
        starts.append(pos)
        parts.append(amplitude * env * np.sin(2 * np.pi * f * t))
        gap = np.zeros(int(GAP_S * SAMPLERATE))
        parts.append(gap)
        pos += n_tone + len(gap)
    parts.append(np.zeros(int(0.5 * SAMPLERATE)))  # tail so the last tone is fully captured
    return np.concatenate(parts).astype(np.float32), starts


def play_and_record(signal: np.ndarray, out_index: int, in_index: int,
                    out_channels: int, channel: int | None = None) -> np.ndarray:
    """Play `signal` on one output channel (None = all) while recording mono from the input."""
    recorded: list[np.ndarray] = []
    errors: list[str] = []
    pos = 0
    done = threading.Event()

    def in_cb(indata, frames, t, status):
        if status:
            errors.append(f"input: {status}")
        recorded.append(indata[:, 0].copy())

    def out_cb(outdata, frames, t, status):
        nonlocal pos
        if status:
            errors.append(f"output: {status}")
        chunk = signal[pos:pos + frames]
        outdata[:] = 0
        if channel is None:
            outdata[:len(chunk)] = chunk[:, None]
        else:
            outdata[:len(chunk), channel] = chunk
        pos += frames
        if pos >= len(signal):
            done.set()
            raise sd.CallbackStop

    with sd.InputStream(device=in_index, samplerate=SAMPLERATE, channels=1,
                        dtype="float32", callback=in_cb):
        time.sleep(0.2)  # let the input settle before the first sample plays
        with sd.OutputStream(device=out_index, samplerate=SAMPLERATE, channels=out_channels,
                             dtype="float32", callback=out_cb):
            if not done.wait(timeout=len(signal) / SAMPLERATE + 5):
                raise RuntimeError("playback did not finish in time")
        time.sleep(0.3)
    if errors:
        print(f"warning: stream status flags: {sorted(set(errors))}", file=sys.stderr)
    if not recorded:
        raise RuntimeError("no audio captured from input device")
    return np.concatenate(recorded)


def analyse(rec: np.ndarray, starts: list[int], freqs: list[float]) -> list[dict]:
    # The input stream started 0.2 s before playback, so the recording leads by that much.
    lead = int(0.2 * SAMPLERATE)
    n = int(ANALYSIS_LEN_S * SAMPLERATE)
    base = rec[lead + int(0.3 * SAMPLERATE): lead + int(0.3 * SAMPLERATE) + n]
    bf, bp = power_spectrum(base, SAMPLERATE)
    results = []
    for f, s in zip(freqs, starts):
        a = lead + s + int(ANALYSIS_OFFSET_S * SAMPLERATE)
        seg = rec[a:a + n]
        if len(seg) < n:
            raise RuntimeError("recording shorter than expected; device dropped audio")
        sf, sp = power_spectrum(seg, SAMPLERATE)
        sig = band_power(sf, sp, f - BAND_HALF_WIDTH_HZ, f + BAND_HALF_WIDTH_HZ)
        noise = band_power(bf, bp, f - BAND_HALF_WIDTH_HZ, f + BAND_HALF_WIDTH_HZ)
        # Local floor in the same tone segment, excluding the tone: catches leakage/
        # distortion that a quiet baseline would hide.
        local = (band_power(sf, sp, f - 400, f - 100) + band_power(sf, sp, f + 100, f + 400)) \
            * (2 * BAND_HALF_WIDTH_HZ) / 600
        results.append({
            "freq_hz": f,
            "level_dbfs": round(db(sig) - db(n / 4), 1),  # peak amplitude re full scale
            "snr_db": round(db(sig) - db(noise), 1),
            "snr_local_db": round(db(sig) - db(local), 1),
        })
    return results


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", dest="out_name", help="output device name substring")
    p.add_argument("--in", dest="in_name", help="input device name substring")
    p.add_argument("--amp", type=float, default=0.1, help=f"tone amplitude 0..{MAX_AMPLITUDE}")
    p.add_argument("--channel", choices=["both", "left", "right"], default="both",
                   help="play on one speaker only (e.g. a single headphone cup)")
    p.add_argument("--lo", type=float, default=16000)
    p.add_argument("--hi", type=float, default=22000)
    p.add_argument("--step", type=float, default=500)
    p.add_argument("--save", type=Path, default=Path("data/hwtest"))
    args = p.parse_args(argv)
    if not 0 < args.amp <= MAX_AMPLITUDE:
        p.error(f"--amp must be in (0, {MAX_AMPLITUDE}]")
    try:
        out_dev = find_device("output", args.out_name)
        in_dev = find_device("input", args.in_name)
    except DeviceError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    freqs = list(np.arange(args.lo, args.hi + 1, args.step, dtype=float))
    signal, starts = build_sweep(freqs, args.amp)
    print(f"out: {out_dev.name}\nin:  {in_dev.name}\nsweeping {freqs[0]:.0f}-{freqs[-1]:.0f} Hz "
          f"at amplitude {args.amp} ({len(signal) / SAMPLERATE:.1f} s)...")
    try:
        ch = {"both": None, "left": 0, "right": 1}[args.channel]
        if ch is not None and ch >= out_dev.channels:
            print(f"error: {out_dev.name} has only {out_dev.channels} channel(s)", file=sys.stderr)
            return 2
        rec = play_and_record(signal, out_dev.index, in_dev.index, out_dev.channels, ch)
    except (sd.PortAudioError, RuntimeError) as e:
        print(f"error: audio I/O failed: {e}", file=sys.stderr)
        return 1
    results = analyse(rec, starts, freqs)
    print(f"{'freq':>7} {'level dBFS':>11} {'SNR vs quiet':>13} {'SNR local':>10}")
    for r in results:
        mark = "  usable" if r["snr_db"] >= USABLE_SNR_DB and r["snr_local_db"] >= USABLE_SNR_DB else ""
        print(f"{r['freq_hz']:7.0f} {r['level_dbfs']:11.1f} {r['snr_db']:13.1f} {r['snr_local_db']:10.1f}{mark}")
    args.save.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = args.save / f"hwtest-{stamp}.json"
    out_path.write_text(json.dumps({"out": out_dev.name, "in": in_dev.name, "amp": args.amp,
                                    "results": results}, indent=2))
    print(f"saved {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
