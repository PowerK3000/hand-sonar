"""Step 2: live mic level meter + scrolling spectrogram around the carrier.

    python -m hand_sonar.visualizer --in YU8 --out Sceptre --carrier 19000
    python -m hand_sonar.visualizer --in YU8 --carrier 0      # listen-only, full band

Keys: F toggles full-band / carrier zoom, Esc quits. Without a carrier it starts in
full-band mode, which doubles as a "is this mic alive?" check.
"""

from __future__ import annotations

import argparse
import sys
import tkinter as tk

import numpy as np
import sounddevice as sd

from .audio_io import SonarIO
from .devices import DeviceError, find_device
from .tone import MAX_AMPLITUDE

SAMPLERATE = 48000
NFFT = 4096          # 11.7 Hz bins: a 1 m/s hand at 19 kHz shifts ~110 Hz
HOP = 1024           # ~21 ms per column
WIDTH, HEIGHT = 640, 320
DYN_RANGE_DB = 70.0
FRAME_MS = 40


def _colormap() -> np.ndarray:
    """256-entry black -> purple -> orange -> yellow ramp (perceptually ordered)."""
    stops = np.array([[0, 0, 0], [60, 10, 110], [190, 40, 90], [250, 140, 20], [255, 250, 160]], float)
    x = np.linspace(0, len(stops) - 1, 256)
    i = np.minimum(x.astype(int), len(stops) - 2)
    frac = (x - i)[:, None]
    return (stops[i] * (1 - frac) + stops[i + 1] * frac).astype(np.uint8)


class SpectrogramView:
    def __init__(self, root: tk.Tk, io: SonarIO, carrier: float, span: float) -> None:
        self.root, self.io, self.carrier, self.span = root, io, carrier, span
        self.full = carrier <= 0
        self.cmap = _colormap()
        self.window = np.hanning(NFFT).astype(np.float32)
        self.freqs = np.fft.rfftfreq(NFFT, 1 / SAMPLERATE)
        self.buf = np.zeros(0, dtype=np.float32)
        self.img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        self.ref_db = -40.0

        root.title("hand-sonar visualizer")
        root.configure(bg="#111")
        self.label = tk.Label(root, font=("Consolas", 11), fg="#ddd", bg="#111", anchor="w", justify="left")
        self.label.pack(fill="x", padx=8, pady=(8, 2))
        self.meter = tk.Canvas(root, width=WIDTH, height=14, bg="#222", highlightthickness=0)
        self.meter.pack(padx=8)
        self.canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg="black", highlightthickness=0)
        self.canvas.pack(padx=8, pady=8)
        self.photo = None
        self.image_item = self.canvas.create_image(0, 0, anchor="nw")
        self.axis = tk.Label(root, font=("Consolas", 9), fg="#999", bg="#111")
        self.axis.pack(pady=(0, 8))
        root.bind("<Escape>", lambda e: root.destroy())
        root.bind("<f>", lambda e: self.toggle())
        root.bind("<F>", lambda e: self.toggle())
        self._update_axis()

    def toggle(self) -> None:
        if self.carrier > 0:
            self.full = not self.full
            self.img[:] = 0
            self._update_axis()

    def _band(self) -> tuple[float, float]:
        if self.full:
            return 0.0, SAMPLERATE / 2
        return self.carrier - self.span, self.carrier + self.span

    def _update_axis(self) -> None:
        lo, hi = self._band()
        mode = "full band" if self.full else f"carrier {self.carrier:.0f} Hz ± {self.span:.0f}"
        self.axis.config(text=f"{mode}   bottom {lo:.0f} Hz  top {hi:.0f} Hz   (F: toggle, Esc: quit)")

    def tick(self) -> None:
        try:
            new = self.io.read()
        except Exception as e:  # surface audio failures in the window, don't die silently
            self.label.config(text=f"audio error: {e}", fg="#f66")
            self.root.after(500, self.tick)
            return
        self.buf = np.concatenate([self.buf, new])[-(NFFT + HOP * WIDTH):]
        cols = []
        while len(self.buf) >= NFFT + HOP:
            frame = self.buf[:NFFT]
            self.buf = self.buf[HOP:]
            spec = np.abs(np.fft.rfft(frame * self.window)) ** 2 / np.sum(self.window ** 2)
            cols.append(10 * np.log10(spec + 1e-20))
        if len(new):
            rms = float(np.sqrt(np.mean(new ** 2)))
            peak = float(np.abs(new).max())
            self._draw_meter(rms, peak)
        if cols:
            self._push_columns(cols)
        if self.io.status_flags or self.io.dropped_blocks:
            extra = f"   xruns {len(self.io.status_flags)} dropped {self.io.dropped_blocks}"
        else:
            extra = ""
        self.root.title(f"hand-sonar visualizer{extra}")
        self.root.after(FRAME_MS, self.tick)

    def _draw_meter(self, rms: float, peak: float) -> None:
        rms_db = 20 * np.log10(rms + 1e-12)
        peak_db = 20 * np.log10(peak + 1e-12)
        frac = np.clip((rms_db + 100) / 100, 0, 1)
        self.meter.delete("all")
        color = "#4c4" if rms_db > -70 else "#c84" if rms_db > -90 else "#c44"
        self.meter.create_rectangle(0, 0, frac * WIDTH, 14, fill=color, width=0)
        verdict = "mic live" if rms_db > -80 else "VERY QUIET: muted or gain at zero?"
        self.label.config(text=f"input rms {rms_db:6.1f} dBFS   peak {peak_db:6.1f} dBFS   {verdict}",
                          fg="#ddd" if rms_db > -80 else "#fa6")

    def _push_columns(self, cols: list[np.ndarray]) -> None:
        lo, hi = self._band()
        mask = (self.freqs >= lo) & (self.freqs <= hi)
        block = np.stack([c[mask] for c in cols], axis=1)            # (bins, n)
        # Track a slow reference so the colour scale follows the carrier level.
        self.ref_db = 0.9 * self.ref_db + 0.1 * float(block.max())
        norm = np.clip((block - (self.ref_db - DYN_RANGE_DB)) / DYN_RANGE_DB, 0, 1)
        rows = np.linspace(norm.shape[0] - 1, 0, HEIGHT).astype(int)  # high freq at top
        rgb = self.cmap[(norm[rows] * 255).astype(np.uint8)]           # (HEIGHT, n, 3)
        n = min(rgb.shape[1], WIDTH)
        self.img = np.concatenate([self.img[:, n:], rgb[:, -n:]], axis=1)
        ppm = b"P6 %d %d 255\n" % (WIDTH, HEIGHT) + self.img.tobytes()
        self.photo = tk.PhotoImage(data=ppm, format="PPM")
        self.canvas.itemconfig(self.image_item, image=self.photo)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="in_name", help="input device name substring")
    p.add_argument("--out", dest="out_name", help="output device name substring")
    p.add_argument("--carrier", type=float, default=0.0, help="carrier Hz (0 = listen only)")
    p.add_argument("--amp", type=float, default=0.1)
    p.add_argument("--span", type=float, default=400.0, help="zoom half-width in Hz")
    args = p.parse_args(argv)
    if not 0 <= args.amp <= MAX_AMPLITUDE:
        p.error(f"--amp must be in [0, {MAX_AMPLITUDE}]")
    try:
        in_dev = find_device("input", args.in_name)
        out_dev = find_device("output", args.out_name) if args.carrier > 0 else None
    except DeviceError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    try:
        io = SonarIO(in_dev, out_dev, SAMPLERATE,
                     [args.carrier] * out_dev.channels if out_dev else None, args.amp)
    except (sd.PortAudioError, ValueError) as e:
        print(f"error: cannot open audio: {e}", file=sys.stderr)
        return 1
    with io:
        root = tk.Tk()
        view = SpectrogramView(root, io, args.carrier, args.span)
        root.after(FRAME_MS, view.tick)
        root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
