"""Mic diagnostic: logs, once per second, our captured peak next to Windows' own
endpoint peak meter (the same meter the Sound control panel draws).

If Windows' meter moves while ours stays flat, the problem is in our capture path;
if both are flat, the device is delivering silence.

    python -m hand_sonar.miccheck --in YU8 --seconds 60
"""

from __future__ import annotations

import argparse
import sys
import time
from ctypes import POINTER, cast

import numpy as np
import sounddevice as sd

from .devices import DeviceError, find_device


def _windows_meter(name: str):
    """IAudioMeterInformation for the active endpoint named `name`, or None if unavailable."""
    try:
        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import AudioUtilities, IAudioMeterInformation
    except ImportError:
        return None
    for d in AudioUtilities.GetAllDevices():
        if d.FriendlyName == name and str(d.state).endswith("Active"):
            iface = d._dev.Activate(IAudioMeterInformation._iid_, CLSCTX_ALL, None)
            return cast(iface, POINTER(IAudioMeterInformation))
    return None


def _db(x: float) -> float:
    return 20 * np.log10(x + 1e-12)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="in_name", help="input device name substring")
    p.add_argument("--seconds", type=int, default=30)
    args = p.parse_args(argv)
    try:
        dev = find_device("input", args.in_name)
    except DeviceError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    meter = _windows_meter(dev.name)
    chunks: list[np.ndarray] = []
    stream = sd.InputStream(device=dev.index, samplerate=dev.samplerate, channels=min(2, dev.channels),
                            dtype="float32", callback=lambda i, f, t, s: chunks.append(i.copy()))
    print(f"device: {dev.name} ({dev.samplerate} Hz)\n sec   ours peak   windows peak", flush=True)
    with stream:
        for sec in range(args.seconds):
            win_peak = 0.0
            for _ in range(10):
                time.sleep(0.1)
                if meter is not None:
                    win_peak = max(win_peak, meter.GetPeakValue())
            got, chunks[:] = list(chunks), []
            ours = float(np.abs(np.concatenate(got)).max()) if got else 0.0
            win = f"{_db(win_peak):8.1f}" if meter is not None else "     n/a"
            print(f"{sec:4d} {_db(ours):10.1f} dB {win} dB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
