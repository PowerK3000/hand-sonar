"""Feasibility probe for Bluetooth LE RSSI hand sensing.

Listens to advertisements from every nearby BLE device through the PC's Bluetooth
adapter and reports, per device: how often it is heard, and how its RSSI differs
between a "hand away" phase and a "hand covering the adapter" phase.

    python -m hand_sonar.ble_probe --seconds 30 --baseline 8

Protocol: keep your hand away for the first `--baseline` seconds, then alternate
covering / uncovering the dongle every ~2 s.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np

try:
    from bleak import BleakScanner
    from bleak.exc import BleakError
except ImportError:  # pragma: no cover - surfaced as a clear CLI error
    BleakScanner = None


async def _scan(seconds: float) -> tuple[dict[str, list[tuple[float, int]]], dict[str, str]]:
    samples: dict[str, list[tuple[float, int]]] = defaultdict(list)
    names: dict[str, str] = {}
    t0 = time.perf_counter()

    def on_adv(device, adv) -> None:
        samples[device.address].append((time.perf_counter() - t0, adv.rssi))
        if adv.local_name:
            names[device.address] = adv.local_name

    scanner = BleakScanner(detection_callback=on_adv, scanning_mode="active")
    await scanner.start()
    try:
        last = 0.0
        while (el := time.perf_counter() - t0) < seconds:
            if el - last >= 2:
                last = el
                total = sum(len(v) for v in samples.values())
                print(f"{el:5.1f}s  {len(samples)} devices, {total} adverts", flush=True)
            await asyncio.sleep(0.1)
    finally:
        await scanner.stop()
    return samples, names


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--baseline", type=float, default=8.0)
    p.add_argument("--save", type=Path, help="write raw samples (t, address, rssi) as CSV")
    args = p.parse_args(argv)
    if BleakScanner is None:
        print("error: bleak not installed (pip install bleak)", file=sys.stderr)
        return 2
    try:
        samples, names = asyncio.run(_scan(args.seconds))
    except (BleakError, OSError) as e:
        print(f"error: Bluetooth scan failed: {e}", file=sys.stderr)
        return 1
    if not samples:
        print("no BLE advertisements heard at all")
        return 0
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        rows = sorted((t, a, r) for a, s in samples.items() for t, r in s)
        args.save.write_text("t,address,rssi\n" + "".join(f"{t:.3f},{a},{r}\n" for t, a, r in rows))
        print(f"saved {args.save}")
    print(f"\n{'device':<20} {'name':<18} {'adv/s':>6} {'base mean':>9} {'base std':>8} "
          f"{'test min':>8} {'test std':>8} {'dip dB':>7}")
    rows = []
    for addr, s in samples.items():
        t = np.array([x[0] for x in s])
        r = np.array([x[1] for x in s], dtype=float)
        base, test = r[t < args.baseline], r[t >= args.baseline]
        rate = len(s) / args.seconds
        if len(base) < 3 or len(test) < 3:
            rows.append((rate, addr, names.get(addr, ""), None))
            continue
        dip = base.mean() - np.percentile(test, 10)
        rows.append((rate, addr, names.get(addr, ""), (base.mean(), base.std(), test.min(), test.std(), dip)))
    for rate, addr, name, st in sorted(rows, reverse=True)[:15]:
        if st is None:
            print(f"{addr:<20} {name[:18]:<18} {rate:6.1f}   (too few samples)")
        else:
            print(f"{addr:<20} {name[:18]:<18} {rate:6.1f} {st[0]:9.1f} {st[1]:8.1f} {st[2]:8.0f} {st[3]:8.1f} {st[4]:7.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
