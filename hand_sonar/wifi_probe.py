"""Feasibility probe for Wi-Fi RSSI hand sensing.

Polls the connected Wi-Fi interface's RSSI through the native WLAN API (wlanapi.dll,
no extra software) as fast as possible and reports how often the driver actually
refreshes the value, and how far it moves. Run once with your hand away, then while
covering / uncovering the adapter:

    python -m hand_sonar.wifi_probe --seconds 20
"""

from __future__ import annotations

import argparse
import ctypes
import sys
import time
from ctypes import wintypes

import numpy as np

WLAN_INTF_OPCODE_RSSI = 0x10000102
ERROR_SUCCESS = 0


class GUID(ctypes.Structure):
    _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD),
                ("Data3", wintypes.WORD), ("Data4", ctypes.c_ubyte * 8)]


class WLAN_INTERFACE_INFO(ctypes.Structure):
    _fields_ = [("InterfaceGuid", GUID), ("strInterfaceDescription", ctypes.c_wchar * 256),
                ("isState", ctypes.c_int)]


class WLAN_INTERFACE_INFO_LIST(ctypes.Structure):
    _fields_ = [("dwNumberOfItems", wintypes.DWORD), ("dwIndex", wintypes.DWORD),
                ("InterfaceInfo", WLAN_INTERFACE_INFO * 1)]


WLAN_STATE_CONNECTED = 1


class WlanError(RuntimeError):
    pass


class WlanClient:
    def __init__(self) -> None:
        try:
            self._api = ctypes.WinDLL("wlanapi.dll")
        except OSError as e:
            raise WlanError("wlanapi.dll not available (WLAN AutoConfig service missing?)") from e
        self._handle = wintypes.HANDLE()
        negotiated = wintypes.DWORD()
        rc = self._api.WlanOpenHandle(2, None, ctypes.byref(negotiated), ctypes.byref(self._handle))
        if rc != ERROR_SUCCESS:
            raise WlanError(f"WlanOpenHandle failed: {rc}")

    def interfaces(self) -> list[tuple[GUID, str, int]]:
        plist = ctypes.POINTER(WLAN_INTERFACE_INFO_LIST)()
        rc = self._api.WlanEnumInterfaces(self._handle, None, ctypes.byref(plist))
        if rc != ERROR_SUCCESS:
            raise WlanError(f"WlanEnumInterfaces failed: {rc}")
        try:
            n = plist.contents.dwNumberOfItems
            arr = ctypes.cast(ctypes.byref(plist.contents.InterfaceInfo),
                              ctypes.POINTER(WLAN_INTERFACE_INFO * n)).contents
            # Copy GUIDs out before the list memory is freed.
            return [(GUID.from_buffer_copy(i.InterfaceGuid), i.strInterfaceDescription, i.isState) for i in arr]
        finally:
            self._api.WlanFreeMemory(plist)

    def rssi(self, guid: GUID) -> int:
        size = wintypes.DWORD()
        data = ctypes.c_void_p()
        rc = self._api.WlanQueryInterface(self._handle, ctypes.byref(guid), WLAN_INTF_OPCODE_RSSI,
                                          None, ctypes.byref(size), ctypes.byref(data), None)
        if rc != ERROR_SUCCESS:
            raise WlanError(f"WlanQueryInterface(rssi) failed: {rc} (is the adapter connected?)")
        try:
            return ctypes.cast(data, ctypes.POINTER(ctypes.c_long)).contents.value
        finally:
            self._api.WlanFreeMemory(data)

    def close(self) -> None:
        if self._handle:
            self._api.WlanCloseHandle(self._handle, None)
            self._handle = wintypes.HANDLE()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--poll-ms", type=float, default=5.0)
    args = p.parse_args(argv)
    try:
        client = WlanClient()
    except WlanError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    try:
        ifaces = client.interfaces()
        if not ifaces:
            print("error: no Wi-Fi adapter found. Plug in the dongle first.", file=sys.stderr)
            return 2
        connected = [i for i in ifaces if i[2] == WLAN_STATE_CONNECTED]
        for _, desc, state in ifaces:
            print(f"adapter: {desc} (state {state}{', connected' if state == WLAN_STATE_CONNECTED else ''})")
        if not connected:
            print("error: adapter not connected to a network; RSSI is only reported while connected.",
                  file=sys.stderr)
            return 2
        guid = connected[0][0]
        samples: list[tuple[float, int]] = []
        t0 = time.perf_counter()
        last_print = t0
        while (now := time.perf_counter()) - t0 < args.seconds:
            v = client.rssi(guid)
            samples.append((now - t0, v))
            if now - last_print >= 0.5:
                last_print = now
                print(f"{now - t0:5.1f}s rssi {v:4d} dBm", flush=True)
            time.sleep(args.poll_ms / 1000)
    except WlanError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    finally:
        client.close()

    t = np.array([s[0] for s in samples])
    r = np.array([s[1] for s in samples])
    changes = np.flatnonzero(np.diff(r)) + 1
    print(f"\n{len(samples)} polls, {len(changes)} value changes in {t[-1]:.1f} s")
    if len(changes) >= 2:
        gaps = np.diff(t[changes])
        print(f"refresh interval: median {np.median(gaps) * 1000:.0f} ms, min {gaps.min() * 1000:.0f} ms")
    print(f"rssi range {r.min()}..{r.max()} dBm, std {r.std():.2f} dB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
