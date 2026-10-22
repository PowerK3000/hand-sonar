"""Audio device discovery and selection.

Devices are matched by name substring on the WASAPI host API, because WASAPI gives
the lowest latency on Windows and exposes the device's native 48 kHz mix format
(MME resamples to 44.1 kHz, which wastes headroom below Nyquist).
"""

from __future__ import annotations

from dataclasses import dataclass

import sounddevice as sd

PREFERRED_HOSTAPI = "Windows WASAPI"


class DeviceError(RuntimeError):
    """Raised when a requested audio device cannot be found or opened."""


@dataclass(frozen=True)
class Device:
    index: int
    name: str
    samplerate: int
    channels: int


def _hostapi_index(name: str) -> int:
    for i, api in enumerate(sd.query_hostapis()):
        if api["name"] == name:
            return i
    raise DeviceError(f"Host API {name!r} not available")


def list_devices(kind: str) -> list[Device]:
    """All WASAPI devices of `kind` ('input' or 'output')."""
    if kind not in ("input", "output"):
        raise ValueError(f"kind must be 'input' or 'output', got {kind!r}")
    api = _hostapi_index(PREFERRED_HOSTAPI)
    key = "max_input_channels" if kind == "input" else "max_output_channels"
    out = []
    for i, d in enumerate(sd.query_devices()):
        if d["hostapi"] == api and d[key] > 0:
            out.append(Device(i, d["name"], int(d["default_samplerate"]), int(d[key])))
    return out


def find_device(kind: str, name_substring: str | None) -> Device:
    """Resolve a device by case-insensitive name substring; None means the WASAPI default."""
    devices = list_devices(kind)
    if not devices:
        raise DeviceError(f"No WASAPI {kind} devices found")
    if name_substring is None:
        api = sd.query_hostapis(_hostapi_index(PREFERRED_HOSTAPI))
        default = api["default_input_device" if kind == "input" else "default_output_device"]
        for d in devices:
            if d.index == default:
                return d
        return devices[0]
    needle = name_substring.lower()
    matches = [d for d in devices if needle in d.name.lower()]
    if not matches:
        names = ", ".join(d.name for d in devices)
        raise DeviceError(f"No {kind} device matching {name_substring!r}. Available: {names}")
    return matches[0]
