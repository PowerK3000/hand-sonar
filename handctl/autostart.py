"""Run-at-login via the per-user Run key (no admin, trivially reversible).

    python -m handctl.autostart status|enable|disable
"""

from __future__ import annotations

import sys
import winreg
from pathlib import Path

from . import paths

RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
NAME = "HandControl"


LAUNCHER = paths.ROOT / "run_handctl.pyw"


def command() -> str:
    # pythonw: no console window at login. The launcher puts the repo on sys.path itself,
    # because the Run key has no working-directory field.
    exe = Path(sys.executable).with_name("pythonw.exe")
    if not exe.is_file():
        raise FileNotFoundError(f"pythonw.exe not found next to {sys.executable}")
    if not LAUNCHER.is_file():
        raise FileNotFoundError(f"launcher missing: {LAUNCHER}")
    return f'"{exe}" "{LAUNCHER}"'


def status() -> str | None:
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as k:
            return winreg.QueryValueEx(k, NAME)[0]
    except FileNotFoundError:
        return None


def enable() -> str:
    cmd = command()
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as k:
        winreg.SetValueEx(k, NAME, 0, winreg.REG_SZ, cmd)
    return cmd


def disable() -> bool:
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as k:
            winreg.DeleteValue(k, NAME)
        return True
    except FileNotFoundError:
        return False


def main(argv: list[str]) -> int:
    if len(argv) != 1 or argv[0] not in ("status", "enable", "disable"):
        print(__doc__)
        return 2
    if argv[0] == "status":
        print(status() or "not enabled")
    elif argv[0] == "enable":
        print(f"enabled: {enable()}")
    else:
        print("disabled" if disable() else "was not enabled")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
