"""Repo-relative paths, so the app works the same when launched at login (any cwd)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "gestures.toml"
LOGS = ROOT / "logs"
RECORDINGS = ROOT / "data" / "recordings"


def resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p
