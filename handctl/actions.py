"""Output vocabulary of the gesture layer. Pure data; `output.py` turns it into OS input."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Kind = Literal["move", "down", "up", "click", "scroll", "wake"]
Button = Literal["left", "right"]


@dataclass(frozen=True)
class Action:
    kind: Kind
    button: Button | None = None
    dx: float = 0.0   # move: screen pixels; scroll: horizontal wheel units (+ = scroll right)
    dy: float = 0.0   # move: screen pixels; scroll: vertical wheel units (+ = scroll up, Windows convention)
