"""Background hand-control app: tray icon, pause hotkey, camera -> controller -> Windows input.

    python -m handctl.app            (pythonw for no console window)

Fail-safe: on pause, quit, or any crash in the worker, held mouse buttons are released first.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
import threading
import time

from . import config, paths
from .controller import Controller
from .engagement import EngagementGate, InputActivityHook
from .hotkey import GlobalHotkey, HotkeyError
from .output import OutputError, WindowsOutput
from .tracker import MODEL_URL, Camera, HandTracker
from .tap_model import TapModelError, from_config as taps_from_config

log = logging.getLogger("handctl")


def setup_logging() -> None:
    paths.LOGS.mkdir(parents=True, exist_ok=True)
    h = logging.handlers.RotatingFileHandler(paths.LOGS / "handctl.log", maxBytes=2_000_000, backupCount=3,
                                             encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(threadName)s %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(h)
    if sys.stderr is not None:  # pythonw has no console
        root.addHandler(logging.StreamHandler())


class HandControlApp:
    def __init__(self, cfg: config.Config) -> None:
        self.cfg = cfg
        self.gate = EngagementGate(cfg.engagement)
        self.controller = Controller(cfg, self.gate, taps_from_config(cfg))
        self.output = WindowsOutput()
        self.hooks = InputActivityHook(self.gate)
        self.hotkey = GlobalHotkey(cfg.app.pause_hotkey, self.toggle_pause)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.icon = None
        self.fatal: BaseException | None = None

    # -- control -------------------------------------------------------------
    def _release(self) -> None:
        with self._lock:
            for a in self.controller.release():
                self._apply(a)
            self.output.release_all()

    def _apply(self, action) -> None:
        try:
            self.output.apply(action)
        except OutputError as e:
            # Typically UIPI: an elevated window is focused. Skip, don't crash the app.
            log.warning("input blocked: %s (%s)", action, e)

    def toggle_pause(self) -> None:
        self.gate.paused = not self.gate.paused
        if self.gate.paused:
            self._release()
        log.info("paused" if self.gate.paused else "resumed")
        if self.icon is not None:
            self.icon.title = self._title()
            self.icon.update_menu()

    def _title(self) -> str:
        return f"Hand control: {'paused' if self.gate.paused else 'active'} ({self.cfg.app.pause_hotkey})"

    def quit(self) -> None:
        self._stop.set()
        if self.icon is not None:
            self.icon.stop()

    # -- worker --------------------------------------------------------------
    def _worker(self) -> None:
        camera = tracker = None
        try:
            tracker = HandTracker(paths.resolve(self.cfg.app.model_path), self.cfg.app.hand,
                                  self.cfg.app.num_hands, self.cfg.app.process_width)
            camera = Camera(self.cfg.app.camera_index)
            log.info("camera %dx%d @ %.0f fps", camera.width, camera.height, camera.fps)
            while not self._stop.is_set():
                t, bgr = camera.latest(timeout=2.0)
                frame = tracker.process(t, bgr)
                actions = self.controller.step(frame, time.monotonic())
                with self._lock:
                    for a in actions:
                        if a.kind != "move":
                            log.info("action %s %s", a.kind, a.button or "")
                        self._apply(a)
        except BaseException as e:  # noqa: BLE001 - any failure must still release buttons
            self.fatal = e
            log.exception("worker stopped")
        finally:
            self._release()
            if camera is not None:
                camera.close()
            if tracker is not None:
                tracker.close()
            self.quit()

    # -- tray ----------------------------------------------------------------
    def _make_icon(self):
        import pystray
        from PIL import Image, ImageDraw
        img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.ellipse((6, 6, 58, 58), fill=(40, 120, 220, 255))
        d.text((22, 18), "H", fill=(255, 255, 255, 255))
        menu = pystray.Menu(
            pystray.MenuItem(lambda _: "Resume" if self.gate.paused else "Pause",
                             lambda: self.toggle_pause(), default=True),
            pystray.MenuItem("Quit", lambda: self.quit()))
        return pystray.Icon("handctl", img, self._title(), menu)

    def run(self) -> int:
        self.hooks.start()
        try:
            self.hotkey.start()
        except HotkeyError as e:
            log.warning("%s; use the tray icon to pause", e)
        worker = threading.Thread(target=self._worker, name="worker", daemon=True)
        worker.start()
        try:
            self.icon = self._make_icon()
            self.icon.run()              # blocks until quit()
        finally:
            self._stop.set()
            worker.join(5)
            self._release()
            self.hotkey.stop()
            self.hooks.stop()
        if self.fatal is not None:
            log.error("exited after error: %s", self.fatal)
            return 1
        return 0


def main() -> int:
    setup_logging()
    try:
        cfg = config.load(paths.CONFIG)
    except config.ConfigError as e:
        log.error("config error: %s", e)
        return 2
    # Fail fast with a clear message before starting threads.
    model = paths.resolve(cfg.app.model_path)
    if not model.is_file():
        log.error("hand model not found at %s. Download it from %s", model, MODEL_URL)
        return 2
    try:
        app = HandControlApp(cfg)
    except TapModelError as e:
        log.error("%s", e)
        return 2
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
