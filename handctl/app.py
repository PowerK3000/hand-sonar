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
from .stabilize import LandmarkStabilizer
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
        self.output = WindowsOutput(cfg.pointer.smooth_ms)
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
            stab = LandmarkStabilizer()
            stats_t0, n_frames, n_hand, track_s, ctl_s = time.monotonic(), 0, 0, 0.0, 0.0
            while not self._stop.is_set():
                t, bgr = camera.latest(timeout=2.0)
                a = time.perf_counter()
                frame = tracker.process(t, bgr)
                track_s += time.perf_counter() - a
                n_frames += 1
                n_hand += frame is not None
                if time.monotonic() - stats_t0 >= 15:
                    el = time.monotonic() - stats_t0
                    log.info("stats: %.1f fps, hand in %d%% of frames, tracking %.0f ms, controller %.1f ms",
                             n_frames / el, 100 * n_hand // max(1, n_frames), 1000 * track_s / max(1, n_frames),
                             1000 * ctl_s / max(1, n_frames))
                    stats_t0, n_frames, n_hand, track_s, ctl_s = time.monotonic(), 0, 0, 0.0, 0.0
                a = time.perf_counter()
                # Clicks read raw landmarks; only the cursor uses the stabilised copy.
                actions = self.controller.step(frame, time.monotonic(), pointer_frame=stab(frame))
                ctl_s += time.perf_counter() - a
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
            self.output.close()
        if self.fatal is not None:
            log.error("exited after error: %s", self.fatal)
            return 1
        return 0


_MUTEX_NAME = r"Local\HandControl.hand-sonar"
_ERROR_ALREADY_EXISTS = 183


def _single_instance():
    """Named mutex held for the process lifetime; None if another instance already holds it.
    With start-at-login plus manual launches, two copies would fight over the camera."""
    import ctypes
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    handle = k.CreateMutexW(None, False, _MUTEX_NAME)
    if not handle:
        raise OSError(ctypes.get_last_error(), "CreateMutexW failed")
    if ctypes.get_last_error() == _ERROR_ALREADY_EXISTS:
        k.CloseHandle(handle)
        return None
    return handle


def main() -> int:
    setup_logging()
    mutex = _single_instance()
    if mutex is None:
        log.info("another instance is already running; exiting")
        return 0
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
