from __future__ import annotations

import ctypes
import ctypes.util
import platform


class CGPoint(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]


class MacMouseCapture:
    """Best-effort cursor hide/warp for Open3D legacy windows on macOS."""

    def __init__(self, window_left: int, window_top: int, window_width: int, window_height: int) -> None:
        self.window_left = int(window_left)
        self.window_top = int(window_top)
        self.window_width = int(window_width)
        self.window_height = int(window_height)
        self.enabled = False
        self._display_id = 0
        self._app_services = None
        if platform.system() == "Darwin":
            self._load_application_services()

    @property
    def available(self) -> bool:
        return self._app_services is not None

    @property
    def center_local(self) -> tuple[float, float]:
        return self.window_width / 2.0, self.window_height / 2.0

    @property
    def center_global(self) -> CGPoint:
        return CGPoint(
            float(self.window_left + self.window_width / 2.0),
            float(self.window_top + self.window_height / 2.0),
        )

    def activate(self) -> None:
        if not self.available:
            return
        self._app_services.CGDisplayHideCursor(self._display_id)
        self.warp_to_center()
        self.enabled = True

    def deactivate(self) -> None:
        if not self.available:
            return
        self._app_services.CGDisplayShowCursor(self._display_id)
        self.enabled = False

    def warp_to_center(self) -> None:
        if not self.available:
            return
        self._app_services.CGWarpMouseCursorPosition(self.center_global)

    def _load_application_services(self) -> None:
        lib_path = ctypes.util.find_library("ApplicationServices")
        if not lib_path:
            return
        app_services = ctypes.CDLL(lib_path)
        app_services.CGMainDisplayID.restype = ctypes.c_uint32
        app_services.CGDisplayHideCursor.argtypes = [ctypes.c_uint32]
        app_services.CGDisplayShowCursor.argtypes = [ctypes.c_uint32]
        app_services.CGWarpMouseCursorPosition.argtypes = [CGPoint]
        app_services.CGWarpMouseCursorPosition.restype = ctypes.c_int32
        self._app_services = app_services
        self._display_id = int(app_services.CGMainDisplayID())
