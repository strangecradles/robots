"""3Dconnexion SpaceMouse teleop input (USB or Bluetooth, e.g. Pro Wireless).

Drop-in replacement for ``KeyboardTeleop``: same ``poll() -> InputState`` and
``gripper_closed`` surface, so the session loop and calibration phase don't care
which device is driving them.

The SpaceMouse 6-DoF puck supplies continuous end-effector velocity in the same
VIEW frame the keyboard uses (x = "to my right", y = "away from me", z = up) plus
a yaw rate from the twist axis; ``run_session`` rotates that into world frame via
``TeleopEnv.view_basis()`` so motion stays egocentric for any display camera.

A ``KeyboardTeleop`` is composed underneath so:
  * edge events (ESC quit, N reset, M mark, P pause) and WASD/RF/QE motion keep
    working as a fallback even while the puck is connected, and
  * pygame events stay pumped every tick (required to keep the window alive).

SpaceMouse axes ADD to any keyboard motion, and SpaceMouse buttons can toggle the
gripper / reset the episode. The puck and keyboard therefore share a single
gripper state (``self.kb.gripper_closed``).

Reading uses raw HID via ``pyspacemouse`` (``pip install pyspacemouse hidapi`` or
``uv sync --extra spacemouse``), which works over the Pro Wireless Bluetooth link
on macOS without 3Dconnexion's proprietary driver. Pair the puck in System
Settings > Bluetooth first; it then shows up as a plain HID device.

Run ``PYTHONPATH=. python teleop/spacemouse.py`` to live-test axes/buttons and
discover button indices for ``--sm-gripper-button`` / ``--sm-reset-button``.
"""

from __future__ import annotations

import numpy as np

from teleop.display import InputState, KeyboardTeleop
from teleop.env import TeleopCommand


class SpaceMouseError(RuntimeError):
    """Raised when a SpaceMouse was requested but could not be opened."""


class SpaceMouseTeleop:
    """SpaceMouse 6-DoF teleop with a composed keyboard fallback.

    Args:
        lin_speed: m/s commanded at full puck deflection on a translation axis.
        yaw_speed: rad/s commanded at full twist deflection.
        deadzone: puck readings with ``abs < deadzone`` are zeroed (the cap never
            recenters perfectly). Applied before rescaling so the usable range is
            preserved.
        gripper_button: button index whose rising edge toggles the gripper.
        reset_button: button index whose rising edge resets the episode
            (``-1`` disables).
        invert_x/y/z/yaw: flip an axis sign if your puck's orientation or mount
            makes a direction feel backwards. Defaults match a Pro Wireless sat
            flat in front of the operator, cable/logo facing away.
        keyboard: an existing ``KeyboardTeleop`` to compose (one is created if
            omitted). Sharing one keeps a single gripper state across devices.
        device: 3Dconnexion model name to open (e.g. ``"SpaceMouse Pro Wireless"``);
            ``None`` opens the first supported device found.
        required: if True (default) raise ``SpaceMouseError`` when no device opens;
            if False, degrade silently to keyboard-only.
    """

    def __init__(
        self,
        lin_speed: float = 0.18,
        yaw_speed: float = 1.2,
        deadzone: float = 0.1,
        gripper_button: int = 5,
        reset_button: int = 6,
        invert_x: bool = False,
        invert_y: bool = False,
        invert_z: bool = False,
        invert_yaw: bool = False,
        keyboard: KeyboardTeleop | None = None,
        device: str | None = None,
        required: bool = True,
    ):
        self.kb = keyboard or KeyboardTeleop(lin_speed, yaw_speed)
        self.lin_speed = lin_speed
        self.yaw_speed = yaw_speed
        self.deadzone = deadzone
        self.gripper_button = gripper_button
        self.reset_button = reset_button
        self._sign = np.array(
            [-1.0 if invert_x else 1.0,
             -1.0 if invert_y else 1.0,
             -1.0 if invert_z else 1.0]
        )
        self._yaw_sign = -1.0 if invert_yaw else 1.0
        self._prev_buttons: list[int] = []

        self.device = self._open(device)
        if self.device is None and required:
            raise SpaceMouseError(
                "Could not open a 3Dconnexion SpaceMouse. This integration reads "
                "the device over RAW HID (pyspacemouse); checklist:\n"
                "  1. Pair the Pro Wireless over Bluetooth (System Settings > "
                "Bluetooth) or plug it in.\n"
                "  2. Python deps: `uv sync --extra spacemouse` (or "
                "`pip install pyspacemouse hidapi`).\n"
                "  3. macOS needs the native hidapi lib: `brew install hidapi`.\n"
                "  4. Grant the host app (Terminal/Cursor) Input Monitoring "
                "(System Settings > Privacy & Security > Input Monitoring), then "
                "restart that app.\n"
                "  5. If you installed 3Dconnexion's driver, its daemon SEIZES the "
                "device and raw HID gets 'Failed to open device'. Quit the "
                "3Dconnexion apps (`pkill -f 3Dconnexion; pkill -f 3Dx`) or "
                "uninstall the driver \u2014 the raw-HID path does not use it.\n"
                "Run `PYTHONPATH=. python teleop/spacemouse.py` to live-test."
            )

    @staticmethod
    def _preload_hidapi() -> None:
        """Make the native hidapi library resolvable before easyhid imports it.

        pyspacemouse's HID backend (easyhid) does ``ffi.dlopen("hidapi")`` and,
        on failure, falls back to ``ffi.dlopen(None)`` (the global symbol table).
        On macOS ``ctypes.util.find_library`` doesn't search Homebrew's
        ``/opt/homebrew/lib``, so we proactively load the dylib with RTLD_GLOBAL;
        the fallback then resolves the symbols. No-op (best effort) elsewhere or
        if the system loader already finds it.
        """
        import ctypes
        import ctypes.util
        import glob

        cands: list[str] = []
        found = ctypes.util.find_library("hidapi")
        if found:
            cands.append(found)
        cands += glob.glob("/opt/homebrew/lib/libhidapi*.dylib")
        cands += glob.glob("/usr/local/lib/libhidapi*.dylib")
        cands += glob.glob("/opt/homebrew/Cellar/hidapi/*/lib/libhidapi*.dylib")
        for path in cands:
            try:
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                return
            except OSError:
                continue

    @classmethod
    def _open(cls, device: str | None):
        cls._preload_hidapi()
        try:
            import pyspacemouse  # noqa: PLC0415  (optional dependency)
        except ImportError:
            return None
        try:
            return pyspacemouse.open(device=device, nonblocking=True)
        except Exception as exc:  # hidapi raises bare Exception on access errors
            print(f"[spacemouse] open failed: {exc}")
            return None

    @property
    def connected(self) -> bool:
        return self.device is not None

    # ``run_session`` reads this for the HUD; delegate to the shared keyboard so
    # SPACE and the puck button stay in sync.
    @property
    def gripper_closed(self) -> bool:
        return self.kb.gripper_closed

    @gripper_closed.setter
    def gripper_closed(self, value: bool) -> None:
        self.kb.gripper_closed = value

    def _deadzone(self, v: float) -> float:
        """Zero the centre slop, then rescale [deadzone, 1] -> [0, 1]."""
        a = abs(v)
        if a <= self.deadzone:
            return 0.0
        scaled = (a - self.deadzone) / (1.0 - self.deadzone)
        return float(np.sign(v) * min(scaled, 1.0))

    def poll(self) -> InputState:
        # Keyboard poll pumps pygame events and applies any key-held motion /
        # edge events; the puck is layered on top of whatever it returns.
        state = self.kb.poll()

        if self.device is not None:
            sm = self.device.read()
            if sm is not None:
                ax = np.array([
                    self._deadzone(sm.x),   # right (+) / left (-)
                    self._deadzone(sm.y),   # away (+) / toward (-)
                    self._deadzone(sm.z),   # up (+) / down (-)
                ]) * self._sign
                state.command.lin_vel = state.command.lin_vel + ax * self.lin_speed
                state.command.yaw_rate += (
                    self._yaw_sign * self._deadzone(sm.yaw) * self.yaw_speed
                )
                self._handle_buttons(list(sm.buttons), state)

        state.command.gripper_close = self.kb.gripper_closed
        return state

    def _handle_buttons(self, buttons: list[int], state: InputState) -> None:
        prev = self._prev_buttons
        for i, pressed in enumerate(buttons):
            was = prev[i] if i < len(prev) else 0
            if pressed and not was:  # rising edge
                if i == self.gripper_button:
                    self.kb.gripper_closed = not self.kb.gripper_closed
                elif self.reset_button >= 0 and i == self.reset_button:
                    state.reset = True
        self._prev_buttons = buttons

    def close(self) -> None:
        if self.device is not None:
            try:
                self.device.close()
            except Exception:
                pass
            self.device = None


def _live_test() -> None:
    """Standalone axis/button monitor: verify orientation and button indices."""
    import time

    sm = SpaceMouseTeleop(required=False)
    if not sm.connected:
        print("No SpaceMouse detected. Pair/connect it and install deps:")
        print("  uv sync --extra spacemouse   (or: pip install pyspacemouse hidapi)")
        return
    print("SpaceMouse connected. Move the puck / press buttons (Ctrl-C to stop).")
    print("Showing raw axes and the resulting VIEW-frame velocity [right, away, up].\n")
    # Read the device directly (no pygame/keyboard) so this runs headless.
    try:
        while True:
            raw = sm.device.read()
            if raw is not None:
                ax = np.array([sm._deadzone(raw.x), sm._deadzone(raw.y),
                               sm._deadzone(raw.z)]) * sm._sign
                v = ax * sm.lin_speed
                yaw = sm._yaw_sign * sm._deadzone(raw.yaw) * sm.yaw_speed
                pressed = [i for i, b in enumerate(raw.buttons) if b]
                print(
                    f"\rraw[x{raw.x:+.2f} y{raw.y:+.2f} z{raw.z:+.2f} "
                    f"yaw{raw.yaw:+.2f}]  vel[{v[0]:+.3f} {v[1]:+.3f} {v[2]:+.3f}] "
                    f"yaw {yaw:+.3f}  buttons {pressed}        ",
                    end="", flush=True,
                )
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        sm.close()


if __name__ == "__main__":
    _live_test()
