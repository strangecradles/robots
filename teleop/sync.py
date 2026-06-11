"""Cross-clock synchronization anchors.

The SLO and pupil camera record on their own free-running clocks. To align
them to the master clock offline we emit anchors that every device can see:

- Photic flashes: full-screen white pulses. Sharp luminance step -> pupillary
  light reflex in the pupil cam, retinal-illumination change in the SLO.
  A START burst uses gaps (0.4 s, 0.8 s) and the END burst uses the mirrored
  gaps (0.8 s, 0.4 s), so the two bursts are unambiguous in any recording
  even if the device clock runs fast or slow. Single periodic flashes every
  `period_s` provide interior anchors for drift fitting.
- Cued saccades: a dot jumps left -> right -> center while the operator is
  told to follow it. Large saccades are unmissable kinematic anchors in both
  the SLO trace and the pupil-cam gaze trace.

Every anchor is logged with the master timestamp at which it was actually
first rendered (not just scheduled), and the per-frame log additionally
records flash intensity per displayed frame.
"""

from __future__ import annotations

import dataclasses

START_BURST_OFFSETS = (0.0, 0.4 + 0.12, 0.4 + 0.12 + 0.8 + 0.12)  # on-times
END_BURST_OFFSETS = (0.0, 0.8 + 0.12, 0.8 + 0.12 + 0.4 + 0.12)
FLASH_DUR = 0.12

SACCADE_SEQUENCE = (  # (label, x_norm, y_norm, duration_s)
    ("left", 0.08, 0.5, 1.5),
    ("right", 0.92, 0.5, 1.5),
    ("center", 0.5, 0.5, 1.5),
)


@dataclasses.dataclass
class Flash:
    t_on: float
    t_off: float
    label: str
    logged: bool = False


@dataclasses.dataclass
class Cue:
    t_on: float
    t_off: float
    label: str
    x_norm: float
    y_norm: float
    logged: bool = False


class SyncScheduler:
    """Drives flash/cue overlay state; call `update(t)` once per frame."""

    def __init__(self, period_s: float = 30.0, start_delay: float = 1.0):
        self.period_s = period_s
        self.start_delay = start_delay
        self.flashes: list[Flash] = []
        self.cues: list[Cue] = []
        self.events: list[dict] = []  # consumed by the recorder
        self._t0: float | None = None
        self._next_periodic: float = 0.0
        self._periodic_idx = 0
        self._end_requested = False

    # ------------------------------------------------------------- schedule

    def start(self, t0: float) -> None:
        self._t0 = t0
        burst_start = t0 + self.start_delay
        for i, off in enumerate(START_BURST_OFFSETS):
            self.flashes.append(Flash(burst_start + off, burst_start + off + FLASH_DUR,
                                      f"start_burst_{i}"))
        cue_t = burst_start + START_BURST_OFFSETS[-1] + FLASH_DUR + 1.0
        for label, x, y, dur in SACCADE_SEQUENCE:
            self.cues.append(Cue(cue_t, cue_t + dur, label, x, y))
            cue_t += dur
        self._next_periodic = cue_t + self.period_s

    def request_end(self, t_now: float) -> float:
        """Schedule the end burst; returns the time when it completes."""
        if not self._end_requested:
            self._end_requested = True
            burst_start = t_now + 0.5
            for i, off in enumerate(END_BURST_OFFSETS):
                self.flashes.append(Flash(burst_start + off, burst_start + off + FLASH_DUR,
                                          f"end_burst_{i}"))
        return max(f.t_off for f in self.flashes) + 0.3

    # --------------------------------------------------------------- update

    def update(self, t: float) -> tuple[float, Cue | None]:
        """Returns (flash_intensity, active_cue) and logs newly-on anchors."""
        if self._t0 is None:
            raise RuntimeError("SyncScheduler.start(t0) was never called")

        # Lazily schedule periodic single flashes (not during the end burst).
        if not self._end_requested and t >= self._next_periodic:
            self.flashes.append(Flash(self._next_periodic, self._next_periodic + FLASH_DUR,
                                      f"periodic_{self._periodic_idx}"))
            self._periodic_idx += 1
            self._next_periodic += self.period_s

        intensity = 0.0
        for f in self.flashes:
            if f.t_on <= t < f.t_off:
                intensity = 1.0
                if not f.logged:
                    f.logged = True
                    self.events.append(
                        {"t_master": t, "kind": "flash_on", "label": f.label,
                         "t_scheduled": f.t_on})

        active_cue = None
        for c in self.cues:
            if c.t_on <= t < c.t_off:
                active_cue = c
                if not c.logged:
                    c.logged = True
                    self.events.append(
                        {"t_master": t, "kind": "saccade_cue", "label": c.label,
                         "x_norm": c.x_norm, "y_norm": c.y_norm,
                         "t_scheduled": c.t_on})
        return intensity, active_cue

    def drain_events(self) -> list[dict]:
        out, self.events = self.events, []
        return out
