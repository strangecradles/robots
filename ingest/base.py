"""Device-stream contract consumed by align/ and attention/.

Both devices are record-only and export files after the session. Whatever
the native export format turns out to be, an adapter must produce a
DeviceStream with:

    t_dev   timestamps on the DEVICE clock (seconds, monotonic)
    data    DataFrame with the device's channels:
              pupil:  gx_raw, gy_raw   raw gaze in device units
                      diam             pupil diameter (mm or device units)
                      illum            scene/eye illumination proxy, used to
                                       find the photic flash anchors
                      conf             sample confidence in [0, 1]
              slo:    x_arcmin, y_arcmin  fine eye-position trace from
                                          retinal registration
                      illum            retinal illumination proxy (flash
                                       anchor channel)
                      quality          registration quality in [0, 1]
                                       (drops/NaNs during big saccades and
                                       blinks, when the SLO loses lock)

align.align fits t_master = f(t_dev) per stream from the flash/saccade
anchors; nothing downstream ever touches a device clock again.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd


@dataclasses.dataclass
class DeviceStream:
    kind: str                 # "pupil" | "slo"
    t_dev: np.ndarray         # (N,) seconds, device clock
    data: pd.DataFrame        # (N, channels) see module docstring
    meta: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        self.t_dev = np.asarray(self.t_dev, dtype=float)
        if len(self.t_dev) != len(self.data):
            raise ValueError("t_dev and data length mismatch")

    @property
    def rate_hz(self) -> float:
        if len(self.t_dev) < 2:
            return float("nan")
        return 1.0 / float(np.median(np.diff(self.t_dev)))
