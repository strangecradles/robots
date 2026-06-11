"""SLO export -> DeviceStream.

Two entry points, depending on what the device hands us tomorrow:

1. `load_slo_trace(path)`: the SLO software already exports an eye-motion
   trace (t, x, y from retinal strip registration). Column-mapped like the
   pupil adapter.

2. `trace_from_video(path, fps)`: we only get raw retinal video. Produces a
   coarse frame-rate eye-motion trace by phase-correlating consecutive
   frames (translation of the retinal image = eye rotation; arcmin scale
   must be set from the device FOV). This loses intra-frame (strip-rate)
   detail but is enough to (a) find flash anchors via the illum channel,
   (b) detect microsaccades at frame rate if the SLO runs fast enough.
   Replace with the vendor's strip-registration trace when available.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ingest.base import DeviceStream

DEFAULT_COLMAP = {
    "t_dev": "t_dev",
    "x_arcmin": "x_arcmin",
    "y_arcmin": "y_arcmin",
    "illum": "illum",
    "quality": "quality",
}


def load_slo_trace(path: str | Path, colmap: dict | None = None) -> DeviceStream:
    cm = {**DEFAULT_COLMAP, **(colmap or {})}
    p = Path(path)
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)

    out = pd.DataFrame()
    t_dev = df[cm["t_dev"]].to_numpy(dtype=float)
    out["x_arcmin"] = df[cm["x_arcmin"]].to_numpy(dtype=float)
    out["y_arcmin"] = df[cm["y_arcmin"]].to_numpy(dtype=float)
    out["illum"] = (df[cm["illum"]].to_numpy(dtype=float)
                    if cm["illum"] in df.columns else np.zeros(len(df)))
    out["quality"] = (df[cm["quality"]].to_numpy(dtype=float)
                      if cm["quality"] in df.columns else np.ones(len(df)))

    order = np.argsort(t_dev, kind="stable")
    return DeviceStream(kind="slo", t_dev=t_dev[order],
                        data=out.iloc[order].reset_index(drop=True),
                        meta={"path": str(p)})


def trace_from_video(path: str | Path, arcmin_per_px: float = 1.0,
                     t0_dev: float = 0.0) -> DeviceStream:
    """Coarse eye trace from raw SLO video via inter-frame phase correlation."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open SLO video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    ts, xs, ys, illum, quality = [], [], [], [], []
    prev = None
    x = y = 0.0
    i = 0
    hann = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if hann is None:
            hann = cv2.createHanningWindow(gray.shape[::-1], cv2.CV_32F)
        if prev is not None:
            (dx, dy), response = cv2.phaseCorrelate(prev, gray, hann)
            # Eye moved opposite to retinal-image shift.
            x -= dx * arcmin_per_px
            y -= dy * arcmin_per_px
            quality.append(float(response))
        else:
            quality.append(1.0)
        ts.append(t0_dev + i / fps)
        xs.append(x)
        ys.append(y)
        illum.append(float(gray.mean()))
        prev = gray
        i += 1
    cap.release()

    data = pd.DataFrame({"x_arcmin": xs, "y_arcmin": ys,
                         "illum": illum, "quality": quality})
    return DeviceStream(kind="slo", t_dev=np.asarray(ts), data=data,
                        meta={"path": str(path), "fps": fps,
                              "arcmin_per_px": arcmin_per_px})
