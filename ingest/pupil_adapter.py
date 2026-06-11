"""Pupil-camera export -> DeviceStream.

The exact export format of the headset's pupil camera is unknown until we
see it at the rig; this adapter therefore takes a column map. The default
matches the synthetic generator. For a new format, call with e.g.

    load_pupil(path, colmap={"t_dev": "timestamp_s", "gx_raw": "px",
                             "gy_raw": "py", "diam": "pupil_diameter_mm",
                             "illum": "ir_brightness", "conf": "confidence"})

If the export is a video instead of a table, extract a table first (pupil
center + diameter per frame) with whatever vendor tooling exists, then load
it here; only `t_dev`, `gx_raw`, `gy_raw`, `diam` are strictly required
(`illum`/`conf` get sane defaults).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ingest.base import DeviceStream

DEFAULT_COLMAP = {
    "t_dev": "t_dev",
    "gx_raw": "gx_raw",
    "gy_raw": "gy_raw",
    "diam": "diam",
    "illum": "illum",
    "conf": "conf",
}


def load_pupil(path: str | Path, colmap: dict | None = None) -> DeviceStream:
    cm = {**DEFAULT_COLMAP, **(colmap or {})}
    p = Path(path)
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)

    out = pd.DataFrame()
    t_dev = df[cm["t_dev"]].to_numpy(dtype=float)
    for key in ("gx_raw", "gy_raw", "diam"):
        out[key] = df[cm[key]].to_numpy(dtype=float)
    out["illum"] = (df[cm["illum"]].to_numpy(dtype=float)
                    if cm["illum"] in df.columns else np.zeros(len(df)))
    out["conf"] = (df[cm["conf"]].to_numpy(dtype=float)
                   if cm["conf"] in df.columns else np.ones(len(df)))

    order = np.argsort(t_dev, kind="stable")
    return DeviceStream(kind="pupil", t_dev=t_dev[order],
                        data=out.iloc[order].reset_index(drop=True),
                        meta={"path": str(p)})
