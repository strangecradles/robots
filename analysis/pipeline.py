"""Offline pipeline: session + device exports -> aligned dataset + a(t) + validity.

    PYTHONPATH=. .venv/bin/python analysis/pipeline.py <session_dir> \
        [--pupil path --slo path]    (defaults to <session_dir>/synthetic/*)

Steps:
    1. ingest device exports via the adapters
    2. align both device clocks to the master clock (sync-flash anchors)
    3. fit gaze calibration from the calib phase; map raw gaze -> display
       coords -> visual degrees
    4. extract attention channels; fuse into a(t)
    5. raycast gaze onto the scene -> attended-object labels
    6. validity report (event-locked a(t), transit baseline, truth corr)

Outputs in <session_dir>/derived/:
    slo_aligned.parquet, pupil_aligned.parquet, gaze.parquet,
    attention.parquet (channels + a/a_z/a_conf), gaze_objects.parquet,
    calibration.json, alignment.json, validity.png, validity.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from align.align import align_stream, to_master
from analysis.raycast import label_gaze_frames
from analysis.validity import validity_report
from attention.a_of_t import fuse
from attention.features import extract_features
from ingest.pupil_adapter import load_pupil
from ingest.slo_adapter import load_slo_trace
from teleop.calibrate import apply_calibration, fit_calibration
from teleop.env import TeleopEnv

FOVY_DEG = 50.0


def run(session_dir: str | Path, pupil_path: str | Path | None = None,
        slo_path: str | Path | None = None) -> dict:
    session_dir = Path(session_dir)
    derived = session_dir / "derived"
    derived.mkdir(exist_ok=True)

    frames = pd.read_parquet(session_dir / "frames.parquet")
    events = [json.loads(l) for l in open(session_dir / "events.jsonl")]
    meta = json.loads((session_dir / "meta.json").read_text())
    w, h = meta.get("render_size", [960, 720])
    aspect = w / h

    pupil_path = pupil_path or session_dir / "synthetic" / "pupil.csv"
    slo_path = slo_path or session_dir / "synthetic" / "slo.csv"

    # 1-2. ingest + align ---------------------------------------------------
    pupil_stream = load_pupil(pupil_path)
    slo_stream = load_slo_trace(slo_path)
    pupil_fit = align_stream(pupil_stream, session_dir)
    slo_fit = align_stream(slo_stream, session_dir)
    print("[align] pupil:", pupil_fit.summary())
    print("[align] slo:  ", slo_fit.summary())
    pupil = to_master(pupil_stream, pupil_fit)
    slo = to_master(slo_stream, slo_fit)

    # 3. calibration --------------------------------------------------------
    calib_events = [e for e in events if e.get("kind") == "calib_point_on"]
    calib = fit_calibration(
        pupil["t_master"].to_numpy(),
        pupil[["gx_raw", "gy_raw"]].to_numpy(), calib_events)
    print(f"[calib] {calib['n_points']} points, rms {calib['rms_norm']:.4f} "
          f"(normalized display units)")
    gaze_norm = apply_calibration(calib, pupil[["gx_raw", "gy_raw"]].to_numpy())
    conf_ok = pupil["conf"].to_numpy() > 0.5
    gaze_norm[~conf_ok] = np.nan

    gaze = pd.DataFrame({
        "t_master": pupil["t_master"],
        "x_norm": gaze_norm[:, 0],
        "y_norm": gaze_norm[:, 1],
        "gx_deg": (gaze_norm[:, 0] - 0.5) * FOVY_DEG * aspect,
        "gy_deg": (gaze_norm[:, 1] - 0.5) * FOVY_DEG,
        "conf": pupil["conf"],
    })

    # 4. attention channels + fusion ----------------------------------------
    feats = extract_features(slo, pupil, gaze, frames)
    attn = fuse(feats)

    # 5. attended-object labels (on the attention grid for compactness) -----
    teleop_frames = frames[frames["phase"] == "teleop"].reset_index(drop=True)
    grid_t = attn["t_master"].to_numpy()
    gx = np.interp(grid_t, gaze["t_master"], gaze["x_norm"])
    gy = np.interp(grid_t, gaze["t_master"], gaze["y_norm"])
    m = (grid_t >= teleop_frames["t_master"].min()) & \
        (grid_t <= teleop_frames["t_master"].max())
    env = TeleopEnv(camera=meta.get("camera", "teleop_cam"))
    gaze_obj = label_gaze_frames(env, teleop_frames, grid_t[m],
                                 np.stack([gx[m], gy[m]], axis=1), aspect)
    print("[raycast] attended objects:",
          gaze_obj["label"].value_counts().to_dict())

    # 6. validity ------------------------------------------------------------
    stats = validity_report(session_dir, attn, events, frames, derived)
    print("[validity]", json.dumps(stats, indent=2))

    # save -------------------------------------------------------------------
    slo.to_parquet(derived / "slo_aligned.parquet", index=False)
    pupil.to_parquet(derived / "pupil_aligned.parquet", index=False)
    gaze.to_parquet(derived / "gaze.parquet", index=False)
    attn.to_parquet(derived / "attention.parquet", index=False)
    gaze_obj.to_parquet(derived / "gaze_objects.parquet", index=False)
    (derived / "calibration.json").write_text(json.dumps(calib, indent=2))
    (derived / "alignment.json").write_text(json.dumps({
        "pupil": {"a": pupil_fit.a, "b": pupil_fit.b, "rms_ms": pupil_fit.rms_ms,
                  "n_anchors": pupil_fit.n_anchors},
        "slo": {"a": slo_fit.a, "b": slo_fit.b, "rms_ms": slo_fit.rms_ms,
                "n_anchors": slo_fit.n_anchors},
    }, indent=2))
    print(f"[pipeline] derived data -> {derived}")
    return stats


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("session_dir")
    p.add_argument("--pupil", default=None)
    p.add_argument("--slo", default=None)
    args = p.parse_args()
    run(args.session_dir, args.pupil, args.slo)
