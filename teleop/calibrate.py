"""Gaze calibration: display side (live) + mapping fit (offline).

Display side: shows a sequence of fixation dots on a constant mid-gray
background (constant luminance = clean pupil baseline), logging
`calib_point_on/off` events on the master clock and recording every frame
through the shared SessionRecorder, so the calibration segment lives in the
same timebase/video as the teleop segment.

Offline side: `fit_calibration` maps raw pupil-cam gaze coordinates to
normalized display coordinates with a quadratic polynomial, using the median
raw gaze inside each dot-hold window. It must be called with gaze timestamps
ALREADY mapped onto the master clock (see align.align), since the pupil cam
records on its own clock.
"""

from __future__ import annotations

import time

import numpy as np

from teleop.display import TeleopDisplay, draw_dot, draw_hud, mean_luminance

GRAY = 110  # background level (constant-luminance baseline for the pupil)


def calibration_points(n_points: int = 9) -> list[tuple[float, float]]:
    if n_points == 5:
        return [(0.5, 0.5), (0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)]
    if n_points == 9:
        g = (0.1, 0.5, 0.9)
        pts = [(0.5, 0.5)] + [(x, y) for y in g for x in g if (x, y) != (0.5, 0.5)]
        return pts
    raise ValueError("n_points must be 5 or 9")


def run_calibration_phase(display: TeleopDisplay, recorder, kb,
                          width: int, height: int, fps: float,
                          n_points: int = 9, hold_s: float = 1.6,
                          gap_s: float = 0.5) -> bool:
    """Blocking calibration loop. Returns False if the operator aborted."""
    pts = calibration_points(n_points)
    tick = 1.0 / fps
    next_t = time.perf_counter()

    for idx, (x, y) in enumerate(pts):
        for stage, dur in (("gap", gap_s), ("hold", hold_s)):
            stage_end = recorder.now() + dur
            logged_on = False
            while recorder.now() < stage_end:
                state = kb.poll()
                if state.quit:
                    return False
                t = recorder.now()
                frame = np.full((height, width, 3), GRAY, dtype=np.uint8)
                if stage == "hold":
                    draw_dot(frame, x, y)
                    if not logged_on:
                        logged_on = True
                        recorder.add_event({
                            "t_master": t, "kind": "calib_point_on",
                            "point_idx": idx, "x_norm": x, "y_norm": y,
                            "hold_s": hold_s})
                draw_hud(frame, [f"calibration {idx + 1}/{len(pts)} - follow the dot"])
                recorder.add_frame(
                    frame, t_master=t, phase="calib", paused=False,
                    mean_lum=mean_luminance(frame), flash=0.0, cue_label="",
                    task_event="", manual_mark=False, env_row=None)
                display.show(frame)
                next_t += tick
                time.sleep(max(0.0, next_t - time.perf_counter()))
            if stage == "hold":
                recorder.add_event({
                    "t_master": recorder.now(), "kind": "calib_point_off",
                    "point_idx": idx})
    return True


# --------------------------------------------------------------- offline fit


def _poly_features(xy: np.ndarray) -> np.ndarray:
    x, y = xy[:, 0], xy[:, 1]
    return np.stack([np.ones_like(x), x, y, x * x, y * y, x * y], axis=1)


def fit_calibration(gaze_t: np.ndarray, gaze_xy_raw: np.ndarray,
                    calib_events: list[dict],
                    settle_s: float = 0.4) -> dict:
    """Fit raw pupil-cam gaze -> normalized display coords.

    gaze_t: gaze timestamps already on the MASTER clock (s).
    gaze_xy_raw: (N, 2) raw gaze from the pupil adapter.
    calib_events: parsed events.jsonl dicts with kind == "calib_point_on".
    """
    raw_pts, disp_pts = [], []
    for ev in calib_events:
        if ev.get("kind") != "calib_point_on":
            continue
        t0 = ev["t_master"] + settle_s
        t1 = ev["t_master"] + ev.get("hold_s", 1.6) - 0.1
        m = (gaze_t >= t0) & (gaze_t <= t1) & np.isfinite(gaze_xy_raw).all(axis=1)
        if m.sum() < 3:
            continue
        raw_pts.append(np.median(gaze_xy_raw[m], axis=0))
        disp_pts.append([ev["x_norm"], ev["y_norm"]])
    if len(raw_pts) < 5:
        raise RuntimeError(
            f"only {len(raw_pts)} usable calibration points; need >= 5")

    raw = np.asarray(raw_pts)
    disp = np.asarray(disp_pts)
    feats = _poly_features(raw)
    coef_x, *_ = np.linalg.lstsq(feats, disp[:, 0], rcond=None)
    coef_y, *_ = np.linalg.lstsq(feats, disp[:, 1], rcond=None)
    pred = np.stack([feats @ coef_x, feats @ coef_y], axis=1)
    rms = float(np.sqrt(np.mean(np.sum((pred - disp) ** 2, axis=1))))
    return {"coef_x": coef_x.tolist(), "coef_y": coef_y.tolist(),
            "rms_norm": rms, "n_points": len(raw_pts)}


def apply_calibration(calib: dict, gaze_xy_raw: np.ndarray) -> np.ndarray:
    feats = _poly_features(np.atleast_2d(gaze_xy_raw))
    out = np.stack([feats @ np.asarray(calib["coef_x"]),
                    feats @ np.asarray(calib["coef_y"])], axis=1)
    return out
