"""Synthetic SLO + pupil exports from a recorded session.

Reads a real session directory (frames.parquet + events.jsonl) and fabricates
physiologically-structured device exports on SKEWED device clocks, plus the
generating ground truth. Purpose: validate the entire offline pipeline
(anchor detection -> clock fit -> calibration fit -> feature extraction ->
a(t) -> validity plot) before the real device exports exist.

Embedded structure (what the pipeline must recover):
- ground-truth attention a_true(t): bumps at grasp/lift/place/drop, elevated
  "quiet-eye" window before each grasp, low during ballistic transit/idle
- gaze: fixations on task-relevant targets with ~250 ms lead over the robot
  state, saccades with main-sequence kinematics, drift + microsaccades
- microsaccade rate inhibited by a_true (rate ~ lambda0 * (1 - 0.8 a))
- pupil: light reflex driven by displayed luminance (flashes included),
  TEPR bump tracking a_true with ~1 s lag, plus noise
- SLO: 480 Hz fine trace; loses lock (NaN, low quality) during big saccades
  and blinks; illum channel sees the display flashes
- device clocks: t_dev = (t_master + offset) * (1 + drift)

Outputs in <session>/synthetic/:
    pupil.csv, slo.csv          device-style exports (load via adapters)
    truth.parquet               t_master grid with a_true, gaze, msacc flags
    truth.json                  clock params + generator config
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.raycast import world_to_pixel
from teleop.env import BIN_CENTER, TeleopEnv

SLO_HZ = 480.0
PUPIL_HZ = 120.0
DEG_PER_YNORM = 50.0          # display vertical extent in visual degrees (fovy)
SACCADE_LOCK_DEG = 1.5        # SLO loses registration above this amplitude

CLOCKS = {
    "pupil": {"offset": 13.7, "drift": 5e-5},
    "slo": {"offset": 4.2, "drift": -8e-5},
}


def _smooth(x: np.ndarray, sigma_samples: float) -> np.ndarray:
    if sigma_samples <= 0:
        return x
    n = int(sigma_samples * 4) | 1
    k = np.exp(-0.5 * ((np.arange(n) - n // 2) / sigma_samples) ** 2)
    k /= k.sum()
    return np.convolve(x, k, mode="same")


def _attention_truth(t: np.ndarray, events: list[dict]) -> np.ndarray:
    """a_true(t) in [0, 1] from task events."""
    a = np.full_like(t, 0.15)
    for ev in events:
        if ev.get("kind") != "task_event":
            continue
        te = ev["t_master"]
        label = ev["label"]
        peaks = {"grasp": 0.9, "lift": 0.6, "place_success": 0.8,
                 "drop": 1.0, "release": 0.5}
        if label in peaks:
            a += peaks[label] * np.exp(-0.5 * ((t - te) / 0.45) ** 2)
        if label == "grasp":  # quiet-eye window before the grasp
            a += 0.55 * np.exp(-0.5 * ((t - (te - 0.5)) / 0.35) ** 2)
    return np.clip(a, 0.0, 1.0)


def _gaze_targets(frames: pd.DataFrame, events: list[dict], t: np.ndarray,
                  env: TeleopEnv, aspect: float, lead_s: float = 0.25) -> np.ndarray:
    """Target display coords per sample, with gaze leading robot state."""
    ft = frames["t_master"].to_numpy()
    # Gaze leads: read the frame `lead_s` in the future.
    idx = np.searchsorted(ft, t + lead_s, side="right") - 1
    idx = np.clip(idx, 0, len(frames) - 1)
    f = frames.iloc[idx].reset_index(drop=True)

    cube_px = np.array([world_to_pixel(env, p, aspect) for p in
                        f[["cube_x", "cube_y", "cube_z"]].to_numpy()])
    bin_px = np.array(world_to_pixel(
        env, np.array([BIN_CENTER[0], BIN_CENTER[1], 0.03]), aspect))

    targets = np.where(f["grasped"].to_numpy(dtype=bool)[:, None],
                       np.broadcast_to(bin_px, cube_px.shape), cube_px)

    # Orienting response: stare at the cube for 1 s after a drop.
    for ev in events:
        if ev.get("kind") == "task_event" and ev["label"] == "drop":
            m = (t >= ev["t_master"]) & (t <= ev["t_master"] + 1.0)
            targets[m] = cube_px[m]

    # Calibration dots and saccade cues override everything.
    for ev in events:
        if ev.get("kind") == "calib_point_on":
            m = (t >= ev["t_master"]) & (t <= ev["t_master"] + ev.get("hold_s", 1.6))
            targets[m] = [ev["x_norm"], ev["y_norm"]]
        elif ev.get("kind") == "saccade_cue":
            m = (t >= ev["t_master"]) & (t <= ev["t_master"] + 1.5)
            targets[m] = [ev["x_norm"], ev["y_norm"]]

    # Calib-phase gaps and any off-screen projections leave NaN targets
    # (env columns are NaN during the calib phase): rest at screen center.
    bad = ~np.isfinite(targets).all(axis=1)
    targets[bad] = [0.5, 0.5]
    return targets


def _synth_eye_trace(t: np.ndarray, targets: np.ndarray, a_true: np.ndarray,
                     rng: np.random.Generator):
    """Fixation/saccade/microsaccade simulator on the master timeline.

    Returns (gaze_xy display-norm, msacc_flag, big_saccade_amp_deg).
    """
    n = len(t)
    dt = float(np.median(np.diff(t)))
    gaze = np.zeros((n, 2))
    msacc = np.zeros(n, dtype=bool)
    sacc_amp = np.zeros(n)

    pos = targets[0].copy()
    sacc_end = 0.0
    sacc_from = pos.copy()
    sacc_to = pos.copy()
    sacc_start = -1.0
    sacc_vgain = 1.0
    next_msacc = rng.exponential(0.5)
    msacc_end = 0.0
    msacc_from = pos.copy()
    msacc_to = pos.copy()
    msacc_start = -1.0

    for i in range(n):
        ti = t[i]
        tgt = targets[i]
        err_deg = np.linalg.norm(tgt - pos) * DEG_PER_YNORM

        if ti >= sacc_end and ti >= msacc_end and err_deg > 1.0:
            # Big saccade. Main-sequence duration; vigor (speed) rises with
            # attention -> shorter duration for the same amplitude when a high.
            amp = err_deg
            sacc_vgain = 1.0 + 0.4 * (a_true[i] - 0.5) + rng.normal(0, 0.06)
            dur = ((21.0 + 2.2 * amp) / 1000.0) / max(sacc_vgain, 0.5)
            sacc_start, sacc_end = ti, ti + dur
            sacc_from, sacc_to = pos.copy(), tgt.copy()

        if ti < sacc_end:
            s = np.clip((ti - sacc_start) / max(sacc_end - sacc_start, 1e-3), 0, 1)
            blend = 10 * s**3 - 15 * s**4 + 6 * s**5
            pos = sacc_from + (sacc_to - sacc_from) * blend
            sacc_amp[i] = np.linalg.norm(sacc_to - sacc_from) * DEG_PER_YNORM
        elif ti < msacc_end:
            # Microsaccade in flight (also min-jerk, ~6-25 ms).
            s = np.clip((ti - msacc_start) / max(msacc_end - msacc_start, 1e-3), 0, 1)
            blend = 10 * s**3 - 15 * s**4 + 6 * s**5
            pos = msacc_from + (msacc_to - msacc_from) * blend
        else:
            # Fixation: ocular drift (slower + tighter when attending) plus
            # attention-inhibited microsaccades. drift_sd in display-norm
            # units/sample; ~2e-5 norm @ 480 Hz ~= 30 arcmin/s RMS drift,
            # the physiological resting rate, dropping under attention.
            drift_sd = 2.2e-5 * (1.0 - 0.6 * a_true[i])
            pos = pos + rng.normal(0, drift_sd, 2)
            next_msacc -= dt * (1.0 - 0.8 * a_true[i])
            if next_msacc <= 0:
                amp_deg = rng.uniform(0.1, 0.5)  # 6-30 arcmin
                ang = rng.uniform(0, 2 * np.pi)
                step = amp_deg / DEG_PER_YNORM * np.array([np.cos(ang), np.sin(ang)])
                dur = (12.0 + 2.5 * amp_deg) / 1000.0
                msacc_start, msacc_end = ti, ti + dur
                msacc_from, msacc_to = pos.copy(), pos + step
                msacc[i] = True
                next_msacc = rng.exponential(1.0 / 1.8)
        gaze[i] = pos
    return gaze, msacc, sacc_amp


def generate(session_dir: str | Path, seed: int = 7) -> Path:
    session_dir = Path(session_dir)
    rng = np.random.default_rng(seed)
    frames = pd.read_parquet(session_dir / "frames.parquet")
    events = [json.loads(l) for l in open(session_dir / "events.jsonl")]
    meta = json.loads((session_dir / "meta.json").read_text())
    w, h = meta.get("render_size", [960, 720])
    aspect = w / h

    # Only used for the display-camera projection; must match the camera
    # the session was displayed/recorded with.
    env = TeleopEnv(camera=meta.get("camera", "teleop_cam"))
    import mujoco
    mujoco.mj_forward(env.model, env.data)

    t_end = float(frames["t_master"].max())
    t = np.arange(0.0, t_end, 1.0 / SLO_HZ)

    a_true = _attention_truth(t, events)
    targets = _gaze_targets(frames, events, t, env, aspect)
    gaze, msacc, sacc_amp = _synth_eye_trace(t, targets, a_true, rng)

    # Display luminance resampled onto the fine grid (drives PLR + illum).
    lum = np.interp(t, frames["t_master"], frames["mean_lum"])

    # Blinks: ~12/min, 250 ms.
    blink = np.zeros(len(t), dtype=bool)
    tb = 2.0
    while tb < t_end:
        blink[(t >= tb) & (t <= tb + 0.25)] = True
        tb += rng.uniform(3.0, 7.0)

    # ------------------------------------------------------------------ SLO
    # Absolute eye position in arcmin (drift + microsaccades live here). The
    # feature extractor uses velocity (gradient), which is offset-invariant,
    # so no detrending is needed -- and detrending would inject a spurious
    # post-saccade transient that corrupts the drift/fixation-stability
    # channel. Lock is lost (NaN) during big saccades + blinks.
    x_arc = gaze[:, 0] * DEG_PER_YNORM * 60.0 + rng.normal(0, 0.15, len(t))
    y_arc = gaze[:, 1] * DEG_PER_YNORM * 60.0 + rng.normal(0, 0.15, len(t))
    # SLO loses registration during big saccades + blinks, plus a short
    # re-acquisition window afterward (mimics strip re-locking).
    lock_lost = (sacc_amp > SACCADE_LOCK_DEG) | blink
    reacq = int(SLO_HZ * 0.04)
    lock_lost = np.convolve(lock_lost.astype(float),
                            np.ones(reacq), mode="full")[:len(t)] > 0
    quality = np.where(lock_lost, 0.05, 0.95) + rng.normal(0, 0.02, len(t))
    x_arc[lock_lost] = np.nan
    y_arc[lock_lost] = np.nan
    slo_illum = 1.0 + 4.0 * lum + rng.normal(0, 0.05, len(t))

    ck = CLOCKS["slo"]
    slo = pd.DataFrame({
        "t_dev": (t + ck["offset"]) * (1.0 + ck["drift"]),
        "x_arcmin": x_arc, "y_arcmin": y_arc,
        "illum": slo_illum, "quality": np.clip(quality, 0, 1),
    })

    # ---------------------------------------------------------------- pupil
    pi = np.arange(0, len(t), int(SLO_HZ / PUPIL_HZ))
    tp = t[pi]
    # PLR: low-passed luminance with ~300 ms lag; TEPR: a_true with ~1 s lag.
    plr = _smooth(lum, SLO_HZ * 0.3)[pi]
    tepr_kernel_lag = int(SLO_HZ * 1.0)
    tepr = _smooth(np.concatenate([np.zeros(tepr_kernel_lag),
                                   a_true])[: len(t)], SLO_HZ * 0.5)[pi]
    diam = 4.2 - 2.2 * plr + 0.35 * tepr + rng.normal(0, 0.02, len(tp))
    diam[blink[pi]] = np.nan

    # Raw gaze in device units: affine + mild nonlinearity of display coords.
    gx_n, gy_n = gaze[pi, 0], gaze[pi, 1]
    gx_raw = 240.0 * gx_n + 12.0 + 8.0 * gx_n**2 + rng.normal(0, 0.4, len(tp))
    gy_raw = 180.0 * gy_n + 30.0 + 6.0 * gy_n * gx_n + rng.normal(0, 0.4, len(tp))
    conf = np.where(blink[pi], 0.0, 0.97)

    ck = CLOCKS["pupil"]
    pupil = pd.DataFrame({
        "t_dev": (tp + ck["offset"]) * (1.0 + ck["drift"]),
        "gx_raw": gx_raw, "gy_raw": gy_raw, "diam": diam,
        "illum": lum[pi] + rng.normal(0, 0.01, len(tp)),
        "conf": conf,
    })

    # ----------------------------------------------------------------- save
    out = session_dir / "synthetic"
    out.mkdir(exist_ok=True)
    pupil.to_csv(out / "pupil.csv", index=False)
    slo.to_csv(out / "slo.csv", index=False)
    pd.DataFrame({
        "t_master": t, "a_true": a_true,
        "gaze_x_norm": gaze[:, 0], "gaze_y_norm": gaze[:, 1],
        "microsaccade": msacc, "blink": blink, "saccade_amp_deg": sacc_amp,
    }).to_parquet(out / "truth.parquet", index=False)
    (out / "truth.json").write_text(json.dumps(
        {"clocks": CLOCKS, "slo_hz": SLO_HZ, "pupil_hz": PUPIL_HZ,
         "deg_per_ynorm": DEG_PER_YNORM, "seed": seed}, indent=2))
    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("session_dir")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    print("synthetic exports ->", generate(args.session_dir, args.seed))
