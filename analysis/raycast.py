"""Displayed pixel -> 3D point-of-regard -> MuJoCo object.

The teleop camera is fixed and fully known, and every logged frame row
contains the full sim state, so any (frame_idx, gaze pixel) pair can be
turned into "which object was the operator looking at, and where in 3D".

Conventions: gaze coords are normalized display coords (x right, y DOWN,
both in [0, 1]) on the rendered frame -- same convention as calibrate.py
and the recorded video.
"""

from __future__ import annotations

import mujoco
import numpy as np

from teleop.env import TeleopEnv


def restore_state(env: TeleopEnv, row: dict) -> None:
    """Restore the sim to a logged frame row (positions only) + forward."""
    d = env.data
    for i in range(9):
        d.qpos[i] = row[f"qpos_{i}"]
        d.qvel[i] = 0.0
    adr = env.cube_jnt_qposadr
    d.qpos[adr:adr + 3] = [row["cube_x"], row["cube_y"], row["cube_z"]]
    d.qpos[adr + 3:adr + 7] = [row[f"cube_quat_{k}"] for k in "wxyz"]
    mujoco.mj_forward(env.model, env.data)


def _cam_basis(env: TeleopEnv) -> tuple[np.ndarray, np.ndarray, float]:
    """Camera origin, rotation (cols = right/up/back), vertical fov (rad)."""
    cam_id = env.cam_id
    pos = env.data.cam_xpos[cam_id].copy()
    rot = env.data.cam_xmat[cam_id].reshape(3, 3).copy()
    fovy = np.deg2rad(env.model.cam_fovy[cam_id])
    return pos, rot, fovy


def pixel_to_ray(env: TeleopEnv, x_norm: float, y_norm: float,
                 aspect: float) -> tuple[np.ndarray, np.ndarray]:
    """Normalized display coords -> world-frame ray (origin, unit dir)."""
    pos, rot, fovy = _cam_basis(env)
    ty = np.tan(fovy / 2.0)
    # Camera frame: x right, y up, looks along -z.
    d_cam = np.array([
        (2.0 * x_norm - 1.0) * ty * aspect,
        (1.0 - 2.0 * y_norm) * ty,
        -1.0,
    ])
    d_world = rot @ d_cam
    return pos, d_world / np.linalg.norm(d_world)


def world_to_pixel(env: TeleopEnv, point: np.ndarray,
                   aspect: float) -> tuple[float, float]:
    """Inverse of pixel_to_ray: world point -> normalized display coords.

    Used by the synthetic device generator to fabricate gaze from sim ground
    truth with exactly the same camera model the analysis assumes.
    """
    pos, rot, fovy = _cam_basis(env)
    v = rot.T @ (np.asarray(point) - pos)   # into camera frame
    if v[2] >= -1e-9:
        return float("nan"), float("nan")   # behind the camera
    ty = np.tan(fovy / 2.0)
    x_norm = (v[0] / (-v[2]) / (ty * aspect) + 1.0) / 2.0
    y_norm = (1.0 - v[1] / (-v[2]) / ty) / 2.0
    return float(x_norm), float(y_norm)


# Coarse semantic labels for analysis.
_LABELS = {
    "cube": "cube",
    "bin": "bin",
    "floor": "floor",
}


def _body_label(env: TeleopEnv, body_id: int) -> str:
    name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
    if name in _LABELS:
        return _LABELS[name]
    if name == "world":
        return "floor"
    if name in ("left_finger", "right_finger", "hand"):
        return "gripper"
    return "robot"   # any panda link


def gaze_to_object(env: TeleopEnv, x_norm: float, y_norm: float,
                   aspect: float = 4.0 / 3.0, max_depth: float = 3.0) -> dict:
    """Cast the gaze ray into the CURRENT sim state (call restore_state first).

    Returns {label, body, geom_id, por_x/y/z, depth}; label == "none" when the
    ray leaves the scene (skybox) or only grazes the floor beyond max_depth
    (the workspace is ~1.5 m from the camera).
    """
    if not (np.isfinite(x_norm) and np.isfinite(y_norm)):
        return {"label": "none", "body": "", "geom_id": -1,
                "por_x": np.nan, "por_y": np.nan, "por_z": np.nan,
                "depth": np.nan}
    origin, direction = pixel_to_ray(env, x_norm, y_norm, aspect)
    geomid = np.full(1, -1, dtype=np.int32)
    dist = mujoco.mj_ray(
        env.model, env.data, origin, direction,
        None,                          # all geom groups
        1,                             # include static geoms
        env.model.body("target").id,   # exclude the translucent IK marker
        geomid)
    if dist < 0 or geomid[0] < 0 or dist > max_depth:
        return {"label": "none", "body": "", "geom_id": -1,
                "por_x": np.nan, "por_y": np.nan, "por_z": np.nan,
                "depth": np.nan}
    body_id = env.model.geom_bodyid[geomid[0]]
    body_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
    por = origin + dist * direction
    return {"label": _body_label(env, body_id), "body": body_name,
            "geom_id": int(geomid[0]), "por_x": float(por[0]),
            "por_y": float(por[1]), "por_z": float(por[2]),
            "depth": float(dist)}


def label_gaze_frames(env: TeleopEnv, frames_df, gaze_t: np.ndarray,
                      gaze_xy: np.ndarray, aspect: float = 4.0 / 3.0):
    """Label each gaze sample with the attended object.

    frames_df: frames.parquet as DataFrame (teleop phase rows).
    gaze_t: gaze timestamps on the master clock; gaze_xy: (N, 2) normalized
    display coords (post-calibration). Returns a DataFrame of labels.

    State restoration is only redone when the gaze sample maps to a new
    frame, so cost is ~one mj_forward per displayed frame.
    """
    import pandas as pd

    ft = frames_df["t_master"].to_numpy()
    idx = np.searchsorted(ft, gaze_t, side="right") - 1
    idx = np.clip(idx, 0, len(ft) - 1)

    out = []
    last_frame = -1
    for k in range(len(gaze_t)):
        fi = int(idx[k])
        if fi != last_frame:
            restore_state(env, frames_df.iloc[fi])
            last_frame = fi
        rec = gaze_to_object(env, float(gaze_xy[k, 0]), float(gaze_xy[k, 1]), aspect)
        rec["t_master"] = float(gaze_t[k])
        rec["frame_idx"] = int(frames_df.iloc[fi]["frame_idx"])
        out.append(rec)
    return pd.DataFrame(out)
