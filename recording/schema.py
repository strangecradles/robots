"""Session recording: per-frame parquet log, streamed events jsonl, video.

Session directory layout (one per run):

    data/sessions/<YYYYmmdd_HHMMSS>/
        frames.mp4      every displayed frame, pixel-identical to the display
        frames.parquet  one row per displayed frame (schema below)
        events.jsonl    sync anchors, task events, episode/calib markers
                        (streamed + flushed line-by-line, crash-safe)
        meta.json       config, timing origin, display info

Timebase: all `t_master` values are seconds since session start, measured on
`time.perf_counter()`. `meta.json` stores `t0_unix_ns` (wallclock at t=0) and
`t0_perf` so absolute time can be reconstructed. Device streams (SLO, pupil
cam) are aligned onto this timebase offline via the sync anchors.

frames.parquet columns:
    frame_idx        int     index into frames.mp4
    t_master         float   displayed-frame timestamp (s, master clock)
    phase            str     "calib" | "teleop"
    paused           bool
    mean_lum         float   mean displayed luminance in [0, 1] (pupil regressor;
                             the gaze-window variant is computed offline from
                             frames.mp4 + aligned gaze)
    flash            float   sync-flash intensity rendered into this frame
    cue_label        str     active saccade-cue label or ""
    task_event       str     ";"-joined auto events this tick (grasp/lift/...)
    manual_mark      bool    operator pressed the marker hotkey
    + every key of TeleopEnv.state_row() (qpos_*, qvel_*, ctrl_*, ee_*,
      target_*, cube_*, gripper_width, grasped/lifted/placed, episode,
      sim_time) -- NaN/defaults during the calib phase.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


class SessionRecorder:
    def __init__(self, session_dir: str | Path, width: int, height: int, fps: float):
        self.dir = Path(session_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self._video = cv2.VideoWriter(
            str(self.dir / "frames.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not self._video.isOpened():
            raise RuntimeError("could not open VideoWriter for frames.mp4")
        self._events_f = open(self.dir / "events.jsonl", "a", buffering=1)
        self._rows: list[dict] = []
        self.t0_unix_ns = time.time_ns()
        self.t0_perf = time.perf_counter()

    # ------------------------------------------------------------------ time

    def now(self) -> float:
        """Seconds since session start on the master clock."""
        return time.perf_counter() - self.t0_perf

    # ----------------------------------------------------------------- write

    def add_frame(self, frame_rgb: np.ndarray, *, t_master: float, phase: str,
                  paused: bool, mean_lum: float, flash: float, cue_label: str,
                  task_event: str, manual_mark: bool,
                  env_row: dict | None) -> int:
        """Write one displayed frame + its log row; returns frame_idx."""
        frame_idx = len(self._rows)
        self._video.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        row = {
            "frame_idx": frame_idx,
            "t_master": t_master,
            "phase": phase,
            "paused": paused,
            "mean_lum": mean_lum,
            "flash": flash,
            "cue_label": cue_label,
            "task_event": task_event,
            "manual_mark": manual_mark,
        }
        if env_row:
            row.update(env_row)
        self._rows.append(row)
        return frame_idx

    def add_event(self, event: dict) -> None:
        """Stream one event (must contain t_master + kind) to events.jsonl."""
        self._events_f.write(json.dumps(event) + "\n")

    def add_events(self, events: list[dict]) -> None:
        for e in events:
            self.add_event(e)

    # ----------------------------------------------------------------- close

    def close(self, meta: dict | None = None) -> Path:
        self._video.release()
        self._events_f.close()
        df = pd.DataFrame(self._rows)
        df.to_parquet(self.dir / "frames.parquet", index=False)
        full_meta = {
            "t0_unix_ns": self.t0_unix_ns,
            "t0_perf": self.t0_perf,
            "fps": self.fps,
            "n_frames": len(self._rows),
            **(meta or {}),
        }
        (self.dir / "meta.json").write_text(json.dumps(full_meta, indent=2))
        return self.dir
