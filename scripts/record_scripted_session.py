"""Record a full session driven by a scripted policy (no human, no display).

Produces exactly the same session artifacts as teleop/run_session.py
(frames.mp4 / frames.parquet / events.jsonl / meta.json), on a synthetic
master clock (t = frame/hz), so the offline pipeline can be developed and
validated end-to-end tonight. Includes a calibration phase, the sync
anchors, and three episodes: clean pick-place, pick with mid-air release
over the floor (drop-like), clean pick-place.

Usage: PYTHONPATH=. .venv/bin/python scripts/record_scripted_session.py
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np

from recording.schema import SessionRecorder
from teleop.calibrate import GRAY, calibration_points
from teleop.display import apply_flash, draw_dot, draw_hud, mean_luminance
from teleop.env import BIN_CENTER, TeleopCommand, TeleopEnv
from teleop.sync import SyncScheduler


class ScriptedPolicy:
    """Waypoint follower emitting TeleopCommands; one pick-place episode."""

    def __init__(self, env: TeleopEnv, drop_midway: bool):
        cube = env.cube_pos()
        bx, by = BIN_CENTER
        self.legs = [
            # (goal, gripper_closed, settle_ticks)
            (cube + [0, 0, 0.12], False, 0),
            (cube + [0, 0, 0.005], False, 10),
            (cube + [0, 0, 0.005], True, 25),
            (cube + [0, 0, 0.20], True, 0),
            ([bx, by, 0.20], True, 0),
        ]
        if drop_midway:
            mid = [(cube[0] + bx) / 2, (cube[1] + by) / 2, 0.20]
            self.legs = self.legs[:4] + [
                (mid, True, 0),
                (mid, False, 30),          # let go mid-air over the floor
                (mid + np.array([0, 0, 0.05]), False, 20),
            ]
        else:
            self.legs += [
                ([bx, by, 0.10], True, 10),
                ([bx, by, 0.10], False, 30),
                ([bx, by, 0.25], False, 10),
            ]
        self.leg = 0
        self.settle = 0

    def done(self) -> bool:
        return self.leg >= len(self.legs)

    def command(self, env: TeleopEnv) -> TeleopCommand:
        if self.done():
            return TeleopCommand()
        goal, close, settle_ticks = self.legs[self.leg]
        err = np.asarray(goal) - env.target_pos
        if np.linalg.norm(err) < 0.012:
            self.settle += 1
            if self.settle >= settle_ticks:
                self.leg += 1
                self.settle = 0
        v = np.clip(err * 3.0, -0.25, 0.25)
        return TeleopCommand(lin_vel=v, gripper_close=close)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hz", type=float, default=50.0)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--out", type=str, default="data/sessions")
    p.add_argument("--sync-period", type=float, default=20.0)
    args = p.parse_args()

    session_dir = Path(args.out) / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_scripted")
    env = TeleopEnv(control_hz=args.hz)
    renderer = mujoco.Renderer(env.model, args.height, args.width)
    recorder = SessionRecorder(session_dir, args.width, args.height, args.hz)
    tick = 1.0 / args.hz
    frame = [0]  # synthetic master clock: t = frame/hz

    def now() -> float:
        return frame[0] * tick

    # ----------------------------------------------------- calibration phase
    for idx, (x, y) in enumerate(calibration_points(9)):
        for stage, dur in (("gap", 0.5), ("hold", 1.6)):
            n_ticks = int(dur * args.hz)
            for k in range(n_ticks):
                t = now()
                img = np.full((args.height, args.width, 3), GRAY, dtype=np.uint8)
                if stage == "hold":
                    draw_dot(img, x, y)
                    if k == 0:
                        recorder.add_event({"t_master": t, "kind": "calib_point_on",
                                            "point_idx": idx, "x_norm": x,
                                            "y_norm": y, "hold_s": dur})
                draw_hud(img, [f"calibration {idx + 1}/9"])
                recorder.add_frame(img, t_master=t, phase="calib", paused=False,
                                   mean_lum=mean_luminance(img), flash=0.0,
                                   cue_label="", task_event="", manual_mark=False,
                                   env_row=None)
                frame[0] += 1
            if stage == "hold":
                recorder.add_event({"t_master": now(), "kind": "calib_point_off",
                                    "point_idx": idx})

    # ---------------------------------------------------------- teleop phase
    sync = SyncScheduler(period_s=args.sync_period)
    sync.start(now())
    recorder.add_event({"t_master": now(), "kind": "teleop_start"})

    episodes = [False, True, False]   # episode 1 releases mid-air
    ep = 0
    policy = ScriptedPolicy(env, episodes[ep])
    idle_until = 0.0
    ending = False
    end_at = float("inf")

    while True:
        t = now()
        if ending and t >= end_at:
            break

        cmd = TeleopCommand() if (policy.done() or t < idle_until) else policy.command(env)
        task_events = env.step(cmd)
        for ev in task_events:
            recorder.add_event({"t_master": t, "kind": "task_event", "label": ev,
                                "episode": env.episode_idx})

        if policy.done() and t >= idle_until and not ending:
            ep += 1
            if ep >= len(episodes):
                ending = True
                end_at = sync.request_end(t)
                recorder.add_event({"t_master": t, "kind": "end_requested"})
            else:
                env.reset(randomize=True)
                recorder.add_event({"t_master": t, "kind": "episode_reset",
                                    "episode": env.episode_idx})
                policy = ScriptedPolicy(env, episodes[ep])
                idle_until = t + 1.0   # brief idle between episodes

        renderer.update_scene(env.data, camera="teleop_cam")
        img = renderer.render()
        flash, cue = sync.update(t)
        if flash > 0.0:
            apply_flash(img, flash)
        if cue is not None:
            draw_dot(img, cue.x_norm, cue.y_norm)
        draw_hud(img, [f"t {t:6.1f}s ep {env.episode_idx} (scripted)"])

        recorder.add_frame(img, t_master=t, phase="teleop", paused=False,
                           mean_lum=mean_luminance(img), flash=flash,
                           cue_label=(cue.label if cue else ""),
                           task_event=";".join(task_events), manual_mark=False,
                           env_row=env.state_row())
        recorder.add_events(sync.drain_events())
        frame[0] += 1

    out = recorder.close({"scripted": True, "control_hz": args.hz,
                          "render_size": [args.width, args.height],
                          "episodes": env.episode_idx + 1})
    print(f"[scripted] done -> {out}")


if __name__ == "__main__":
    main()
