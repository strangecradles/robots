"""Master session loop: the single process that owns the clock.

Per tick (fixed rate on time.perf_counter):
    poll keyboard -> step sim -> render offscreen -> sync overlays (flash /
    saccade cue / HUD) drawn INTO the frame -> luminance -> record frame+row
    -> blit to the headset display -> sleep to the next tick.

Usage (laptop dev):
    PYTHONPATH=. .venv/bin/python teleop/run_session.py --windowed
Usage (headset as display 1, with calibration):
    PYTHONPATH=. .venv/bin/python teleop/run_session.py --display 1 --calibrate
Usage (3Dconnexion SpaceMouse Pro Wireless over Bluetooth):
    PYTHONPATH=. .venv/bin/python teleop/run_session.py --device spacemouse

Keys: WASD move | R/F up/down | Q/E yaw | SPACE gripper | N reset episode
      M manual marker | P pause | ESC end session (plays end sync burst)
SpaceMouse: push/twist the puck for 6-DoF EE motion (keyboard stays active as a
      fallback); front-left button toggles the gripper, the next button resets
      the episode (remap via --sm-gripper-button / --sm-reset-button).
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np

from recording.schema import SessionRecorder
from teleop.calibrate import run_calibration_phase
from teleop.display import (KeyboardTeleop, TeleopDisplay, apply_flash,
                            draw_dot, draw_hud, mean_luminance)
from teleop.env import TeleopEnv
from teleop.spacemouse import SpaceMouseTeleop
from teleop.sync import SyncScheduler


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hz", type=float, default=50.0, help="control/display/log rate")
    p.add_argument("--camera", choices=["ego", "side"], default="side",
                   help="display viewpoint: spectator 3/4 view (clear sightline "
                        "to cube+bin, arm never occludes) or egocentric "
                        "over-the-shoulder (natural eye-hand geometry)")
    p.add_argument("--width", type=int, default=960, help="render width")
    p.add_argument("--height", type=int, default=720, help="render height")
    p.add_argument("--display", type=int, default=0, help="pygame display index (headset)")
    p.add_argument("--windowed", action="store_true", help="windowed instead of fullscreen")
    p.add_argument("--calibrate", action="store_true", help="run 9-point gaze calibration first")
    p.add_argument("--device", choices=["keyboard", "spacemouse"], default="keyboard",
                   help="teleop input device (spacemouse keeps keyboard as fallback)")
    p.add_argument("--sm-deadzone", type=float, default=0.1, help="SpaceMouse axis deadzone")
    p.add_argument("--sm-gripper-button", type=int, default=5,
                   help="SpaceMouse button index that toggles the gripper")
    p.add_argument("--sm-reset-button", type=int, default=6,
                   help="SpaceMouse button index that resets the episode (-1 = off)")
    p.add_argument("--sm-invert", type=str, default="",
                   help="comma-separated axes to invert: any of x,y,z,yaw")
    p.add_argument("--duration", type=float, default=0.0, help="auto-end after N seconds (0 = manual)")
    p.add_argument("--sync-period", type=float, default=30.0, help="periodic sync flash interval (s)")
    p.add_argument("--out", type=str, default="data/sessions", help="sessions root dir")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    session_dir = Path(args.out) / datetime.now().strftime("%Y%m%d_%H%M%S")

    cam_name = {"ego": "ego_cam", "side": "teleop_cam"}[args.camera]
    env = TeleopEnv(control_hz=args.hz, camera=cam_name)
    renderer = mujoco.Renderer(env.model, args.height, args.width)
    display = TeleopDisplay(width=args.width, height=args.height,
                            display_index=args.display,
                            fullscreen=not args.windowed)
    kb = KeyboardTeleop()
    if args.device == "spacemouse":
        inv = {a.strip() for a in args.sm_invert.split(",") if a.strip()}
        teleop = SpaceMouseTeleop(
            lin_speed=kb.lin_speed, yaw_speed=kb.yaw_speed,
            deadzone=args.sm_deadzone,
            gripper_button=args.sm_gripper_button,
            reset_button=args.sm_reset_button,
            invert_x="x" in inv, invert_y="y" in inv,
            invert_z="z" in inv, invert_yaw="yaw" in inv,
            keyboard=kb, required=True,
        )
        print("[session] SpaceMouse connected (keyboard active as fallback)")
    else:
        teleop = kb
    recorder = SessionRecorder(session_dir, args.width, args.height, args.hz)
    print(f"[session] recording to {session_dir}")

    if args.calibrate:
        ok = run_calibration_phase(display, recorder, teleop,
                                   args.width, args.height, args.hz)
        if not ok:
            print("[session] calibration aborted; closing")
            recorder.close({"aborted_during_calibration": True})
            display.close()
            return

    sync = SyncScheduler(period_s=args.sync_period)
    sync.start(recorder.now())
    recorder.add_event({"t_master": recorder.now(), "kind": "teleop_start"})

    tick = 1.0 / args.hz
    next_t = time.perf_counter()
    paused = False
    ending = False
    end_at = float("inf")
    teleop_t0 = recorder.now()
    pending_reset_at = float("inf")
    last_event_str = ""
    overruns = 0

    while True:
        t = recorder.now()
        state = teleop.poll()

        if (state.quit or (args.duration > 0 and t - teleop_t0 > args.duration)) and not ending:
            ending = True
            end_at = sync.request_end(t)
            recorder.add_event({"t_master": t, "kind": "end_requested"})
        if t >= end_at:
            break

        if state.toggle_pause:
            paused = not paused
        if state.reset:
            env.reset()
            recorder.add_event({"t_master": t, "kind": "episode_reset",
                                "episode": env.episode_idx})
        if state.mark:
            recorder.add_event({"t_master": t, "kind": "manual_mark"})

        # Rotate view-frame operator intent into the world frame so the
        # controls stay egocentric for whichever camera is displayed.
        fwd_h, right_h = env.view_basis()
        v_view = state.command.lin_vel
        state.command.lin_vel = (right_h * v_view[0] + fwd_h * v_view[1]
                                 + np.array([0.0, 0.0, v_view[2]]))

        task_events: list[str] = []
        if not paused:
            task_events = env.step(state.command)
        for ev in task_events:
            recorder.add_event({"t_master": t, "kind": "task_event", "label": ev,
                                "episode": env.episode_idx})
            if ev == "place_success":
                pending_reset_at = t + 1.5
        if task_events:
            last_event_str = task_events[-1]

        if t >= pending_reset_at:
            pending_reset_at = float("inf")
            env.reset()
            recorder.add_event({"t_master": recorder.now(), "kind": "episode_reset",
                                "episode": env.episode_idx})

        # Render and overlay (overlays must precede luminance/record/blit).
        renderer.update_scene(env.data, camera=cam_name)
        frame = renderer.render()
        flash, cue = sync.update(t)
        if flash > 0.0:
            apply_flash(frame, flash)
        if cue is not None:
            draw_dot(frame, cue.x_norm, cue.y_norm)
        hud = [
            f"t {t - teleop_t0:6.1f}s  ep {env.episode_idx}",
            f"gripper {'CLOSED' if teleop.gripper_closed else 'open'}"
            + ("  PAUSED" if paused else ""),
        ]
        if cue is not None:
            hud.append(f"look {cue.label.upper()}")
        if ending:
            hud.append("ending - sync burst")
        if last_event_str:
            hud.append(f"last: {last_event_str}")
        draw_hud(frame, hud)

        lum = mean_luminance(frame)
        recorder.add_frame(
            frame, t_master=t, phase="teleop", paused=paused, mean_lum=lum,
            flash=flash, cue_label=(cue.label if cue else ""),
            task_event=";".join(task_events), manual_mark=state.mark,
            env_row=env.state_row())
        recorder.add_events(sync.drain_events())
        display.show(frame)

        next_t += tick
        sleep_for = next_t - time.perf_counter()
        if sleep_for > 0:
            time.sleep(sleep_for)
        else:
            overruns += 1
            next_t = time.perf_counter()  # don't spiral after a long stall

    if isinstance(teleop, SpaceMouseTeleop):
        teleop.close()

    out = recorder.close({
        "args": vars(args),
        "camera": cam_name,
        "device": args.device,
        "scene_xml": env.xml_path,
        "control_hz": args.hz,
        "render_size": [args.width, args.height],
        "display_size": list(display.size),
        "tick_overruns": overruns,
        "episodes": env.episode_idx + 1,
    })
    display.close()
    print(f"[session] done -> {out}  (overruns: {overruns})")


if __name__ == "__main__":
    main()
