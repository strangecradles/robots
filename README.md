# Gaze-Attention Teleoperation Rig

Capture a continuous **temporal attention signal** `a(t)` -- *how deeply* a
human teleoperator is attending moment-to-moment -- from headset
**SLO fixational dynamics** (microsaccades, saccadic vigor, fixation
stability) and **luminance-corrected pupil**, time-synchronized to a MuJoCo
teleoperation video, and inject it into robot learning.

The novel claim is the **temporal** channel. Spatial gaze ("where") is
commodity (any VR headset); the contribution here is the SLO-derived scalar
for attentional depth, co-registered to the task and the robot's actions.

> Day-1 deliverable is **not** a trained robot. It is a clean, time-aligned
> multimodal dataset plus a plot showing `a(t)` tracks task events
> (grasp / lift / place / drop) vs ballistic transit -- i.e. construct
> validity of `a(t)` before any injection.

## Pipeline

```
teleop (live)                          offline
-----------------------------------    ----------------------------------------
keyboard -> MuJoCo Franka -> render     ingest SLO + pupil exports (adapters)
   -> sync overlays (flash + saccade       -> align device clocks to master
      cue) into frame -> luminance            (sync-flash anchors, linear fit)
   -> record frame + state row          -> fit gaze calibration (calib phase)
   -> blit to headset display           -> features: msacc inhibition, vigor,
                                            fixation stability, pupil TEPR
device cams record independently        -> fuse -> a(t)
(SLO, pupil; own clocks, export later)  -> raycast gaze -> attended object
                                        -> validity report (event-locked a(t))
```

See `data/sessions/<id>/derived/validity.png` for the deliverable plot.

## Setup

```bash
uv venv --python 3.13
uv sync                     # core deps
uv sync --extra policy      # + torch, for Phase C behavioral cloning
uv sync --extra spacemouse  # + pyspacemouse/hidapi, for SpaceMouse teleop
```

The SpaceMouse path reads the device over **raw HID** (`pyspacemouse`), so on
macOS it also needs the native lib (`brew install hidapi`) and the host app
(Terminal/Cursor) granted **Input Monitoring**. It does *not* use 3Dconnexion's
official driver -- if that driver is installed its daemon seizes the device, so
quit it (`pkill -f 3Dconnexion`) or uninstall it.

The Franka Panda model is vendored under `assets/franka_emika_panda/`
(from [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie),
Apache-2.0) with two local additions: an `ee_site` tool-center point and
`scene_teleop.xml` (cube + bin + operator camera + IK target).

## Run it tonight (no hardware)

End-to-end on synthetic device data, which exercises every offline stage:

```bash
# 1. record a scripted session (calibration + 3 pick-place episodes)
PYTHONPATH=. .venv/bin/python scripts/record_scripted_session.py

# 2. fabricate SLO + pupil exports with embedded ground-truth attention
PYTHONPATH=. .venv/bin/python ingest/synthetic.py data/sessions/<id>

# 3. align -> features -> a(t) -> validity plot
PYTHONPATH=. .venv/bin/python analysis/pipeline.py data/sessions/<id>
```

On the synthetic data the extracted `a(t)` recovers the generator's hidden
attention signal (teleop-phase correlation r ~ 0.68) and rises into grasp
events, confirming the math before real exports exist.

## Run it at the rig (with the headset)

```bash
# headset as macOS display index 1; 9-point gaze calibration first
PYTHONPATH=. .venv/bin/python teleop/run_session.py --display 1 --calibrate

# drive the arm with a 3Dconnexion SpaceMouse (keyboard stays active too)
PYTHONPATH=. .venv/bin/python teleop/run_session.py --device spacemouse
```

Keys: `WASD` move (x/y), `R/F` up/down, `Q/E` yaw, `SPACE` gripper,
`N` reset episode, `M` manual marker, `P` pause, `ESC` end (plays the end
sync burst -- **let it finish** so the clocks can be drift-fit).

**SpaceMouse** (`--device spacemouse`): push/twist the puck for 6-DoF
end-effector motion in the same egocentric view frame as the keys; buttons
toggle gripper / reset episode (`--sm-gripper-button` / `--sm-reset-button`,
`--sm-invert x,y,z,yaw` to flip an axis). The keyboard remains live as a
fallback. Run `PYTHONPATH=. .venv/bin/python teleop/spacemouse.py` to live-test
axes and discover button indices.

The display viewpoint defaults to an elevated 3/4 spectator view (`--camera
side`) with a clear sightline to the cube and bin throughout the task; pass
`--camera ego` for the over-the-shoulder egocentric view used for gaze geometry.

Then export the SLO and pupil recordings, point the adapters at them, and run
the same offline pipeline:

```bash
PYTHONPATH=. .venv/bin/python analysis/pipeline.py data/sessions/<id> \
    --pupil /path/to/pupil_export --slo /path/to/slo_export
```

### Plugging in the real device formats

Both devices are **record-only**; their export formats are unknown until the
rig. Ingestion is isolated behind adapters:

- `ingest/pupil_adapter.py::load_pupil(path, colmap=...)` -- table of gaze +
  diameter + an illumination channel (for flash anchors).
- `ingest/slo_adapter.py::load_slo_trace(path, colmap=...)` -- an eye-motion
  trace; or `trace_from_video(path)` if you only get raw retinal video.

Required channels and the synchronization contract are documented in
`ingest/base.py`. Nothing downstream touches a device clock -- `align/align.py`
maps every stream onto the master clock first.

## Layout

| Path | Role |
|------|------|
| `teleop/env.py` | Franka + cube + bin, diff-IK from a 6-DoF target, auto event detection |
| `teleop/display.py` | headset display output, keyboard teleop, luminance |
| `teleop/spacemouse.py` | 3Dconnexion SpaceMouse 6-DoF teleop (raw HID, keyboard fallback) |
| `teleop/sync.py` | photic-flash + saccade-cue anchor scheduler |
| `teleop/calibrate.py` | 9-point gaze calibration (display + offline fit) |
| `teleop/run_session.py` | **master loop** (owns the clock) |
| `recording/schema.py` | session recorder (mp4 + parquet + jsonl + meta) |
| `ingest/` | device adapters (`base`, `pupil_adapter`, `slo_adapter`) + `synthetic` |
| `align/align.py` | anchor detection, clock fit, resampling |
| `attention/features.py` | msacc inhibition, vigor, fixation stability, pupil TEPR |
| `attention/a_of_t.py` | fuse channels -> `a(t)` |
| `analysis/raycast.py` | gaze pixel -> 3D point-of-regard -> attended object |
| `analysis/validity.py` | event-locked `a(t)`, transit baseline, truth corr |
| `analysis/pipeline.py` | offline driver tying the above together |
| `policy/attention_bc.py` | Phase C: attention-weighted behavioral cloning |
| `scripts/record_scripted_session.py` | hardware-free session generator |

## Phasing

- **A (tonight, laptop):** teleop + display + sync + recording + calibration,
  validated end-to-end against the synthetic generator.
- **B (at the rig):** swap in real pupil + SLO exports; run sessions; produce
  the `a(t)`-vs-events validity plot. **This is the result to get first.**
- **C (only if validity holds):** attention-weighted behavioral cloning
  (`policy/attention_bc.py`, with a `--uniform` ablation). RL reward shaping
  is deliberately deferred -- `a(t)` is an attention/intention prior, not a
  value signal, so it belongs in the imitation objective, not a reward.

## Caveats baked into the design

- **Pupil is display-light-dominated.** `mean_lum` is logged per displayed
  frame and regressed out (PLR kernel + lag) before the task-evoked response
  is read. Treat pupil as a weak corroborating channel.
- **SLO has a tiny field of view.** It is used only for fixational/temporal
  features, never scene-level trajectory (that is the pupil-cam gaze).
- **Constant device latency** (sensor exposure -> file timestamp) is absorbed
  into the clock-fit offset and is unobservable without a hardware probe; it
  does not affect relative timing, which is what `a(t)` needs.
