"""Franka Panda pick-and-place teleop environment.

The operator commands a 6-DoF end-effector target (position + yaw, with the
gripper held top-down) which is tracked by damped-least-squares differential IK
feeding the Panda's stock position actuators. A visual-only mocap body marks
the target. Task events (grasp / lift / place / drop) are auto-detected from
contacts and cube kinematics so the analysis stage gets event labels that do
not depend on the operator remembering to press annotation keys.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import mujoco
import numpy as np

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
SCENE_XML = ASSETS_DIR / "franka_emika_panda" / "scene_teleop.xml"

# Workspace clamp for the commanded target (meters, world frame).
WORKSPACE_LO = np.array([0.20, -0.35, 0.012])
WORKSPACE_HI = np.array([0.70, 0.45, 0.60])

CUBE_HALF = 0.02
LIFT_Z = 0.07          # cube counts as lifted above this height
BIN_CENTER = np.array([0.45, 0.28])
BIN_HALF = 0.06        # inner half-extent of the bin base


@dataclasses.dataclass
class TeleopCommand:
    """One control tick of operator intent (already scaled to velocities)."""

    lin_vel: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(3))  # m/s, world
    yaw_rate: float = 0.0       # rad/s about world z
    gripper_close: bool = False  # True = close, False = open


class TeleopEnv:
    def __init__(self, xml_path: str | Path = SCENE_XML, control_hz: float = 50.0):
        self.xml_path = str(xml_path)
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.control_hz = control_hz
        self.n_substeps = max(1, round(1.0 / control_hz / self.model.opt.timestep))

        self.ee_site = self.model.site("ee_site").id
        self.cam_id = self.model.camera("teleop_cam").id
        self.cube_body = self.model.body("cube").id
        self.cube_geom = self.model.geom("cube_geom").id
        self.cube_jnt_qposadr = self.model.joint("cube_freejoint").qposadr[0]
        self.left_finger_body = self.model.body("left_finger").id
        self.right_finger_body = self.model.body("right_finger").id
        self.target_mocap = self.model.body("target").mocapid[0]
        self.home_key = self.model.key("home").id

        self.arm_dofs = 7
        self.home_qpos = self.model.key_qpos[self.home_key, : self.arm_dofs].copy()
        # Joint limits for the 7 arm joints (they are the first 7 actuators).
        self.arm_ctrl_range = self.model.actuator_ctrlrange[: self.arm_dofs].copy()

        # Commanded target state (integrated from operator velocities).
        self.target_pos = np.zeros(3)
        self.target_yaw = 0.0
        self.home_ee_quat = np.zeros(4)

        self.episode_idx = -1
        self._rng = np.random.default_rng()
        self._grasped = False
        self._lifted = False
        self._placed = False
        self._contact_frames = 0
        self._no_contact_frames = 0

        self.reset(randomize=False)

    # ------------------------------------------------------------------ reset

    def reset(self, randomize: bool = True) -> None:
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_key)
        if randomize:
            adr = self.cube_jnt_qposadr
            self.data.qpos[adr + 0] = 0.55 + self._rng.uniform(-0.06, 0.06)
            self.data.qpos[adr + 1] = -0.10 + self._rng.uniform(-0.08, 0.08)
        mujoco.mj_forward(self.model, self.data)

        # Target starts at the current end-effector pose, top-down orientation.
        self.target_pos = self.data.site_xpos[self.ee_site].copy()
        self.target_yaw = 0.0
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, self.data.site_xmat[self.ee_site])
        self.home_ee_quat = quat

        self.episode_idx += 1
        self._grasped = False
        self._lifted = False
        self._placed = False
        self._contact_frames = 0
        self._no_contact_frames = 0
        self._sync_target_marker()

    # ------------------------------------------------------------------- step

    def step(self, cmd: TeleopCommand) -> list[str]:
        """Advance one control period; returns auto-detected task events."""
        dt = 1.0 / self.control_hz
        self.target_pos = np.clip(self.target_pos + cmd.lin_vel * dt, WORKSPACE_LO, WORKSPACE_HI)
        self.target_yaw = float(np.clip(self.target_yaw + cmd.yaw_rate * dt, -2.8, 2.8))
        self._sync_target_marker()

        self._set_arm_ctrl()
        self.data.ctrl[7] = 0.0 if cmd.gripper_close else 255.0

        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        return self._detect_events(cmd)

    def _target_quat(self) -> np.ndarray:
        yaw_quat = np.array([np.cos(self.target_yaw / 2), 0.0, 0.0, np.sin(self.target_yaw / 2)])
        out = np.zeros(4)
        mujoco.mju_mulQuat(out, yaw_quat, self.home_ee_quat)
        return out

    def _sync_target_marker(self) -> None:
        self.data.mocap_pos[self.target_mocap] = self.target_pos
        self.data.mocap_quat[self.target_mocap] = self._target_quat()

    def _set_arm_ctrl(self) -> None:
        """Damped-least-squares diff-IK from current EE pose to the target."""
        nv = self.model.nv
        jacp = np.zeros((3, nv))
        jacr = np.zeros((3, nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site)
        jac = np.vstack([jacp[:, : self.arm_dofs], jacr[:, : self.arm_dofs]])

        pos_err = self.target_pos - self.data.site_xpos[self.ee_site]

        site_quat = np.zeros(4)
        mujoco.mju_mat2Quat(site_quat, self.data.site_xmat[self.ee_site])
        site_quat_inv = np.zeros(4)
        mujoco.mju_negQuat(site_quat_inv, site_quat)
        err_quat = np.zeros(4)
        mujoco.mju_mulQuat(err_quat, self._target_quat(), site_quat_inv)
        rot_err = np.zeros(3)
        mujoco.mju_quat2Vel(rot_err, err_quat, 1.0)

        err = np.concatenate([pos_err, 0.5 * rot_err])

        damping = 1e-4
        jjt = jac @ jac.T + damping * np.eye(6)
        dq = jac.T @ np.linalg.solve(jjt, err)

        # Mild nullspace pull toward the home posture to avoid drift.
        qpos_arm = self.data.qpos[: self.arm_dofs]
        jpinv = jac.T @ np.linalg.inv(jjt)
        null_proj = np.eye(self.arm_dofs) - jpinv @ jac
        dq += null_proj @ (0.05 * (self.home_qpos - qpos_arm))

        max_step = 0.05  # rad per control tick
        scale = max_step / max(np.max(np.abs(dq)), max_step)
        dq *= scale

        ctrl = np.clip(qpos_arm + dq, self.arm_ctrl_range[:, 0], self.arm_ctrl_range[:, 1])
        self.data.ctrl[: self.arm_dofs] = ctrl

    # ------------------------------------------------------------- task state

    def _fingers_on_cube(self) -> tuple[bool, bool]:
        left = right = False
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            g1, g2 = con.geom1, con.geom2
            if self.cube_geom not in (g1, g2):
                continue
            other = g2 if g1 == self.cube_geom else g1
            body = self.model.geom_bodyid[other]
            if body == self.left_finger_body:
                left = True
            elif body == self.right_finger_body:
                right = True
        return left, right

    def cube_pos(self) -> np.ndarray:
        return self.data.xpos[self.cube_body].copy()

    def cube_quat(self) -> np.ndarray:
        return self.data.xquat[self.cube_body].copy()

    def cube_in_bin(self) -> bool:
        p = self.cube_pos()
        return (
            abs(p[0] - BIN_CENTER[0]) < BIN_HALF
            and abs(p[1] - BIN_CENTER[1]) < BIN_HALF
            and p[2] < 0.06
        )

    def _detect_events(self, cmd: TeleopCommand) -> list[str]:
        events: list[str] = []
        left, right = self._fingers_on_cube()
        both = left and right
        cube_z = self.cube_pos()[2]

        if both and cmd.gripper_close:
            self._contact_frames += 1
            self._no_contact_frames = 0
        else:
            self._no_contact_frames += 1
            self._contact_frames = 0

        if not self._grasped and self._contact_frames >= 3:
            self._grasped = True
            events.append("grasp")
        elif self._grasped:
            # Commanded open -> intentional release (1 frame is enough).
            # Still commanded closed but contact gone for a while -> slip/drop.
            # The long debounce absorbs transient contact flicker during transit.
            if not cmd.gripper_close:
                self._grasped = False
                events.append("release")
                self._lifted = False
            elif self._no_contact_frames >= 10:
                self._grasped = False
                events.append("drop")
                self._lifted = False

        if self._grasped and not self._lifted and cube_z > LIFT_Z:
            self._lifted = True
            events.append("lift")

        if not self._placed and not self._grasped and self.cube_in_bin():
            speed = float(np.linalg.norm(self.data.qvel[self.cube_jnt_qposadr : self.cube_jnt_qposadr + 3]))
            if speed < 0.05:
                self._placed = True
                events.append("place_success")

        return events

    # --------------------------------------------------------------- logging

    def state_row(self) -> dict:
        """Loggable snapshot of the sim, flat scalars only."""
        d = self.data
        row: dict = {
            "sim_time": float(d.time),
            "episode": self.episode_idx,
            "gripper_width": float(d.qpos[7] + d.qpos[8]),
            "grasped": self._grasped,
            "lifted": self._lifted,
            "placed": self._placed,
        }
        for i in range(9):
            row[f"qpos_{i}"] = float(d.qpos[i])
            row[f"qvel_{i}"] = float(d.qvel[i])
        for i in range(8):
            row[f"ctrl_{i}"] = float(d.ctrl[i])
        ee = d.site_xpos[self.ee_site]
        ee_quat = np.zeros(4)
        mujoco.mju_mat2Quat(ee_quat, d.site_xmat[self.ee_site])
        for i, k in enumerate("xyz"):
            row[f"ee_{k}"] = float(ee[i])
            row[f"target_{k}"] = float(self.target_pos[i])
        for i, k in enumerate("wxyz"):
            row[f"ee_quat_{k}"] = float(ee_quat[i])
        row["target_yaw"] = self.target_yaw
        cube = self.cube_pos()
        cube_q = self.cube_quat()
        for i, k in enumerate("xyz"):
            row[f"cube_{k}"] = float(cube[i])
        for i, k in enumerate("wxyz"):
            row[f"cube_quat_{k}"] = float(cube_q[i])
        return row
