"""Headset display output + keyboard capture.

Pipeline invariant: every overlay (sync flash, saccade cue, calibration dot,
HUD) is drawn into the numpy RGB frame BEFORE (a) luminance computation,
(b) video recording, and (c) blitting. The recorded video is therefore
pixel-identical to what the operator saw, and the logged luminance is the
luminance of the displayed pixels -- which is what drives the pupil.
"""

from __future__ import annotations

import dataclasses

import cv2
import numpy as np
import pygame

from teleop.env import TeleopCommand

# ----------------------------------------------------------------- overlays


def apply_flash(frame: np.ndarray, intensity: float) -> None:
    """Blend the frame toward full white in-place. intensity in [0, 1]."""
    if intensity <= 0.0:
        return
    white = np.full_like(frame, 255)
    cv2.addWeighted(frame, 1.0 - intensity, white, intensity, 0.0, dst=frame)


def draw_dot(frame: np.ndarray, x_norm: float, y_norm: float,
             radius_px: int = 14, color: tuple = (255, 40, 40)) -> None:
    """Draw a high-contrast fixation dot at normalized display coords."""
    h, w = frame.shape[:2]
    c = (int(x_norm * w), int(y_norm * h))
    cv2.circle(frame, c, radius_px + 4, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, c, radius_px, color, -1, cv2.LINE_AA)
    cv2.circle(frame, c, 3, (0, 0, 0), -1, cv2.LINE_AA)


def draw_hud(frame: np.ndarray, lines: list[str]) -> None:
    y = 22
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (240, 240, 240), 1, cv2.LINE_AA)
        y += 24


def mean_luminance(frame: np.ndarray) -> float:
    """Mean relative luminance (Rec.601) of the displayed frame, in [0, 1].

    Downsampled 4x for speed. The gaze-window luminance variant is computed
    OFFLINE (align stage) from the recorded video + aligned gaze, since gaze
    is not available live (pupil cam is record-only).
    """
    small = frame[::4, ::4].astype(np.float32)
    lum = 0.299 * small[..., 0] + 0.587 * small[..., 1] + 0.114 * small[..., 2]
    return float(lum.mean() / 255.0)


# ------------------------------------------------------------------ display


class TeleopDisplay:
    """Fullscreen (or windowed) pygame surface, targeted at a display index.

    Put the headset display's index in `display_index` (macOS lists it as a
    second display). Windowed mode is for development on the laptop panel.
    """

    def __init__(self, width: int = 1280, height: int = 960,
                 display_index: int = 0, fullscreen: bool = True):
        pygame.init()
        pygame.display.set_caption("gaze-attention teleop")
        if fullscreen:
            # (0, 0) = native resolution of the target display.
            self.screen = pygame.display.set_mode(
                (0, 0), pygame.FULLSCREEN, display=display_index)
        else:
            self.screen = pygame.display.set_mode(
                (width, height), 0, display=display_index)
        self.size = self.screen.get_size()

    def show(self, frame: np.ndarray) -> None:
        """Blit an RGB (H, W, 3) uint8 frame, scaling to the display size."""
        # pygame surfarray wants (W, H, 3)
        surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        if surf.get_size() != self.size:
            surf = pygame.transform.scale(surf, self.size)
        self.screen.blit(surf, (0, 0))
        pygame.display.flip()

    def close(self) -> None:
        pygame.quit()


# ----------------------------------------------------------------- keyboard


@dataclasses.dataclass
class InputState:
    command: TeleopCommand
    quit: bool = False
    reset: bool = False
    mark: bool = False          # manual "something interesting" annotation
    toggle_pause: bool = False


class KeyboardTeleop:
    """Maps held keys to continuous EE velocities and edge keys to events.

    The emitted lin_vel is in VIEW coordinates: x = "to my right",
    y = "away from me", z = up. The session loop rotates it into the world
    frame via TeleopEnv.view_basis(), so the controls feel egocentric for
    any display camera.

    Held:  W/S = away/toward | A/D = left/right | R/F = up/down
           Q/E = yaw ccw/cw  | LSHIFT = 2x speed
    Edge:  SPACE = toggle gripper | N = reset episode | M = manual marker
           P = pause physics      | ESC = quit
    """

    def __init__(self, lin_speed: float = 0.18, yaw_speed: float = 1.2):
        self.lin_speed = lin_speed
        self.yaw_speed = yaw_speed
        self.gripper_closed = False

    def poll(self) -> InputState:
        state = InputState(command=TeleopCommand())
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                state.quit = True
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    state.quit = True
                elif ev.key == pygame.K_SPACE:
                    self.gripper_closed = not self.gripper_closed
                elif ev.key == pygame.K_n:
                    state.reset = True
                elif ev.key == pygame.K_m:
                    state.mark = True
                elif ev.key == pygame.K_p:
                    state.toggle_pause = True

        keys = pygame.key.get_pressed()
        speed = self.lin_speed * (2.0 if keys[pygame.K_LSHIFT] else 1.0)
        v = np.zeros(3)
        if keys[pygame.K_w]:
            v[1] += speed
        if keys[pygame.K_s]:
            v[1] -= speed
        if keys[pygame.K_d]:
            v[0] += speed
        if keys[pygame.K_a]:
            v[0] -= speed
        if keys[pygame.K_r]:
            v[2] += speed
        if keys[pygame.K_f]:
            v[2] -= speed
        yaw = 0.0
        if keys[pygame.K_q]:
            yaw += self.yaw_speed
        if keys[pygame.K_e]:
            yaw -= self.yaw_speed

        state.command = TeleopCommand(
            lin_vel=v, yaw_rate=yaw, gripper_close=self.gripper_closed)
        return state
