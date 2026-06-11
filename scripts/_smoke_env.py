"""Scripted open-loop pick to verify IK tracking + event detection."""
import numpy as np
from teleop.env import TeleopEnv, TeleopCommand

env = TeleopEnv()
env.reset(randomize=False)

def goto(env, goal, close, n, yaw=0.0):
    evs = []
    for _ in range(n):
        err = goal - env.target_pos
        v = np.clip(err * 3.0, -0.25, 0.25)
        evs += env.step(TeleopCommand(lin_vel=v, gripper_close=close))
    return evs

cube = env.cube_pos()
print("cube", cube.round(3), "ee start", env.data.site_xpos[env.ee_site].round(3))

all_events = []
all_events += goto(env, cube + [0, 0, 0.12], close=False, n=60)   # above cube
all_events += goto(env, cube + [0, 0, 0.005], close=False, n=60)  # descend
all_events += goto(env, cube + [0, 0, 0.005], close=True, n=25)   # grasp
all_events += goto(env, cube + [0, 0, 0.20], close=True, n=70)    # lift
bin_xy = [0.45, 0.28]
all_events += goto(env, [bin_xy[0], bin_xy[1], 0.20], close=True, n=90)  # transit
all_events += goto(env, [bin_xy[0], bin_xy[1], 0.10], close=True, n=50)  # descend bin
all_events += goto(env, [bin_xy[0], bin_xy[1], 0.10], close=False, n=40) # release
all_events += goto(env, [bin_xy[0], bin_xy[1], 0.25], close=False, n=40) # retreat

print("events:", all_events)
print("final cube", env.cube_pos().round(3), "in_bin", env.cube_in_bin())
print("ee tracking err",
      np.linalg.norm(env.target_pos - env.data.site_xpos[env.ee_site]).round(4))
