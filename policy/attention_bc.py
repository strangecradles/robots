"""Attention-weighted behavioral cloning (Phase C).

The injection: scale the per-timestep BC loss by the operator's attention
a(t), so the policy learns hardest from the moments the human was deeply
attending (alignment, contact, recovery) and discounts ballistic transit.
Run with --uniform for the ablation baseline (same data, weight = 1).

Requires the optional torch dependency:  uv sync --extra policy

    PYTHONPATH=. .venv/bin/python policy/attention_bc.py \
        data/sessions/<session> [more sessions ...] --epochs 50

Observations (sim-state v0; swap in frames.mp4 pixels later):
    qpos[0:9], ee xyz, ee quat, gripper width, cube xyz, cube quat -> 24 dims
Actions:
    commanded EE-target velocity xyz (m/s), target yaw rate, gripper close
Weights:
    w(t) = (a(t) ** gamma), renormalized to mean 1 over the dataset;
    samples with a_conf < 0.3 fall back to weight 1.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

OBS_COLS = ([f"qpos_{i}" for i in range(9)]
            + ["ee_x", "ee_y", "ee_z"]
            + [f"ee_quat_{k}" for k in "wxyz"]
            + ["gripper_width"]
            + ["cube_x", "cube_y", "cube_z"]
            + [f"cube_quat_{k}" for k in "wxyz"])


def build_bc_dataset(session_dirs: list[str | Path], gamma: float = 1.0,
                     uniform: bool = False
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (obs, act, weight) arrays pooled over sessions."""
    obs_l, act_l, w_l = [], [], []
    for sd in session_dirs:
        sd = Path(sd)
        frames = pd.read_parquet(sd / "frames.parquet")
        frames = frames[(frames["phase"] == "teleop")
                        & (~frames["paused"])].reset_index(drop=True)
        attn_path = sd / "derived" / "attention.parquet"
        if attn_path.exists():
            attn = pd.read_parquet(attn_path)[["t_master", "a", "a_conf"]]
            frames = pd.merge_asof(frames, attn, on="t_master",
                                   direction="nearest",
                                   tolerance=0.05)
        else:
            frames["a"] = np.nan
            frames["a_conf"] = 0.0

        # Action: commanded target velocity (finite difference of the
        # operator's integrated target), yaw rate, and gripper command.
        dt = np.diff(frames["t_master"].to_numpy())
        ok = dt > 1e-6
        tgt = frames[["target_x", "target_y", "target_z"]].to_numpy()
        vel = np.diff(tgt, axis=0) / dt[:, None]
        yaw_rate = np.diff(frames["target_yaw"].to_numpy()) / dt
        grip = (frames["ctrl_7"].to_numpy()[:-1] < 128).astype(float)  # closed=1

        # Episode boundaries: don't difference across resets.
        ep = frames["episode"].to_numpy()
        same_ep = ep[1:] == ep[:-1]
        keep = ok & same_ep

        obs = frames[OBS_COLS].to_numpy(dtype=np.float32)[:-1][keep]
        act = np.concatenate([vel, yaw_rate[:, None], grip[:, None]],
                             axis=1).astype(np.float32)[keep]

        a = frames["a"].to_numpy(dtype=float)[:-1][keep]
        conf = frames["a_conf"].to_numpy(dtype=float)[:-1][keep]
        w = np.power(np.nan_to_num(a, nan=0.5), gamma)
        w[conf < 0.3] = np.nan  # filled with mean weight below
        if uniform:
            w = np.ones_like(w)

        obs_l.append(obs)
        act_l.append(act)
        w_l.append(w)

    obs = np.concatenate(obs_l)
    act = np.concatenate(act_l)
    w = np.concatenate(w_l)
    w = np.where(np.isfinite(w), w, np.nanmean(w))
    w = w / max(w.mean(), 1e-9)  # mean-1 so the lr is comparable to uniform
    return obs, act, w.astype(np.float32)


def train(obs: np.ndarray, act: np.ndarray, w: np.ndarray,
          epochs: int = 50, batch: int = 256, lr: float = 3e-4,
          val_frac: float = 0.1, seed: int = 0, out: str | Path | None = None
          ) -> dict:
    try:
        import torch
        import torch.nn as nn
    except ImportError as e:
        raise SystemExit(
            "torch is required for BC training: uv sync --extra policy") from e

    g = torch.Generator().manual_seed(seed)
    n = len(obs)
    n_val = max(1, int(n * val_frac))
    perm = torch.randperm(n, generator=g).numpy()
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    mu, sd = obs[tr_idx].mean(0), obs[tr_idx].std(0) + 1e-6

    def to_t(x):
        return torch.as_tensor(x, dtype=torch.float32)

    obs_t = to_t((obs - mu) / sd)
    act_t = to_t(act)
    w_t = to_t(w)

    model = nn.Sequential(
        nn.Linear(obs.shape[1], 256), nn.GELU(),
        nn.Linear(256, 256), nn.GELU(),
        nn.Linear(256, act.shape[1]))
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    def loss_fn(pred, target, weight):
        cont = ((pred[:, :4] - target[:, :4]) ** 2).mean(dim=1)
        grip = bce(pred[:, 4], target[:, 4])
        return ((cont + 0.1 * grip) * weight).mean()

    history = []
    for ep in range(epochs):
        model.train()
        order = torch.randperm(len(tr_idx), generator=g).numpy()
        tot = 0.0
        for k in range(0, len(order), batch):
            idx = tr_idx[order[k:k + batch]]
            opt.zero_grad()
            loss = loss_fn(model(obs_t[idx]), act_t[idx], w_t[idx])
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        model.eval()
        with torch.no_grad():
            # Validation is ALWAYS unweighted: both variants are judged on
            # the same plain imitation objective.
            vloss = loss_fn(model(obs_t[val_idx]), act_t[val_idx],
                            torch.ones(len(val_idx)))
        history.append({"epoch": ep, "train": tot / len(tr_idx),
                        "val": float(vloss)})
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"epoch {ep:3d}  train {history[-1]['train']:.5f}  "
                  f"val {history[-1]['val']:.5f}")

    result = {"val_loss": history[-1]["val"], "history": history,
              "n_train": len(tr_idx), "n_val": n_val}
    if out:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "obs_mu": mu, "obs_sd": sd,
                    "obs_cols": OBS_COLS}, out / "bc_policy.pt")
        (out / "bc_result.json").write_text(json.dumps(result, indent=2))
        print(f"[bc] saved -> {out}")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("sessions", nargs="+")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--gamma", type=float, default=1.0,
                   help="attention weight exponent")
    p.add_argument("--uniform", action="store_true",
                   help="ablation: uniform weights instead of a(t)")
    p.add_argument("--out", default="data/bc")
    args = p.parse_args()

    obs, act, w = build_bc_dataset(args.sessions, gamma=args.gamma,
                                   uniform=args.uniform)
    tag = "uniform" if args.uniform else f"attn_g{args.gamma}"
    print(f"[bc] dataset: {len(obs)} steps, weight spread "
          f"p10={np.percentile(w,10):.2f} p90={np.percentile(w,90):.2f} ({tag})")
    train(obs, act, w, epochs=args.epochs, out=Path(args.out) / tag)
