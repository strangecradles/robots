"""Construct validity of a(t): does it track task structure?

The thesis only holds if the extracted attention scalar behaves like
attention: rises into grasps (quiet eye), spikes at drops (orienting
response), and sits low during ballistic transit. This module produces the
day-1 deliverable: event-locked averages of a(t) + a full-session trace,
and -- when the session has synthetic ground truth -- the correlation
between extracted a(t) and the generating a_true(t).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EVENT_COLORS = {
    "grasp": "tab:green",
    "lift": "tab:olive",
    "release": "tab:blue",
    "place_success": "tab:purple",
    "drop": "tab:red",
}


def event_locked(a_df: pd.DataFrame, event_times: list[float],
                 window: tuple[float, float] = (-2.0, 2.0),
                 hz: float = 60.0) -> tuple[np.ndarray, np.ndarray]:
    """(rel_time grid, segments matrix) of a(t) around each event."""
    rel = np.arange(window[0], window[1], 1.0 / hz)
    t = a_df["t_master"].to_numpy()
    a = a_df["a_z"].to_numpy(dtype=float)
    ok = np.isfinite(a)
    segs = []
    for te in event_times:
        if te + window[0] < t[0] or te + window[1] > t[-1]:
            continue
        segs.append(np.interp(rel + te, t[ok], a[ok]))
    return rel, (np.vstack(segs) if segs else np.empty((0, len(rel))))


def transit_mask(a_df: pd.DataFrame, frames: pd.DataFrame,
                 event_times: list[float], clear_s: float = 1.0) -> np.ndarray:
    """Samples during 'ballistic transit': carrying the cube, no event near."""
    t = a_df["t_master"].to_numpy()
    grasped = np.interp(t, frames["t_master"], frames["grasped"].astype(float)) > 0.5
    near_event = np.zeros(len(t), dtype=bool)
    for te in event_times:
        near_event |= np.abs(t - te) < clear_s
    return grasped & ~near_event


def validity_report(session_dir: str | Path, a_df: pd.DataFrame,
                    events: list[dict], frames: pd.DataFrame,
                    out_dir: str | Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    task_events = [e for e in events if e.get("kind") == "task_event"]
    by_label: dict[str, list[float]] = {}
    for e in task_events:
        by_label.setdefault(e["label"], []).append(e["t_master"])
    all_times = [e["t_master"] for e in task_events]

    t = a_df["t_master"].to_numpy()
    a = a_df["a_z"].to_numpy(dtype=float)

    tmask = transit_mask(a_df, frames, all_times)
    transit_mean = float(np.nanmean(a[tmask])) if tmask.any() else float("nan")
    transit_sd = float(np.nanstd(a[tmask])) if tmask.any() else float("nan")

    fig, axes = plt.subplots(3, 1, figsize=(12, 11))

    # Panel 1: full-session trace.
    ax = axes[0]
    ax.plot(t, a, lw=0.8, color="k", label="a(t) (z)")
    if "a_conf" in a_df:
        ax.fill_between(t, -3, 3, where=a_df["a_conf"].to_numpy() < 0.5,
                        color="0.85", label="low confidence")
    for label, times in by_label.items():
        c = EVENT_COLORS.get(label, "gray")
        for i, te in enumerate(times):
            ax.axvline(te, color=c, lw=1.2, alpha=0.8,
                       label=label if i == 0 else None)
    ax.set_xlabel("t_master (s)")
    ax.set_ylabel("a(t) z")
    ax.set_title("attention a(t) over the session")
    ax.legend(loc="upper right", fontsize=8, ncol=3)

    # Panel 2: event-locked averages vs transit baseline.
    ax = axes[1]
    stats: dict = {"transit_mean": transit_mean, "transit_sd": transit_sd}
    if np.isfinite(transit_mean):
        ax.axhspan(transit_mean - transit_sd, transit_mean + transit_sd,
                   color="0.9", label="transit baseline +/- sd")
        ax.axhline(transit_mean, color="0.6", lw=1)
    for label, times in by_label.items():
        rel, segs = event_locked(a_df, times)
        if not len(segs):
            continue
        mean = np.nanmean(segs, axis=0)
        sem = np.nanstd(segs, axis=0) / max(np.sqrt(len(segs)), 1)
        c = EVENT_COLORS.get(label, "gray")
        ax.plot(rel, mean, color=c, label=f"{label} (n={len(segs)})")
        ax.fill_between(rel, mean - sem, mean + sem, color=c, alpha=0.2)
        stats[f"peak_{label}"] = float(np.nanmax(mean))
        stats[f"peak_t_{label}"] = float(rel[np.nanargmax(mean)])
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("time relative to event (s)")
    ax.set_ylabel("a(t) z")
    ax.set_title("event-locked a(t) vs ballistic-transit baseline")
    ax.legend(fontsize=8)

    # Panel 3: ground truth comparison (synthetic sessions only).
    ax = axes[2]
    truth_path = Path(session_dir) / "synthetic" / "truth.parquet"
    if truth_path.exists():
        truth = pd.read_parquet(truth_path)
        at = np.interp(t, truth["t_master"], truth["a_true"])
        # Correlate over the teleop phase only: calibration has no task
        # attention, so including it just adds noise-vs-flat baseline.
        teleop_t0 = min((e["t_master"] for e in events
                         if e.get("kind") == "teleop_start"), default=t[0])
        ok = np.isfinite(a) & (t >= teleop_t0)
        r = float(np.corrcoef(a[ok], at[ok])[0, 1])
        stats["corr_with_truth"] = r
        stats["corr_window"] = "teleop"
        ax.plot(t, at, color="tab:orange", lw=1.0, label="a_true (generator)")
        ax2 = ax.twinx()
        ax2.plot(t, a, color="k", lw=0.7, alpha=0.8, label="a(t) extracted")
        ax.axvline(teleop_t0, color="0.6", ls=":", lw=1)
        ax.set_title(f"extracted a(t) vs synthetic ground truth  "
                     f"r = {r:.3f} (teleop phase)")
        ax.set_xlabel("t_master (s)")
        ax.legend(loc="upper left", fontsize=8)
        ax2.legend(loc="upper right", fontsize=8)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "no synthetic ground truth for this session",
                ha="center", va="center")

    fig.tight_layout()
    png = out_dir / "validity.png"
    fig.savefig(png, dpi=130)
    plt.close(fig)

    (out_dir / "validity.json").write_text(json.dumps(stats, indent=2))
    return stats
