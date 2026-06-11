"""Cross-clock alignment: device streams -> master timeline.

Pipeline per device stream:
1. Detect photic-flash pulses in the stream's `illum` channel (high-pass to
   kill slow display-luminance variation, then a robust MAD threshold; pulse
   onsets refined to the threshold crossing).
2. Match detected onsets to the master flash anchors (events.jsonl
   `flash_on` times) by sliding the gap pattern -- the start/end bursts have
   mirrored, asymmetric gaps, so the match is unambiguous even with missed
   or spurious pulses.
3. Fit t_master = a * t_dev + b (offset + drift) by least squares on the
   matched pairs; report per-anchor residuals.

Note: any CONSTANT device latency (exposure -> file timestamp) is absorbed
into b and cannot be observed without a hardware probe; it is the same for
every anchor so relative timing -- which is what a(t) needs -- is preserved.

`to_master` then rewrites a stream onto the master clock; downstream code
never sees a device clock again.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

from ingest.base import DeviceStream


# ------------------------------------------------------------ pulse detect


def detect_flash_onsets(t: np.ndarray, illum: np.ndarray,
                        min_gap_s: float = 0.15,
                        mad_k: float = 8.0) -> np.ndarray:
    """Onset times of photic-flash pulses in an illumination channel."""
    x = np.asarray(illum, dtype=float)
    ok = np.isfinite(x)
    if ok.sum() < 10:
        return np.array([])
    x = np.where(ok, x, np.nanmedian(x))

    rate = 1.0 / max(float(np.median(np.diff(t))), 1e-6)
    win = max(3, int(rate * 1.0) | 1)
    hp = x - median_filter(x, size=win)

    mad = np.median(np.abs(hp - np.median(hp))) + 1e-12
    thresh = np.median(hp) + mad_k * mad
    above = hp > thresh
    if not above.any():
        return np.array([])

    onsets = []
    idx = np.flatnonzero(above)
    last_t = -np.inf
    for i in idx:
        if t[i] - last_t < min_gap_s:
            last_t = t[i]
            continue
        # Sub-sample refinement: linear interp of the threshold crossing.
        if i > 0 and hp[i - 1] < thresh < hp[i]:
            frac = (thresh - hp[i - 1]) / (hp[i] - hp[i - 1])
            onsets.append(t[i - 1] + frac * (t[i] - t[i - 1]))
        else:
            onsets.append(t[i])
        last_t = t[i]
    return np.asarray(onsets)


# ----------------------------------------------------------------- matching


def match_anchors(master_onsets: np.ndarray, dev_onsets: np.ndarray,
                  gap_tol_s: float = 0.08) -> list[tuple[float, float]]:
    """Match master flash times to detected device pulse times.

    Slides every contiguous window of either sequence against the other and
    scores agreement of consecutive gaps (clock drift over a session is
    <<1 ms, so gaps must agree to within detection noise). Returns matched
    (t_master, t_dev) pairs from the best window; falls back to a greedy
    per-gap match when counts differ wildly.
    """
    M, D = np.asarray(master_onsets), np.asarray(dev_onsets)
    if len(M) < 2 or len(D) < 2:
        return []

    def score_window(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.mean(np.abs(np.diff(a) - np.diff(b))))

    best: tuple[float, list[tuple[float, float]]] = (np.inf, [])
    if len(D) >= len(M):
        for j in range(len(D) - len(M) + 1):
            w = D[j:j + len(M)]
            s = score_window(M, w)
            if s < best[0]:
                best = (s, list(zip(M, w)))
    else:
        for i in range(len(M) - len(D) + 1):
            w = M[i:i + len(D)]
            s = score_window(w, D)
            if s < best[0]:
                best = (s, list(zip(w, D)))

    if best[0] < gap_tol_s:
        return best[1]

    # Greedy fallback: anchor on the best-matching single gap pair, then
    # extend with nearest-neighbor matching under the predicted offset.
    dm, dd = np.diff(M), np.diff(D)
    pairs: list[tuple[float, float]] = []
    ij = [(i, j) for i in range(len(dm)) for j in range(len(dd))
          if abs(dm[i] - dd[j]) < gap_tol_s]
    if not ij:
        return []
    i0, j0 = ij[0]
    offset = D[j0] - M[i0]
    for m in M:
        k = int(np.argmin(np.abs(D - (m + offset))))
        if abs(D[k] - (m + offset)) < gap_tol_s:
            pairs.append((m, D[k]))
    return pairs


# ---------------------------------------------------------------------- fit


@dataclasses.dataclass
class ClockFit:
    a: float                  # t_master = a * t_dev + b
    b: float
    rms_ms: float
    max_ms: float
    n_anchors: int
    pairs: list = dataclasses.field(default_factory=list)

    def dev_to_master(self, t_dev: np.ndarray) -> np.ndarray:
        return self.a * np.asarray(t_dev) + self.b

    def summary(self) -> str:
        drift_ppm = (self.a - 1.0) * 1e6
        return (f"t_master = {self.a:.9f} * t_dev + {self.b:+.4f}s "
                f"(drift {drift_ppm:+.0f} ppm, {self.n_anchors} anchors, "
                f"residual rms {self.rms_ms:.1f} ms, max {self.max_ms:.1f} ms)")


def fit_clock(pairs: list[tuple[float, float]]) -> ClockFit:
    if len(pairs) < 2:
        raise RuntimeError(f"need >= 2 matched anchors, got {len(pairs)}")
    tm = np.array([p[0] for p in pairs])
    td = np.array([p[1] for p in pairs])
    A = np.stack([td, np.ones_like(td)], axis=1)
    (a, b), *_ = np.linalg.lstsq(A, tm, rcond=None)
    resid = (a * td + b) - tm
    return ClockFit(a=float(a), b=float(b),
                    rms_ms=float(np.sqrt(np.mean(resid ** 2)) * 1000),
                    max_ms=float(np.max(np.abs(resid)) * 1000),
                    n_anchors=len(pairs), pairs=pairs)


# ------------------------------------------------------------- session API


def load_master_anchors(session_dir: str | Path) -> np.ndarray:
    evs = [json.loads(l) for l in open(Path(session_dir) / "events.jsonl")]
    return np.array([e["t_master"] for e in evs if e.get("kind") == "flash_on"])


def align_stream(stream: DeviceStream, session_dir: str | Path,
                 max_rms_ms: float = 25.0) -> ClockFit:
    """Detect -> match -> fit for one device stream; raises if too poor."""
    master = load_master_anchors(session_dir)
    onsets = detect_flash_onsets(stream.t_dev, stream.data["illum"].to_numpy())
    pairs = match_anchors(master, onsets)
    if len(pairs) < 2:
        raise RuntimeError(
            f"[{stream.kind}] anchor matching failed: "
            f"{len(master)} master anchors, {len(onsets)} detected pulses")
    fit = fit_clock(pairs)
    if fit.rms_ms > max_rms_ms:
        raise RuntimeError(
            f"[{stream.kind}] alignment residual too large: {fit.summary()}")
    return fit


def to_master(stream: DeviceStream, fit: ClockFit) -> pd.DataFrame:
    """Stream channels with a t_master column, sorted, device clock dropped."""
    df = stream.data.copy()
    df.insert(0, "t_master", fit.dev_to_master(stream.t_dev))
    return df.sort_values("t_master").reset_index(drop=True)


def resample_to_grid(df: pd.DataFrame, rate_hz: float,
                     t0: float | None = None, t1: float | None = None,
                     max_gap_s: float = 0.1) -> pd.DataFrame:
    """Interpolate channels onto a uniform master grid; NaN across big gaps."""
    t = df["t_master"].to_numpy()
    t0 = float(t[0]) if t0 is None else t0
    t1 = float(t[-1]) if t1 is None else t1
    grid = np.arange(t0, t1, 1.0 / rate_hz)
    out = {"t_master": grid}
    gap = np.diff(t, prepend=t[0])
    for col in df.columns:
        if col == "t_master":
            continue
        x = df[col].to_numpy(dtype=float)
        ok = np.isfinite(x)
        if ok.sum() < 2:
            out[col] = np.full_like(grid, np.nan)
            continue
        y = np.interp(grid, t[ok], x[ok])
        # Mask grid points that fall inside large source gaps (blinks, lock
        # loss): nearest source sample further than max_gap_s away.
        nearest = np.searchsorted(t[ok], grid)
        nearest = np.clip(nearest, 1, ok.sum() - 1)
        tt = t[ok]
        dist = np.minimum(np.abs(grid - tt[nearest - 1]), np.abs(tt[nearest] - grid))
        y[dist > max_gap_s] = np.nan
        out[col] = y
    _ = gap
    return pd.DataFrame(out)
