"""Attention feature channels from aligned device streams.

All inputs are on the MASTER clock (post align.align). All outputs are
returned on a uniform "attention grid" (default 60 Hz), each channel
oriented so that LARGER = deeper attention:

    msacc_inhib    -z(microsaccade rate)        [SLO]   Engbert-Kliegl
                   microsaccade rate drops under load / before action
    fix_stability  -z(log drift speed)          [SLO]   tighter fixation =
                   deeper processing ("quiet eye")
    vigor          z(saccadic peak-velocity residual vs the session's own
                   main sequence)               [pupil gaze]  saccades to
                   targets the subject values are faster than the main
                   sequence predicts
    tepr           z(luminance-corrected pupil) [pupil] task-evoked pupil
                   response after regressing out the display-driven light
                   reflex (mean_lum convolved with a PLR kernel)

Each channel comes with a validity mask (NaN where the source lost lock).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

ATTN_HZ = 60.0


# ------------------------------------------------------------------ helpers


def _grid(t0: float, t1: float, hz: float = ATTN_HZ) -> np.ndarray:
    return np.arange(t0, t1, 1.0 / hz)


def _nan_z(x: np.ndarray) -> np.ndarray:
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    return (x - mu) / (sd + 1e-12)


def _causal_smooth(x: np.ndarray, hz: float, tau_s: float) -> np.ndarray:
    """Exponential moving average; NaNs propagated through gaps.

    Causal (real-time) filter. The offline pipeline should prefer
    `_smooth_zerophase` so event timing is not lagged; this is kept for any
    future real-time/closed-loop use.
    """
    alpha = 1.0 - np.exp(-1.0 / (hz * tau_s))
    out = np.empty_like(x)
    acc = np.nan
    for i, v in enumerate(x):
        if np.isfinite(v):
            acc = v if not np.isfinite(acc) else acc + alpha * (v - acc)
        out[i] = acc
    return out


def _smooth_zerophase(x: np.ndarray, hz: float, tau_s: float) -> np.ndarray:
    """NaN-aware symmetric Gaussian smoother (zero phase lag).

    Used for OFFLINE feature extraction so a(t) is not shifted relative to
    task events. NaNs are interpolated for the convolution then restored, so
    gaps (blinks, SLO lock loss) stay NaN in the output.
    """
    x = np.asarray(x, dtype=float)
    sigma = max(tau_s * hz, 0.5)
    n = int(sigma * 4) | 1
    k = np.exp(-0.5 * ((np.arange(n) - n // 2) / sigma) ** 2)
    k /= k.sum()
    nan = ~np.isfinite(x)
    if nan.all():
        return x.copy()
    filled = x.copy()
    if nan.any():
        idx = np.arange(len(x))
        filled[nan] = np.interp(idx[nan], idx[~nan], x[~nan])
    out = np.convolve(filled, k, mode="same")
    out[nan] = np.nan
    return out


def _interp_to(grid: np.ndarray, t: np.ndarray, x: np.ndarray,
               max_gap_s: float = 0.15) -> np.ndarray:
    ok = np.isfinite(x) & np.isfinite(t)
    if ok.sum() < 2:
        return np.full_like(grid, np.nan)
    y = np.interp(grid, t[ok], x[ok])
    tt = t[ok]
    k = np.clip(np.searchsorted(tt, grid), 1, len(tt) - 1)
    dist = np.minimum(np.abs(grid - tt[k - 1]), np.abs(tt[k] - grid))
    y[dist > max_gap_s] = np.nan
    return y


# ------------------------------------------------- saccade/microsaccade det


@dataclasses.dataclass
class SaccadeEvent:
    t_on: float
    t_off: float
    amp: float        # amplitude (units of the input trace)
    v_peak: float     # peak velocity (units/s)


def detect_saccades(t: np.ndarray, x: np.ndarray, y: np.ndarray,
                    lam: float = 6.0, min_dur_s: float = 0.006,
                    min_amp: float = 0.0, max_amp: float = np.inf
                    ) -> list[SaccadeEvent]:
    """Engbert-Kliegl velocity-threshold (micro)saccade detector.

    Works at any rate / unit: SLO arcmin trace for microsaccades, calibrated
    pupil-cam gaze in degrees for ordinary saccades. NaN samples split the
    trace into independent valid segments (blinks / lock loss).
    """
    events: list[SaccadeEvent] = []
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(t)
    # Segment boundaries at NaN runs.
    seg_starts = np.flatnonzero(ok & ~np.roll(ok, 1))
    seg_ends = np.flatnonzero(ok & ~np.roll(ok, -1))
    if ok[0]:
        seg_starts = np.r_[0, seg_starts[seg_starts != 0]]
    if ok[-1]:
        seg_ends = np.r_[seg_ends[seg_ends != len(ok) - 1], len(ok) - 1]

    for s, e in zip(seg_starts, seg_ends):
        if e - s < 5:
            continue
        ts, xs, ys = t[s:e + 1], x[s:e + 1], y[s:e + 1]
        dt = np.gradient(ts)
        vx = np.gradient(xs) / dt
        vy = np.gradient(ys) / dt
        # Engbert-Kliegl median-based velocity SD.
        sx = np.sqrt(np.median(vx ** 2) - np.median(vx) ** 2) + 1e-9
        sy = np.sqrt(np.median(vy ** 2) - np.median(vy) ** 2) + 1e-9
        crit = (vx / (lam * sx)) ** 2 + (vy / (lam * sy)) ** 2 > 1.0

        i = 0
        n = len(crit)
        while i < n:
            if not crit[i]:
                i += 1
                continue
            j = i
            while j + 1 < n and crit[j + 1]:
                j += 1
            dur = ts[j] - ts[i]
            if dur >= min_dur_s:
                amp = float(np.hypot(xs[j] - xs[i], ys[j] - ys[i]))
                vpk = float(np.max(np.hypot(vx[i:j + 1], vy[i:j + 1])))
                if min_amp <= amp <= max_amp:
                    events.append(SaccadeEvent(float(ts[i]), float(ts[j]), amp, vpk))
            i = j + 1
    return events


# ----------------------------------------------------------- SLO channels


def microsaccade_channels(slo: pd.DataFrame, grid: np.ndarray,
                          rate_tau_s: float = 0.6,
                          arcmin_max: float = 60.0) -> pd.DataFrame:
    """msacc_inhib and fix_stability from the aligned SLO trace."""
    t = slo["t_master"].to_numpy()
    x = slo["x_arcmin"].to_numpy(dtype=float)
    y = slo["y_arcmin"].to_numpy(dtype=float)
    if "quality" in slo:
        bad = slo["quality"].to_numpy(dtype=float) < 0.5
        x = np.where(bad, np.nan, x)
        y = np.where(bad, np.nan, y)

    # Microsaccades: 3-60 arcmin excursions.
    events = detect_saccades(t, x, y, lam=6.0, min_dur_s=0.006,
                             min_amp=3.0, max_amp=arcmin_max)
    hz = 1.0 / float(np.median(np.diff(grid)))
    counts = np.zeros(len(grid))
    if events:
        idx = np.clip(np.searchsorted(grid, [e.t_on for e in events]),
                      0, len(grid) - 1)
        np.add.at(counts, idx, 1.0)
    rate = _smooth_zerophase(counts * hz, hz, rate_tau_s)

    # Drift speed inside fixations (saccade samples masked out).
    in_sacc = np.zeros(len(t), dtype=bool)
    for e in events:
        in_sacc |= (t >= e.t_on - 0.005) & (t <= e.t_off + 0.005)
    xs = np.where(in_sacc, np.nan, x)
    ys = np.where(in_sacc, np.nan, y)
    dt = np.gradient(t)
    drift_speed = np.hypot(np.gradient(np.nan_to_num(xs, nan=np.nan)),
                           np.gradient(np.nan_to_num(ys, nan=np.nan))) / dt
    drift_speed[~(np.isfinite(xs) & np.isfinite(ys))] = np.nan
    drift_on_grid = _interp_to(grid, t, drift_speed)
    drift_on_grid = _smooth_zerophase(drift_on_grid, hz, 0.3)

    valid = _interp_to(grid, t, np.isfinite(x).astype(float)) > 0.5
    out = pd.DataFrame({
        "t_master": grid,
        "msacc_rate_hz": rate,
        "msacc_inhib": -_nan_z(rate),
        "fix_stability": -_nan_z(np.log(drift_on_grid + 1e-3)),
        "slo_valid": valid,
    })
    out.loc[~valid, ["msacc_inhib", "fix_stability"]] = np.nan
    return out


# --------------------------------------------------------- pupil channels


def vigor_channel(gaze_deg: pd.DataFrame, grid: np.ndarray,
                  hold_tau_s: float = 1.5) -> pd.DataFrame:
    """Saccadic vigor from calibrated pupil-cam gaze (columns gx_deg, gy_deg).

    Fits this session's own main sequence v_peak = Vm * (1 - exp(-amp/c))
    and scores each saccade by its multiplicative residual; the sparse
    per-saccade scores are written onto the grid with exponential decay
    back to 0 (= typical vigor).
    """
    t = gaze_deg["t_master"].to_numpy()
    events = detect_saccades(t, gaze_deg["gx_deg"].to_numpy(),
                             gaze_deg["gy_deg"].to_numpy(),
                             lam=6.0, min_dur_s=0.012, min_amp=1.0)
    out = pd.DataFrame({"t_master": grid})
    if len(events) < 6:
        out["vigor"] = np.nan
        out["n_saccades"] = len(events)
        return out

    amp = np.array([e.amp for e in events])
    vpk = np.array([e.v_peak for e in events])
    # Fit Vm, c by coarse grid search (robust, tiny problem).
    best = (np.inf, 1.0, 1.0)
    for c in np.geomspace(1.0, 20.0, 40):
        pred_unit = 1.0 - np.exp(-amp / c)
        vm = np.median(vpk / pred_unit)
        loss = np.median(np.abs(vm * pred_unit - vpk))
        if loss < best[0]:
            best = (loss, vm, c)
    _, vm, c = best
    resid = np.log(vpk / (vm * (1.0 - np.exp(-amp / c))))

    z = (resid - np.mean(resid)) / (np.std(resid) + 1e-12)
    hz = 1.0 / float(np.median(np.diff(grid)))
    sig = np.zeros(len(grid))
    decay = np.exp(-1.0 / (hz * hold_tau_s))
    ei = 0
    ev_t = [e.t_on for e in events]
    val = 0.0
    for i, g in enumerate(grid):
        val *= decay
        while ei < len(events) and ev_t[ei] <= g:
            val = float(z[ei])
            ei += 1
        sig[i] = val
    out["vigor"] = sig
    out["n_saccades"] = len(events)
    return out


def tepr_channel(pupil: pd.DataFrame, frames: pd.DataFrame,
                 grid: np.ndarray) -> pd.DataFrame:
    """Luminance-corrected pupil: regress out the display light reflex.

    diam ~ b0 + b1 * lum_plr(t - lag) with the PLR kernel approximated by a
    300 ms exponential low-pass and the best lag in [0.15, 0.45] s chosen by
    correlation. tepr = z(residual), low-passed to the TEPR band (~0.5 Hz).
    """
    hz = 1.0 / float(np.median(np.diff(grid)))
    diam = _interp_to(grid, pupil["t_master"].to_numpy(),
                      pupil["diam"].to_numpy(dtype=float))
    lum = _interp_to(grid, frames["t_master"].to_numpy(),
                     frames["mean_lum"].to_numpy(dtype=float), max_gap_s=0.5)
    # PLR kernel approximation: causal lag is physical here (the pupil
    # responds AFTER the light), so the regressor stays causal.
    lum_lp = _causal_smooth(lum, hz, 0.3)

    best_lag, best_r = 0.25, 0.0
    ok_base = np.isfinite(diam) & np.isfinite(lum_lp)
    for lag in np.arange(0.15, 0.46, 0.05):
        sh = int(lag * hz)
        l_sh = np.r_[np.full(sh, np.nan), lum_lp[:-sh]] if sh else lum_lp
        ok = ok_base & np.isfinite(l_sh)
        if ok.sum() < 50:
            continue
        r = np.corrcoef(diam[ok], l_sh[ok])[0, 1]
        if abs(r) > abs(best_r):
            best_r, best_lag = r, lag
    sh = int(best_lag * hz)
    l_sh = np.r_[np.full(sh, np.nan), lum_lp[:-sh]] if sh else lum_lp

    ok = np.isfinite(diam) & np.isfinite(l_sh)
    A = np.stack([np.ones(ok.sum()), l_sh[ok]], axis=1)
    coef, *_ = np.linalg.lstsq(A, diam[ok], rcond=None)
    resid = np.full_like(diam, np.nan)
    resid[ok] = diam[ok] - A @ coef
    tepr = _smooth_zerophase(resid, hz, 0.6)

    return pd.DataFrame({
        "t_master": grid,
        "pupil_diam": diam,
        "lum_regressor": l_sh,
        "tepr": _nan_z(tepr),
        "plr_beta": np.full(len(grid), coef[1]),
        "plr_lag_s": np.full(len(grid), best_lag),
    })


# ------------------------------------------------------------ entry point


def extract_features(slo_aligned: pd.DataFrame, pupil_aligned: pd.DataFrame,
                     gaze_deg: pd.DataFrame, frames: pd.DataFrame,
                     hz: float = ATTN_HZ) -> pd.DataFrame:
    """All channels joined on one attention grid."""
    t0 = max(float(slo_aligned["t_master"].min()),
             float(pupil_aligned["t_master"].min()))
    t1 = min(float(slo_aligned["t_master"].max()),
             float(pupil_aligned["t_master"].max()))
    grid = _grid(t0, t1, hz)

    ms = microsaccade_channels(slo_aligned, grid)
    vg = vigor_channel(gaze_deg, grid)
    tp = tepr_channel(pupil_aligned, frames, grid)

    out = ms.merge(vg, on="t_master").merge(tp, on="t_master")
    return out
