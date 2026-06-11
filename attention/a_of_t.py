"""Fuse attention feature channels into the scalar a(t).

v0 fusion is deliberately simple and inspectable: a NaN-aware weighted sum
of the z-scored channels (all oriented larger = deeper attention), causally
smoothed, plus a squashed [0, 1] variant and a per-sample confidence equal
to the fraction of channel weight that was actually observed (channels drop
out during blinks / SLO lock loss).

Later: replace `fuse` with a learned combiner trained against behavioral
outcomes (e.g. grasp success), keeping the same interface.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from attention.features import _smooth_zerophase

DEFAULT_WEIGHTS = {
    "msacc_inhib": 0.40,    # strongest, SLO-unique channel
    "fix_stability": 0.25,
    "tepr": 0.20,
    "vigor": 0.15,
}

# Physiological response latencies, compensated so every channel reports on
# the same "neural now": the task-evoked pupil response peaks ~0.9 s after
# the cognitive event it indexes, so its value at t+lag is assigned to t.
DEFAULT_LAGS_S = {
    "tepr": 0.9,
}


def fuse(features: pd.DataFrame, weights: dict[str, float] | None = None,
         smooth_tau_s: float = 0.25,
         lags_s: dict[str, float] | None = None) -> pd.DataFrame:
    w = weights or DEFAULT_WEIGHTS
    lags = DEFAULT_LAGS_S if lags_s is None else lags_s
    grid = features["t_master"].to_numpy()
    hz = 1.0 / float(np.median(np.diff(grid)))

    num = np.zeros(len(grid))
    den = np.zeros(len(grid))
    for ch, wi in w.items():
        if ch not in features:
            continue
        x = features[ch].to_numpy(dtype=float)
        lag = lags.get(ch, 0.0)
        if lag:
            sh = int(round(lag * hz))
            x = np.r_[x[sh:], np.full(sh, np.nan)]  # value at t+lag -> t
        ok = np.isfinite(x)
        num[ok] += wi * x[ok]
        den[ok] += wi

    raw = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
    a_z = _smooth_zerophase(raw, hz, smooth_tau_s)
    a01 = 1.0 / (1.0 + np.exp(-a_z))

    out = features.copy()
    out["a_z"] = a_z                      # fused attention, z-scale
    out["a"] = a01                        # squashed to (0, 1)
    out["a_conf"] = den / sum(w.values())  # fraction of weight observed
    return out
