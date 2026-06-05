"""Metrics for TrackMaze localization / state-tracking.

The headline objects:
  - drift_curve:      localization error vs step  (does latent state drift as the episode goes on?)
  - correction_curve: error vs steps-since-last-re-anchor  (does seeing a known landmark pull error back?)
  - generalization:   any metric vs maze size      (train small -> test large: where does each method collapse?)
  - spl:              navigation success weighted by path-optimality (Anderson 2018)

dead_reckon is the open-loop reference: integrate the agent's OWN commanded actions, no collision
knowledge. Its error == the drift the env injects, so it's the sanity baseline for "is there drift to
correct" and the ceiling a learned tracker must beat by using the egocentric observations.
"""
import math
import numpy as np

def dead_reckon(actions, p0, ang0, mv=0.22, rot=0.20):
    """Open-loop path integration of commanded actions (no collisions). Returns est pos (T,2), aligned
    so out[t] = estimate at obs-time t (out[0]=p0; out[t]=p0 after applying actions[0..t-1])."""
    x, y, a = float(p0[0]), float(p0[1]), float(ang0); out = [(x, y)]
    for t in range(len(actions) - 1):
        act = int(actions[t])
        if   act == 2: a -= rot
        elif act == 3: a += rot
        else:
            f = mv if act == 0 else -mv
            x += math.cos(a) * f; y += math.sin(a) * f
        out.append((x, y))
    return np.asarray(out, dtype=np.float32)

def per_step_error(pred, true):
    """pred,true: (n_eps,T,2) or (T,2). Returns Euclidean error (n_eps,T) or (T,)."""
    return np.linalg.norm(np.asarray(pred) - np.asarray(true), axis=-1)

def drift_curve(errs, nbins=12):
    """errs: (n_eps,T). Returns list[(step, mean_error)] binned over the episode."""
    errs = np.atleast_2d(errs); T = errs.shape[1]
    edges = np.linspace(0, T, nbins + 1).astype(int)
    out = []
    for b in range(nbins):
        lo, hi = edges[b], max(edges[b] + 1, edges[b + 1])
        out.append((int((lo + hi) // 2), float(errs[:, lo:hi].mean())))
    return out

def correction_curve(errs, reanchor, max_gap=40):
    """Bin error by #steps since the last informative (unique-landmark) observation.
    errs,reanchor: (n_eps,T). Returns list[(steps_since_reanchor, mean_error, count)].
    A falling curve at small gaps == seeing a known landmark corrects the latent estimate."""
    errs = np.atleast_2d(errs); reanchor = np.atleast_2d(reanchor)
    bucket_sum = np.zeros(max_gap + 1); bucket_cnt = np.zeros(max_gap + 1)
    for e in range(errs.shape[0]):
        since = max_gap
        for t in range(errs.shape[1]):
            if reanchor[e, t]: since = 0
            else: since = min(since + 1, max_gap)
            bucket_sum[since] += errs[e, t]; bucket_cnt[since] += 1
    out = []
    for g in range(max_gap + 1):
        if bucket_cnt[g] > 0:
            out.append((g, float(bucket_sum[g] / bucket_cnt[g]), int(bucket_cnt[g])))
    return out

def spl(success, shortest_len, actual_len):
    """Success weighted by (inverse) Path Length. Anderson 2018. Arrays over episodes."""
    success = np.asarray(success, dtype=float); shortest = np.asarray(shortest_len, dtype=float)
    actual = np.maximum(np.asarray(actual_len, dtype=float), 1e-6)
    return float((success * shortest / np.maximum(shortest, actual)).mean())

def generalization(rows):
    """rows: list of (size, metric_value). Returns sorted [(size, value)] — the collapse curve."""
    return sorted((int(s), float(v)) for s, v in rows)
