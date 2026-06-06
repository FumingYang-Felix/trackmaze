"""se2heading: HEADING-drift estimation & correction loop closer.

Tracker model:  est = R(theta) . true_disp_from_start + drift,  theta (heading error) and drift each a random
walk per move. Two visits to the SAME true cell at times t0<t1 have ests differing by R(theta(t1)-theta(t0))
plus a drift difference. At scale theta reaches >1 rad, so absolute-est gating (TightGate) both MISSES real
loops (ests rotate apart) and FALSE-MERGES different cells (ests rotate together) -> success collapses.

The fix has two pillars:

(A) DIRECT HEADING ESTIMATION FROM ODOMETRY (the key move).  Every explore step is one cell along a CARDINAL
    direction in the true maze, so the est-delta of a single step is  R(theta) . cardinal + noise.  We track
    theta_hat online: de-rotate each step-delta by -theta_hat, snap to the nearest cardinal axis, and feed the
    residual angle through a low-pass filter.  This recovers theta even when it grows to >1 rad WITHOUT needing
    any loop closures (so it does not get stuck at a self-consistent theta_hat=0 fixed point the way a
    closure-only estimator does).  Closures additionally refine theta_hat when their baseline is long.

(B) HEADING-CORRECTED, ROTATION-INVARIANT-PREGATED GATE.  De-rotate the current est by -theta_hat and each
    node by -its-theta-at-creation into a common start frame.  Pre-gate on |est| (distance-from-start
    magnitude), which is INVARIANT to the heading rotation and is the strongest defence against false merges
    of different cells; then a tight angular gate in the de-rotated frame.  De-rotation lets real loops whose
    raw angle has rotated away still align, while the tight angular + magnitude gate keeps false merges out.
"""
import math
import numpy as np

_CARD = [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)]


def _rot(v, th):
    c, s = math.cos(th), math.sin(th)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


class MyCloser:
    def __init__(self,
                 ang_gate=0.6,       # tight accept radius in the de-rotated (start) frame
                 mag_tol=0.4,        # rotation-invariant magnitude pre-gate (absolute)
                 odo_lp=0.15,        # low-pass gain for the per-step odometry heading update
                 clo_lp=0.10,        # low-pass gain for the per-closure heading refinement
                 step_lo=0.5,        # a delta this short or shorter is not a clean single step (skip)
                 step_hi=1.8,        # a delta longer than this spans a backtrack/teleport (skip)
                 min_baseline=3.0,   # only refine heading from closures with baseline magnitude above this
                 th_deadband=0.8,    # only DE-ROTATE the gate when |theta_hat| exceeds this (else mag-only,
                                     # which is already excellent and not worth perturbing with theta noise)
                 derot_slack=0.0):   # extra angular-gate slack when de-rotating (absorbs theta_hat residual)
        self.ang_gate = ang_gate
        self.mag_tol = mag_tol
        self.odo_lp = odo_lp
        self.clo_lp = clo_lp
        self.step_lo = step_lo
        self.step_hi = step_hi
        self.min_baseline = min_baseline
        self.th_deadband = th_deadband
        self.derot_slack = derot_slack

    def reset(self):
        self.theta_hat = 0.0
        self.node_theta = []     # theta_hat believed when each node was created
        self.prev_est = None     # est at the previous match() call (for odometry heading update)

    def _ensure(self, nodes):
        while len(self.node_theta) < len(nodes):
            self.node_theta.append(self.theta_hat)

    def _odo_update(self, est):
        """Update theta_hat from a single-step est-delta by snapping the de-rotated delta to a cardinal axis."""
        if self.prev_est is not None:
            delta = est - self.prev_est
            L = math.hypot(delta[0], delta[1])
            if self.step_lo < L < self.step_hi:
                dd = _rot(delta, -self.theta_hat)               # de-rotate by current belief
                best = max(_CARD, key=lambda c: dd[0] * c[0] + dd[1] * c[1])
                resid = math.atan2(dd[1], dd[0]) - math.atan2(best[1], best[0])
                resid = math.atan2(math.sin(resid), math.cos(resid))
                self.theta_hat = self.theta_hat + self.odo_lp * resid
        self.prev_est = est

    def match(self, cell, est, opens, nodes):
        self._ensure(nodes)
        est = np.asarray(est, float)
        self._odo_update(est)

        if not nodes:
            return None

        # Only DE-ROTATE the angular gate when there is meaningful accumulated heading to correct. Below the
        # dead-band the magnitude pre-gate + raw-frame angular gate already work very well, and de-rotating by
        # a near-zero (but noisy) theta_hat only injects error -> false merges. When theta_hat is large
        # (the catastrophic-drift mazes) de-rotation is essential and earns its keep.
        derotating = abs(self.theta_hat) > self.th_deadband
        use_th = self.theta_hat if derotating else 0.0

        cur_c = _rot(est, -use_th)                     # current est in (de-rotated) frame
        cur_mag = float(np.hypot(cur_c[0], cur_c[1]))
        ang = self.ang_gate + (self.derot_slack if derotating else 0.0)
        agg = ang * ang

        best, bd = None, agg
        for i, nd in enumerate(nodes):
            if nd["opens"] != opens:
                continue
            n_th = self.node_theta[i] if abs(self.node_theta[i]) > self.th_deadband else 0.0
            nd_c = _rot(nd["est"], -n_th)
            nd_mag = float(np.hypot(nd_c[0], nd_c[1]))
            if abs(cur_mag - nd_mag) > self.mag_tol:   # rotation-invariant pre-gate
                continue
            d = cur_c - nd_c
            dd = float(d @ d)
            if dd < bd:
                bd, best = dd, i

        if best is None:
            return None

        # refine theta_hat from a long-baseline confirmed closure
        nd = nodes[best]
        n_th_best = self.node_theta[best] if abs(self.node_theta[best]) > self.th_deadband else 0.0
        nd_c = _rot(nd["est"], -n_th_best)
        n_ref = math.hypot(nd_c[0], nd_c[1])
        if n_ref > self.min_baseline:
            a_est = math.atan2(est[1], est[0])        # angle(true_disp)+theta_now
            a_ref = math.atan2(nd_c[1], nd_c[0])       # ~angle(true_disp)
            meas = math.atan2(math.sin(a_est - a_ref), math.cos(a_est - a_ref))
            jump = math.atan2(math.sin(meas - self.theta_hat), math.cos(meas - self.theta_hat))
            self.theta_hat = self.theta_hat + self.clo_lp * jump

        return best
