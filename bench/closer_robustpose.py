"""closer_robustpose.py -- robust pose-graph loop closer for round12_sim.

Goal: keep success ~1.0 AND steps near oracle as size grows, at the calibrated heading-drift
op point, beating TightGate (0.819) and being MORE ROBUST at scale than the posegraph closer
(whose n40 success drops to ~0.67 on independent seeds).

Core (shared with posegraph): the tracker error is dominated by a slowly-varying HEADING error
theta. The *absolute* est rotates away from truth, so a naive absolute-distance gate fails at
scale, but the GRAPH TOPOLOGY is exact: nbr[dir]=nid means nid is exactly one true cell-step in
true direction `dir`. So we keep an EXACT integer graph position gpos per node, propagated from
node 0 along odometry edges (rotation-free, oracle-free). A genuine revisit lands EXACTLY on an
existing node's integer gpos.

KEY IMPROVEMENT over posegraph
------------------------------
posegraph predicts the current cell's gpos from `self.last_id` (the last node it returned). But
the harness only calls match() on FORWARD frontier-expansion moves; during backtracking it moves
its internal `cur` along the frontier path WITHOUT calling match(). So `last_id` desyncs from the
node we actually just moved out of, and posegraph's single-step prediction (and its back-edge
prune) are then computed from the WRONG predecessor -> occasional bad snaps / missed-or-false
closures that crater whole episodes at scale.

We instead RECOVER the true predecessor exactly from the harness invariant (round12_sim lines
107/114/116): `explored` gets the move dir `d` added just BEFORE match(); `nbr[d]` is wired only
AFTER. So the unique node with an explored dir not yet in `nbr` is exactly the node we came from
(`cur`) and that dir is the move `d` we just took. With the correct `cur`+`d` we get:

  - g_pred  = exact gpos[cur] + de-rotated single unit step (tight, same local theta-frame).
  - back    = -step : a real revisit j must have an open exit toward cur (rotation-free prune).

Acceptance cues (all rotation-robust):
  (C1) same open-exit pattern (necessary).
  (C2) structural back-edge: j open toward cur.
  (C3) pose-graph residual: g_pred == j's exact integer gpos (res < res_tol).
  (C4) uniqueness: if two same-opens nodes tie on residual, refuse (ambiguous twin -> a false
       merge would skip the exit branch and lose the whole episode; a missed loop only costs a
       few steps). Precision-first is what raises success at scale.

A WANDER GUARD bounds the cost of being conservative: after enough declined-but-valid candidates
it accepts the nearest structurally-valid one, so a hard maze converges to compact behaviour
instead of looping forever (which would explode the steps/oracle ratio).
"""
import math
from collections import deque
import numpy as np

DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]


class MyCloser:
    def __init__(self,
                 res_tol=0.9,       # max single-step pose-graph residual (cells) to accept a closure
                 uniq_margin=0.3,   # runner-up residual must beat winner by this, else ambiguous -> refuse
                 snap_tol=0.7,      # max snap error of the de-rotated single-step delta to a unit dir.
                                    # A FAR wrong merge needs a bad snap; with the correct predecessor
                                    # the single-step delta is in the same local theta-frame so a clean
                                    # snap is reliable -> this rejects most far false merges (the n40 killer).
                 abs_tol=3.5,       # base de-rotated ABSOLUTE est residual bound (corroboration)
                 abs_grow=0.30,     # growth of the abs bound per cell of distance from start (heading
                                    # error accumulates with travel -> loose far so true loops aren't vetoed)
                 edge_k=10,         # edges incident to the K graph-nearest nodes for the local heading fit
                 win=14,            # nearest nodes used by the Procrustes fallback heading fit
                 min_anchor=3,      # min (est,gpos) pairs to fit a heading
                 guard0=8,          # declined-but-valid streak before the wander-guard relaxes
                 guard_floor=2):
        self.res_tol = res_tol
        self.uniq_margin = uniq_margin
        self.snap_tol = snap_tol
        self.abs_tol = abs_tol
        self.abs_grow = abs_grow
        self.edge_k = edge_k
        self.win = win
        self.min_anchor = min_anchor
        self.guard0 = guard0
        self.guard_floor = guard_floor

    # ------------------------------------------------------------------ #
    def reset(self):
        self.gpos = {0: (0, 0)}
        self.miss_streak = 0
        self.guard = self.guard0

    # ------------------------------------------------------------------ #
    def _find_cur(self, nodes):
        """Recover the predecessor node `cur` and move dir `d` EXACTLY from the harness invariant:
        the unique node with an explored dir not yet present in nbr is the node we came from, and
        that dir is the move we just took (round12_sim lines 107/114/116)."""
        for i, nd in enumerate(nodes):
            pend = [e for e in nd["explored"] if e not in nd["nbr"]]
            if pend:
                return i, pend[0]
        return (len(nodes) - 1 if nodes else 0), None

    def _propagate_gpos(self, nodes):
        """Exact integer graph positions from odometry (nbr) edges via BFS from node 0.
        nbr keys are TRUE directions -> positions are exact integers, rotation-free."""
        gpos = {0: (0, 0)}
        q = deque([0])
        while q:
            u = q.popleft()
            ux, uy = gpos[u]
            for d, v in nodes[u]["nbr"].items():
                if v not in gpos:
                    gpos[v] = (ux + d[0], uy + d[1])
                    q.append(v)
        self.gpos = gpos

    def _fit_heading_edges(self, nodes, anchor, K):
        """Local edge-based heading: each odometry edge i->j (true dir d) gives
        theta_edge = angle(est[j]-est[i]) - angle(d). Average (as unit vectors) over edges
        incident to the K graph-nearest nodes to `anchor`. Uses EXACT edge dirs + single-step est
        deltas, so it tracks the LOCAL heading where we currently are."""
        if anchor not in self.gpos:
            ids = sorted(i for i in self.gpos if i < len(nodes))
            if not ids:
                return None
            anchor = ids[-1]
        ax, ay = self.gpos[anchor]
        ids = sorted((i for i in self.gpos if i < len(nodes)),
                     key=lambda i: (self.gpos[i][0] - ax) ** 2 + (self.gpos[i][1] - ay) ** 2)[:K]
        sx = sy = 0.0
        for i in ids:
            for d, j in nodes[i]["nbr"].items():
                if j >= len(nodes):
                    continue
                de = nodes[j]["est"] - nodes[i]["est"]
                m = math.hypot(de[0], de[1])
                if m < 1e-6:
                    continue
                ang = math.atan2(de[1], de[0]) - math.atan2(d[1], d[0])
                sx += math.cos(ang); sy += math.sin(ang)
        if abs(sx) < 1e-9 and abs(sy) < 1e-9:
            return None
        nrm = math.hypot(sx, sy)
        return (sx / nrm, sy / nrm)   # cos, sin of theta

    def _fit_heading_procrustes(self, nodes):
        """Global 2D Procrustes fallback: est_i ~ R(theta) gpos_i + drift. Returns (cos,sin)."""
        ids = [i for i in self.gpos if i < len(nodes)]
        if len(ids) < self.min_anchor:
            return None
        ids = sorted(ids)[-self.win:] if len(ids) > self.win else ids
        if len(ids) < self.min_anchor:
            return None
        G = np.array([self.gpos[i] for i in ids], float)
        E = np.array([nodes[i]["est"] for i in ids], float)
        Gc = G - G.mean(0); Ec = E - E.mean(0)
        if float((Gc * Gc).sum()) < 1e-6:
            return None
        a = float((Gc[:, 0] * Ec[:, 0] + Gc[:, 1] * Ec[:, 1]).sum())
        b = float((Gc[:, 0] * Ec[:, 1] - Gc[:, 1] * Ec[:, 0]).sum())
        nrm = math.hypot(a, b)
        if nrm < 1e-6:
            return None
        return (a / nrm, b / nrm)

    def _abs_in_graph(self, nodes, rot, est):
        """De-rotate the ABSOLUTE est into the graph frame using a windowed drift estimate, so it
        can be compared to a candidate's exact integer gpos as a distance-scaled corroboration."""
        if rot is None:
            return np.asarray(est, float)
        c, s = rot
        ids = [i for i in self.gpos if i < len(nodes)]
        ids = sorted(ids)[-self.win:] if len(ids) > self.win else ids
        if not ids:
            return np.asarray(est, float)
        G = np.array([self.gpos[i] for i in ids], float)
        E = np.array([nodes[i]["est"] for i in ids], float)
        gm = G.mean(0); em = E.mean(0)
        Rg = np.array([c * gm[0] - s * gm[1], s * gm[0] + c * gm[1]])
        drift = em - Rg
        de = np.asarray(est, float) - drift
        return np.array([c * de[0] + s * de[1], -s * de[0] + c * de[1]])

    # ------------------------------------------------------------------ #
    def match(self, cell, est, opens, nodes):
        cur, d = self._find_cur(nodes)
        self._propagate_gpos(nodes)

        rot = self._fit_heading_edges(nodes, cur, self.edge_k)
        if rot is None:
            rot = self._fit_heading_procrustes(nodes)

        # single-step prediction of current gpos from the TRUE predecessor cur (tight, same frame)
        g_pred = None
        step = None
        snap_err = 9.9
        if cur in self.gpos and cur < len(nodes):
            d_est = np.asarray(est, float) - np.asarray(nodes[cur]["est"], float)
            if rot is not None:
                c, s = rot
                d_g = np.array([c * d_est[0] + s * d_est[1], -s * d_est[0] + c * d_est[1]])
            else:
                d_g = d_est
            step = min(DIRS, key=lambda u: (d_g[0] - u[0]) ** 2 + (d_g[1] - u[1]) ** 2)
            snap_err = math.hypot(d_g[0] - step[0], d_g[1] - step[1])
            gl = self.gpos[cur]
            g_pred = (gl[0] + step[0], gl[1] + step[1])

        g_cur = np.array(g_pred, float) if g_pred is not None else np.asarray(est, float)
        g_abs = self._abs_in_graph(nodes, rot, est)
        back = (-step[0], -step[1]) if step is not None else None

        # gather same-opens, structurally-valid, pose-consistent candidates
        survivors = []  # (res, j)
        for j, nd in enumerate(nodes):
            if j == cur:
                continue
            if nd["opens"] != opens:
                continue
            if j not in self.gpos:
                continue
            if back is not None and back not in nd["opens"]:
                continue
            gj = np.array(self.gpos[j], float)
            res = float(np.hypot(g_cur[0] - gj[0], g_cur[1] - gj[1]))
            if res < self.res_tol:
                survivors.append((res, j))
        survivors.sort()

        chosen = None
        if survivors:
            res0, j0 = survivors[0]
            # (C4) uniqueness: ambiguous twin -> refuse (a false merge loses the whole episode)
            ambiguous = len(survivors) >= 2 and (survivors[1][0] - res0) < self.uniq_margin
            if not ambiguous:
                # snap-confidence + distance-scaled absolute corroboration: a FAR wrong merge
                # needs a bad single-step snap AND/OR a de-rotated absolute est far from j's gpos.
                # With the correct predecessor the single-step delta is reliable, so a clean snap +
                # loose-far absolute agreement confidently confirms the closure and kills the
                # n40-killing far false merges, while staying loose enough for true far loops.
                gj = np.array(self.gpos[j0], float)
                abs_res = float(np.hypot(g_abs[0] - gj[0], g_abs[1] - gj[1]))
                dist = math.hypot(gj[0], gj[1])
                abs_bound = self.abs_tol + self.abs_grow * dist
                if snap_err <= self.snap_tol and abs_res <= abs_bound:
                    chosen = j0

        # wander guard: bound the cost of being conservative
        if chosen is None and survivors:
            self.miss_streak += 1
            if self.miss_streak >= self.guard:
                chosen = survivors[0][1]
                self.miss_streak = 0
                self.guard = max(self.guard_floor, self.guard // 2)
        elif chosen is not None:
            self.miss_streak = 0

        if chosen is not None:
            return chosen

        new_id = len(nodes)
        self.gpos[new_id] = g_pred if g_pred is not None else \
            (int(round(g_cur[0])), int(round(g_cur[1])))
        return None
