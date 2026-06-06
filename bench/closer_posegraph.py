"""POSE-GRAPH loop closer for round12_sim.

Key idea
--------
The tracker error is dominated by HEADING drift: est = rotate(true_disp, theta) + drift,
where theta is a slow random walk. So the *absolute* est of two far-apart places diverges
(TightGate fails at scale), but the *graph topology* still encodes exact geometry: every
odometry edge nbr[dir] = nid means node nid sits one true cell-step in direction `dir`.

We therefore keep a pose graph:
  - nodes carry a *graph position* gpos = exact integer cell coords, propagated from the
    start along odometry edges (dir keys are TRUE directions, so this is exact and
    rotation-free). This is legitimate graph structure (not oracle place-id).
  - On a candidate closure (current cell vs existing node j), we test TWO things:
      (1) STRUCTURAL: same open-exit pattern (necessary for a real revisit).
      (2) GEOMETRIC POSE-GRAPH RESIDUAL: if we add the loop edge cur<->j, is it consistent?
          We estimate the current accumulated heading theta_hat by least-squares aligning
          recent est-deltas to their known graph-direction deltas, rotate the est-implied
          relative displacement back, and require it to match the graph-implied relative
          displacement (which for a true revisit must be ~0 between cur and j since they are
          the SAME cell). Accept only if the de-rotated residual is small.

Because we de-rotate using a locally fitted heading, this rejects the rotation-induced
false merges that kill the naive absolute-distance gate, while still catching real loops
even when the absolute est has drifted far.
"""
import math
import numpy as np

DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]


class MyCloser:
    def __init__(self, res_tol=0.9, abs_tol=4.0, abs_grow=0.30, snap_tol=1e9,
                 min_anchor=3, win=12, use_abs=False, local_fit=False,
                 heading="procrustes", edge_k=8, struct_prune=True):
        self.use_abs = use_abs
        self.local_fit = local_fit
        self.heading = heading          # "procrustes" | "edges"
        self.edge_k = edge_k
        self.struct_prune = struct_prune
        # res_tol : max single-step-predicted pose-graph residual (cells) to accept a closure.
        # abs_tol : base de-rotated ABSOLUTE est residual to corroborate (+abs_grow*dist).
        # abs_grow: growth of the absolute-residual bound per cell of distance from start.
        # snap_tol: max snap error of the de-rotated single-step delta to a unit dir.
        self.res_tol = res_tol
        self.abs_tol = abs_tol
        self.abs_grow = abs_grow
        self.snap_tol = snap_tol
        self.min_anchor = min_anchor   # need this many est/gpos pairs to fit a heading
        self.win = win                 # number of (nearest) nodes used to fit local heading

    # ------------------------------------------------------------------ #
    def reset(self):
        # gpos[i] = exact integer (gx,gy) graph position of node i (propagated via odometry)
        self.gpos = {0: (0, 0)}
        self.last_id = 0          # node id we are currently sitting on (where last move ended)
        self.n_seen = 0

    # ------------------------------------------------------------------ #
    def _propagate_gpos(self, nodes):
        """Recompute exact graph positions from odometry (nbr) edges via BFS from node 0.
        nbr keys are TRUE directions, so positions are exact integers, rotation-free."""
        gpos = {0: (0, 0)}
        from collections import deque
        q = deque([0])
        while q:
            u = q.popleft()
            ux, uy = gpos[u]
            for d, v in nodes[u]["nbr"].items():
                if v not in gpos:
                    gpos[v] = (ux + d[0], uy + d[1])
                    q.append(v)
        self.gpos = gpos

    def _fit_heading(self, nodes, anchor=None):
        """Least-squares fit of accumulated heading theta from (est, gpos) anchor pairs.
        est_i ~ R(theta) gpos_i + drift.  Solve for R via the 2D Procrustes (rotation) of
        the centered point clouds. Returns (cos, sin) or None if too few/degenerate anchors.
        If `anchor` (a node id) is given, prefer nodes whose graph pos is NEAR it (heading is
        a slow spatial walk, so a local fit around where we are tracks the true local theta)."""
        ids = [i for i in self.gpos if i < len(nodes)]
        if len(ids) < self.min_anchor:
            return None
        if anchor is not None and anchor in self.gpos:
            ax, ay = self.gpos[anchor]
            ids.sort(key=lambda i: (self.gpos[i][0] - ax) ** 2 + (self.gpos[i][1] - ay) ** 2)
            ids = ids[:self.win]
        else:
            # prefer the most recently created nodes (heading is a slow walk -> local fit best)
            ids = sorted(ids)[-self.win:] if len(ids) > self.win else ids
        if len(ids) < self.min_anchor:
            return None
        G = np.array([self.gpos[i] for i in ids], float)       # graph (true) coords
        E = np.array([nodes[i]["est"] for i in ids], float)    # est coords
        Gc = G - G.mean(0)
        Ec = E - E.mean(0)
        if float((Gc * Gc).sum()) < 1e-6:
            return None
        # best rotation aligning G->E: H = Gc^T Ec ; via cross/dot sums for pure 2D rotation
        a = float((Gc[:, 0] * Ec[:, 0] + Gc[:, 1] * Ec[:, 1]).sum())   # dot
        b = float((Gc[:, 0] * Ec[:, 1] - Gc[:, 1] * Ec[:, 0]).sum())   # cross
        nrm = math.hypot(a, b)
        if nrm < 1e-6:
            return None
        return (a / nrm, b / nrm)   # cos, sin of theta (rotation G->E)

    def _fit_heading_edges(self, nodes, anchor=None, K=8):
        """Edge-based local heading: every odometry edge i->j (true dir d) gives an estimate
        theta_edge = angle(est[j]-est[i]) - angle(d). Average (as unit vectors) the theta_edge
        over edges incident to the K graph-nearest nodes to `anchor`. This uses the EXACT edge
        directions (true) and only single-step est deltas, so it tracks the LOCAL heading where
        we currently are -- more robust than a global Procrustes fit over far-apart positions
        that sit at different (drifted) headings."""
        if anchor is None or anchor not in self.gpos:
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

    # ------------------------------------------------------------------ #
    def match(self, cell, est, opens, nodes):
        self.n_seen += 1
        # nodes already includes existing graph; the new cell is NOT yet in nodes.
        # Refresh exact graph positions from current odometry edges.
        self._propagate_gpos(nodes)
        if self.heading == "edges":
            rot = self._fit_heading_edges(nodes, anchor=self.last_id, K=self.edge_k)
            if rot is None:
                rot = self._fit_heading(nodes, anchor=(self.last_id if self.local_fit else None))
        else:
            rot = self._fit_heading(nodes, anchor=(self.last_id if self.local_fit else None))

        # Predict the current cell's TRUE graph position from where we came from (last_id).
        # We came from self.last_id via the move just taken; the harness will set our gpos
        # by odometry only if we are declared NEW, but we can predict it as last_id + step.
        # However we don't know the step dir here directly, so instead de-rotate est into
        # graph frame and round to nearest integer cell as the predicted graph pos.
        # The current cell is exactly ONE true cell-step from self.last_id. Estimate that step
        # direction by de-rotating the est-DELTA from last_id (single-step => minimal extra
        # heading error), then snap to the nearest of the 4 unit dirs. Predicted graph pos is
        # last_id's exact gpos + that unit step.  This is far more robust than de-rotating the
        # absolute est, because consecutive-node heading drift is tiny.
        g_pred = None
        snap_err = 0.0
        if self.last_id in self.gpos and self.last_id < len(nodes):
            d_est = est - nodes[self.last_id]["est"]
            if rot is not None:
                c, s = rot
                # R(-theta) on the delta to bring it into the graph (true) frame
                d_g = np.array([c * d_est[0] + s * d_est[1], -s * d_est[0] + c * d_est[1]])
            else:
                d_g = d_est
            # snap to nearest unit dir; record snap confidence (how cleanly it snapped)
            step = min(DIRS, key=lambda u: (d_g[0] - u[0]) ** 2 + (d_g[1] - u[1]) ** 2)
            snap_err = math.hypot(d_g[0] - step[0], d_g[1] - step[1])
            gl = self.gpos[self.last_id]
            g_pred = (gl[0] + step[0], gl[1] + step[1])

        # Also compute a de-rotated absolute est as a fallback / cross-check.
        if rot is not None:
            c, s = rot
            ids = [i for i in self.gpos if i < len(nodes)]
            ids = sorted(ids)[-self.win:] if len(ids) > self.win else ids
            G = np.array([self.gpos[i] for i in ids], float)
            E = np.array([nodes[i]["est"] for i in ids], float)
            gm = G.mean(0); em = E.mean(0)
            Rg = np.array([c * gm[0] - s * gm[1], s * gm[0] + c * gm[1]])
            drift = em - Rg
            de = est - drift
            g_abs = np.array([c * de[0] + s * de[1], -s * de[0] + c * de[1]])
        else:
            g_abs = est

        # Use the single-step prediction when available (much tighter), else the abs estimate.
        g_cur = np.array(g_pred, float) if g_pred is not None else g_abs

        # Closure rule (pose-graph consistency): a true revisit lands EXACTLY on an existing
        # node's exact graph pos. Search same-opens nodes for the smallest residual to g_cur.
        # We additionally cross-check with the de-rotated ABSOLUTE est (g_abs): for a genuine
        # loop both the single-step prediction and the absolute pose must agree on the node,
        # which kills snap-induced false merges far from start.
        # Structural prune: if we declare cur == j reached by unit step `step` from last_id,
        # then in the TRUE maze j must have an open exit pointing BACK toward last_id (-step),
        # and that back-edge must connect to last_id (or be unexplored). This is rotation-free
        # and rejects wrong-snap merges onto a same-opens node that can't actually be adjacent
        # to last_id in the claimed direction.
        back = None
        if g_pred is not None and self.last_id < len(nodes):
            gl = self.gpos[self.last_id]
            sx, sy = g_pred[0] - gl[0], g_pred[1] - gl[1]
            back = (-sx, -sy)
        best, best_res = None, self.res_tol
        for j, nd in enumerate(nodes):
            if nd["opens"] != opens:
                continue
            if j not in self.gpos:
                continue
            if self.struct_prune and back is not None:
                # j must be structurally open toward last_id (necessary for a real adjacency in
                # the claimed step direction). Rotation-free; prunes impossible wrong-snap merges.
                if back not in nd["opens"]:
                    continue
            gj = np.array(self.gpos[j], float)
            res = float(np.hypot(g_cur[0] - gj[0], g_cur[1] - gj[1]))
            if res < best_res:
                best_res, best = res, j

        if best is not None:
            gj = np.array(self.gpos[best], float)
            abs_res = float(np.hypot(g_abs[0] - gj[0], g_abs[1] - gj[1]))
            # Primary defense against snap-induced false merges is the snap-confidence guard:
            # an exact pose-graph hit on the WRONG node requires the de-rotated single-step
            # delta to have snapped to a wrong unit dir, i.e. a large snap_err. Reject those.
            # Absolute corroboration is kept only as a VERY loose, distance-scaled sanity bound
            # (heading error grows with travel) so it almost never vetoes a true far loop.
            dist = math.hypot(gj[0], gj[1])
            abs_bound = self.abs_tol + self.abs_grow * dist
            abs_ok = (not self.use_abs) or (abs_res <= abs_bound)
            if snap_err <= self.snap_tol and abs_ok:
                self.last_id = best
                return best
        # NEW node: its id will be len(nodes). Its exact gpos = last_id + the snapped unit step
        # (predicted above). This keeps the pose graph exact for connected nodes. We don't rely
        # on this for new ids though: after the harness wires the nbr edge, _propagate_gpos will
        # set it exactly on the next call. We set it here so it's available immediately.
        new_id = len(nodes)
        if g_pred is not None:
            self.gpos[new_id] = g_pred
        else:
            self.gpos[new_id] = (int(round(g_cur[0])), int(round(g_cur[1])))
        self.last_id = new_id
        return None
