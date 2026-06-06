"""RELATIVE-FRAME loop closure for size-invariant maze navigation.

The dominant failure of the substrate's simulated tracker is HEADING drift: the
position est ROTATES away from truth as you travel, so the ABSOLUTE est of
far-apart places diverges and naive absolute gating (TightGate) false-merges or
misses real loops at scale.

Relative-frame insight, pushed to its limit: the est between two nodes RELATIVE
to a graph path has little accumulated rotation -- and the PERFECTLY relative
frame is the graph itself.  The online graph stores, for every edge, the TRUE
cardinal direction it was traversed (nbr:{dir:node_id}); the agent commanded
those moves, so the directions are exact and the integer cell-grid does not
drift (only the continuous est does).  Summing the cardinal directions along the
BFS path between two nodes therefore gives their EXACT relative displacement,
with ZERO accumulated heading error AT ANY PATH LENGTH.

Gate: when we step one cardinal cell `d` from the current node `cur` into a new
cell, a candidate existing node `j` is the SAME true place iff (a) its open-exit
pattern matches AND (b) the exact graph displacement cur->j equals the step `d`.
Because both sides are pure true geometry, this closes arbitrarily LONG loops
without ever trusting the rotated est, and it cannot false-merge when correctly
anchored (only one true cell sits at cur + d).  The drifted est is used only to
disambiguate the current anchor / step when the graph alone is ambiguous, and as
a tie-break among geometry-valid candidates -- both SHORT-path quantities with
little rotation.  No absolute-est gating anywhere, so it is size-invariant.
"""
import math
import numpy as np
from collections import deque

DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]


class MyCloser:
    def __init__(self, geom_tol=0.25):
        # geom_tol: tolerance on the EXACT graph-vs-step geometry check (cells).
        # Graph displacements are integer cardinal sums, so anything < 0.5 admits
        # only the exact integer match and rejects every non-coincident offset.
        self.geom_tol = geom_tol

    def reset(self):
        pass

    def _find_cur(self, nodes, est):
        """The harness can backtrack between match() calls WITHOUT telling us, so we
        cannot track cur incrementally.  At match() time the agent has just stepped
        direction d FROM the current node into the new cell: the harness does
        node['explored'].add(d) BEFORE driving but assigns node['nbr'][d] only AFTER
        match returns.  So the current node is a node with some explored direction not
        yet in its nbr.  Several nodes can satisfy this (earlier partially-explored
        frontiers), so among those candidates we pick the one whose est is closest to
        ONE CARDINAL STEP from the new est (the agent really did just step one cell
        from cur), which uniquely identifies the true anchor and prevents the wrong
        anchor from producing a spurious exact-geometry false merge."""
        cands = [i for i, nd in enumerate(nodes)
                 if nd["explored"] - set(nd["nbr"].keys())]
        if not cands:
            return len(nodes) - 1
        if len(cands) == 1:
            return cands[0]
        best, bd = cands[-1], math.inf
        for i in cands:
            extra = nodes[i]["explored"] - set(nodes[i]["nbr"].keys())
            de = est - nodes[i]["est"]
            # distance to the nearest candidate cardinal direction this node could
            # have just been exited along
            md = min(float(np.hypot(de[0] - dd[0], de[1] - dd[1])) for dd in extra)
            if md < bd:
                bd, best = md, i
        return best

    def _cur_step_dir(self, nodes, cur, est):
        """The cardinal direction just stepped from cur (explored but not yet in nbr).
        If cur has several such pending directions, pick the one best matching the est
        displacement of the new cell from cur (the move we actually just made)."""
        extra = nodes[cur]["explored"] - set(nodes[cur]["nbr"].keys())
        if not extra:
            return None
        if len(extra) == 1:
            return next(iter(extra))
        de = est - nodes[cur]["est"]
        return min(extra, key=lambda dd: float(np.hypot(de[0] - dd[0], de[1] - dd[1])))

    # ----- graph helpers (operate on the TRUE-direction nbr edges) -----
    def _graph_disp(self, nodes, src, dst):
        """Exact relative true displacement src->dst by summing cardinal nbr
        directions along the shortest nbr path.  Returns (np.array(2,), pathlen)
        or (None, inf) if unreachable / too far."""
        if src == dst:
            return np.zeros(2), 0
        prev = {src: None}
        pdir = {src: None}
        q = deque([src])
        while q:
            u = q.popleft()
            for d, nb in nodes[u]["nbr"].items():
                if nb not in prev:
                    prev[nb] = u
                    pdir[nb] = d
                    if nb == dst:
                        # reconstruct
                        disp = np.zeros(2)
                        plen = 0
                        v = dst
                        while prev[v] is not None:
                            dd = pdir[v]
                            disp += np.array([dd[0], dd[1]], float)
                            plen += 1
                            v = prev[v]
                        return disp, plen
                    q.append(nb)
        return None, math.inf

    def match(self, cell, est, opens, nodes):
        cur = self._find_cur(nodes, est)
        d = self._cur_step_dir(nodes, cur, est)

        # If we cannot identify the exact cardinal step that brought us here, we have
        # no trustworthy true-geometry signal -> stay safe and declare NEW (a missed
        # loop only costs steps; a false merge can skip the exit and fail completeness).
        if d is None:
            return None

        e_cur = nodes[cur]["est"]
        # the est-relative displacement of the NEW cell from the current node
        rel_new = est - e_cur
        # the EXACT cardinal step we just took
        step = np.array([d[0], d[1]], float)

        # PRIMARY: exact-geometry candidates.  If the new cell nc is a revisit of an
        # existing node j, then the EXACT graph displacement cur->j (sum of cardinal
        # nbr directions; ZERO drift at any path length) must equal the cardinal step
        # we just took (cur->nc).  Both sides are pure true geometry, so this closes
        # arbitrarily LONG loops without ever trusting the drifted est.  Among the
        # geometrically-valid candidates (there can be several with the same opens at
        # the same relative offset, in a braided maze) we break ties with the est
        # residual -- a SHORT-path relative-frame quantity with little rotation.
        geom_best = None
        geom_best_res = math.inf
        for j, nd in enumerate(nodes):
            if nd["opens"] != opens or j == cur:
                continue
            g, plen = self._graph_disp(nodes, cur, j)
            if g is None:
                continue
            geom_res = float(np.hypot(step[0] - g[0], step[1] - g[1]))
            if geom_res > self.geom_tol:
                continue  # graph says j is not where we stepped -> not this revisit
            # tie-break by est agreement (and prefer the shorter loop, lower drift)
            res_vec = rel_new - g
            res = float(np.hypot(res_vec[0], res_vec[1])) + 0.001 * plen
            if res < geom_best_res:
                geom_best_res = res
                geom_best = j
        # No EXACT-geometry candidate -> declare NEW.  We deliberately do NOT fall back
        # to comparing drifted absolute/rotated est: that heuristic produces false
        # merges (it once merged a true (1,1) into a node whose exact graph offset was
        # (-1,-1) while we had only stepped (-1,0)), which corrupt the graph and skip
        # the exit branch.  A missed loop only costs steps; completeness is preserved.
        return geom_best
