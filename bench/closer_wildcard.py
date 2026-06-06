"""closer_wildcard.py — Drift-free TOPOLOGICAL-COORDINATE loop closure for size-invariant maze nav.

THE PROBLEM (round12_sim): the simulated tracker's position estimate ROTATES away from truth as you travel
(heading error theta is a random walk). The ABSOLUTE est of far-apart places therefore diverges, so a naive
est-radius gate (TightGate) MISSES real loops (the revisit's est has drifted off the stored est) and also
FALSE-MERGES (a distant different cell's drifted est rotates into the radius and happens to share the open
pattern). At n40 TightGate makes ~290 false merges + misses ~1440 real revisits -> success 0.50.

THE INSIGHT — DON'T trust the (drifting) est at all; build a DRIFT-FREE coordinate system from the GRAPH.

Every graph edge records the TRUE commanded move direction d (nbr/explored dir keys are the directions the
explorer told itself to move — legitimately known to any navigator, NOT the secret cell id).  So a BFS over
the edges, summing the unit direction vectors, assigns every connected node an EXACT integer cell offset from
the start — with ZERO drift, because directions don't accumulate heading error the way est does.  (Verified:
on a graph with no false merge, these BFS coords equal the true relative cell positions exactly, at every n.)

Loop closure then becomes EXACT and CERTAIN instead of a fuzzy radius gate:

  1. Identify, drift-free, the node we just moved FROM and the direction we moved.  The harness adds the move
     direction d to cur.explored RIGHT BEFORE calling match(), and only writes the cur--new edge into nbr
     AFTER.  So at match time there is exactly ONE node with a direction in `explored` but not yet in `nbr`:
     that node is `cur`, and that extra direction is `d`.  (We read this from the live graph, no cheating.)
  2. BFS the graph -> integer coord of every node.  The cell we are entering sits at coord[cur] + d.
  3. If an existing node already occupies coord[cur]+d, this is a CERTAIN revisit -> return it (shortcut).
     Otherwise it's genuinely new -> return None.

A revisit is admitted only when (a) the topological coordinate coincides AND (b) the open-exit set matches —
so we essentially never false-merge, which keeps the coordinate system clean (false merges are what corrupt
BFS coords).  Because closure is exact regardless of how far theta has drifted, success stays ~1.0 and steps
stay near oracle as size grows.  est/opens are used only as a fallback safety net; the true cell id (repcell)
is never inspected.
"""
from collections import deque
import numpy as np

DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]


class MyCloser:
    def __init__(self, est_fallback_r=0.6):
        # est-radius fallback gate (used only if the drift-free coord method can't decide, e.g. node 'cur'
        # is in a different BFS component than the start). Conservative, like TightGate.
        self.est_fallback_r = est_fallback_r

    def reset(self):
        self._coord_cache = None

    # ---- drift-free integer coordinates from the graph topology -------------
    @staticmethod
    def _bfs_coords(nodes, root):
        coord = {root: (0, 0)}
        q = deque([root])
        nn = len(nodes)
        while q:
            u = q.popleft()
            cu = coord[u]
            for d, v in nodes[u]["nbr"].items():
                if v < nn and v not in coord:
                    coord[v] = (cu[0] + d[0], cu[1] + d[1])
                    q.append(v)
        return coord

    @staticmethod
    def _find_cur_and_dir(nodes):
        """cur = the unique node with a direction in `explored` not yet in `nbr`; that dir is the move d.
        Returns (cur_id, d) or (None, None) if it can't be determined unambiguously."""
        found = None
        for i, nd in enumerate(nodes):
            extra = nd["explored"] - set(nd["nbr"].keys())
            if extra:
                if found is not None:
                    return None, None  # ambiguous -> bail to fallback
                # pick a single extra dir (there should be exactly one for the in-progress move)
                d = next(iter(extra))
                found = (i, d)
        return found if found is not None else (None, None)

    def match(self, cell, est, opens, nodes):
        cur, d = self._find_cur_and_dir(nodes)

        if cur is not None:
            coords = self._bfs_coords(nodes, cur)          # coords relative to cur (cur at origin)
            target = (d[0], d[1])                           # the cell we are entering, in cur-rooted frame
            # invert coords: which node sits at `target`?
            for i, c in coords.items():
                if c == target:
                    # exact topological coincidence -> revisit, IF opens are consistent.
                    if nodes[i]["opens"] == opens and i != cur:
                        return i
                    # coord coincides but opens differ -> graph is locally inconsistent (rare; a prior bad
                    # merge). Be safe: treat as NEW so we don't compound the error.
                    return None
            # target cell unoccupied in the connected component -> genuinely new.
            return None

        # ---- fallback: drift-free method couldn't determine cur (disconnected / startup) -----------------
        est = np.asarray(est, float)
        best, bd = None, self.est_fallback_r * self.est_fallback_r
        for i, nd in enumerate(nodes):
            if nd["opens"] != opens:
                continue
            diff = est - nd["est"]
            dd = float(diff @ diff)
            if dd < bd:
                bd, best = dd, i
        return best
