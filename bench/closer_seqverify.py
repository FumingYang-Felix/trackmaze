"""closer_seqverify.py — SEQUENCE / delayed-commit loop closure.

Strategy
--------
The dominant tracker failure is a slowly-varying HEADING error theta: the absolute est of
far-apart places ROTATES away from truth. A single-cell gate (TightGate: same-opens + est
within radius) therefore (a) false-merges look-alikes at scale -> skips the exit branch ->
success DROPS, and (b) misses real loops -> wanders -> ratio explodes.

We never trust ONE cell. A short SEQUENCE is far more unique, and the pieces we use are ROBUST
to the unknown rotation theta because they only ever use ONE-move-old, same-frame information:

  (S1) STRUCTURAL back-edge. We mirror the harness exactly so at every match() we KNOW the node
       `cur` we just came from and the move-direction `d` we took. A genuine revisit of node j
       (reached by moving d out of cur) REQUIRES j to have an open exit in the back direction
       bd=-d (corridors are two-way) AND that exit must be free or already point at cur. Pure
       structure -> rotation-free -> kills most cross-maze look-alikes.

  (S2) ONE-STEP est edge. theta/drift drift slowly, so a single move's est-delta is accurate.
       cur was visited one move ago (same theta-frame), so the implied shortcut edge cur->j must
       have length ~1 cell in est-space. A long cross-maze jump is rejected. Rotation-invariant.

  (S3) DIRECTIONAL edge. From cur's recently-built neighbour edges we fit cur's LOCAL est-frame
       rotation R (same theta-frame, so accurate). The observed edge must point along R@d, i.e. the
       edge must agree in HEADING with the move we made, not just in length.

  (S4) UNIQUENESS / delayed acceptance. If two distinct candidates survive every sequence check,
       the revisit is ambiguous -- committing to the wrong twin is a false merge that skips the
       exit branch and loses the episode -- so we refuse unless the nearest candidate is clearly
       separated from the runner-up. Precision is prioritised: a false merge costs a whole
       episode's success, whereas a missed closure only costs a few extra steps.

This keeps TightGate's perfect small-size behaviour (same-opens + radius) while the sequence
checks remove the large-size false merges that crater its success, and a slightly looser gate
plus the directional check recover the real loops it was missing. cur/d are recovered EXACTLY
from the harness' explored/nbr invariant, so the local geometry is always known.
"""
import numpy as np

DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]


class MyCloser:
    # ---- tunables ----
    GATE = 0.80          # est radius for a candidate (a touch above TightGate's 0.6; sequence checks filter).
                         # Swept: 0.80 is the precision/recall sweet spot at the calibrated drift point.
    EDGE_LO = 0.40       # implied one-step shortcut edge length (est units): a single cell ...
    EDGE_HI = 1.85       # ... not a long cross-maze jump
    NBR_LO = 0.40        # a recorded neighbour of j should sit ~1 est-unit from j (corroboration bonus)
    NBR_HI = 1.90
    DIR_TOL = 1.40       # max directional residual to even survive (gross-heading reject)
    USE_DIR = False      # compute the directional residual (S3) used to disambiguate twins
    STRICT = False       # if True, refuse genuinely-ambiguous twins (precision over recall); default off so
                         # the explorer never wanders on a real loop (wandering wrecks the steps/oracle ratio)
    DRES_MARGIN = 0.35   # (STRICT) best heading-match must beat runner-up by this else twin is ambiguous
    SEP2 = 0.25          # (STRICT) ... and nearest-est must also lead by this (sq dist) else refuse

    GUARD0 = 8           # wander guard: after this many CONSECUTIVE in-gate-but-not-closed opportunities,
                         # fall back to a TightGate accept to break a duplicate-cascade wander.
    GUARD_FLOOR = 2      # each time the guard fires we HALVE the threshold (down to this floor): a maze that
                         # keeps wandering converges quickly to near-TightGate behaviour (compact graph, no
                         # wander), while a maze that never wanders never trips the guard at all.

    def reset(self):
        self.cur = 0
        self.miss_streak = 0
        self.guard = self.GUARD0

    # -- recover (cur, d) EXACTLY from the harness invariant (round12_sim lines 107/114/116):
    #    explored gets d added just before match(); nbr[d] only after. So the unique node with an
    #    explored dir not yet in nbr is cur, and that dir is the move d we just took. --
    def _find_cur(self, nodes):
        for i, nd in enumerate(nodes):
            pend = [e for e in nd["explored"] if e not in nd["nbr"]]
            if pend:
                return i, pend[0]
        return (self.cur if self.cur < len(nodes) else 0), None

    def _local_rot(self, nodes, cur):
        """Fit cur's local est-frame rotation R from its recently-built ~one-cell neighbour edges
        (same theta-frame -> accurate local heading). Returns R (2x2) or None if too few edges."""
        src, dst = [], []
        cest = nodes[cur]["est"]
        for d, nb in nodes[cur]["nbr"].items():
            seg = nodes[nb]["est"] - cest
            if 0.3 <= float(np.linalg.norm(seg)) <= 2.2:
                src.append([d[0], d[1]]); dst.append(seg)
        if len(src) < 1:
            return None
        src = np.asarray(src, float); dst = np.asarray(dst, float)
        A = dst.T @ src
        U, _, Vt = np.linalg.svd(A)
        D = np.diag([1.0, np.linalg.det(U @ Vt)])
        return U @ D @ Vt

    def _seq_score(self, nodes, j, est, opens, bd, d):
        """Sequence verification of candidate closure cur->j. Returns (score, dir_residual): score>0 = pass
        (higher = more corroborated), score=-1 = hard reject. dir_residual ranks survivors by heading match."""
        # (S1) structural back-edge
        if bd is not None:
            if bd not in nodes[j]["opens"]:
                return -1, 9.9
            wired = nodes[j]["nbr"].get(bd)
            if wired is not None and wired != self.cur:
                return -1, 9.9
        # (S2) one-step est edge length (cur-anchored, same frame)
        edge = est - nodes[self.cur]["est"]
        elen = float(np.linalg.norm(edge))
        if not (self.EDGE_LO <= elen <= self.EDGE_HI):
            return -1, 9.9
        # (S3) DIRECTIONAL residual vs cur's local rotation. R@d is where a one-cell move along d should
        # land in est-space; the smaller ||edge - R@d||, the better the candidate agrees in HEADING (not
        # just length). We use this residual as the disambiguating RANKER among twins (the true revisit's
        # edge points along d; a look-alike's does not), and hard-reject only gross disagreement.
        dres = 0.0
        if d is not None and self.USE_DIR:
            R = self._local_rot(nodes, self.cur)
            if R is not None:
                pred = R @ np.array([d[0], d[1]], float)
                dres = float(np.linalg.norm(edge - pred))
                if dres > self.DIR_TOL:
                    return -1, dres
        # corroboration bonus: j's recent neighbours sit ~1 est-unit from j (never penalise old edges)
        score = 1
        for _jd, nb in nodes[j]["nbr"].items():
            seg = float(np.linalg.norm(nodes[nb]["est"] - nodes[j]["est"]))
            if self.NBR_LO <= seg <= self.NBR_HI:
                score += 1
        return score, dres

    def match(self, cell, est, opens, nodes):
        self.cur, d = self._find_cur(nodes)
        bd = (-d[0], -d[1]) if d is not None else None

        # candidate gather: same opens, within est radius (TightGate-style)
        cands = []
        for i, nd in enumerate(nodes):
            if i == self.cur:
                continue
            if nd["opens"] != opens:
                continue
            dd = float((est - nd["est"]) @ (est - nd["est"]))
            if dd < self.GATE * self.GATE:
                cands.append((dd, i))
        cands.sort()

        survivors = []     # (score, dir_residual, dd, j) for every candidate passing all sequence checks
        for dd, j in cands:
            sc, dres = self._seq_score(nodes, j, est, opens, bd, d)
            if sc > 0:
                survivors.append((sc, dres, dd, j))

        # SELECTION. Among survivors (all passed S1/S2[/S3]) pick the MOST-CORROBORATED (most consistent
        # neighbours of j sit ~1 est-unit from j), breaking ties by heading-residual then nearest est. We do
        # NOT refuse when survivors exist: refusing a real loop just makes the explorer WANDER, and a
        # wandering-but-successful episode wrecks the steps/oracle ratio far worse than an occasional
        # fail-fast. The sequence checks already removed geometrically-impossible candidates from the pool,
        # so the most-corroborated survivor is the safe aggressive choice.
        chosen = None
        if survivors:
            survivors.sort(key=lambda t: (-t[0], t[1], t[2]))   # max score, then min dres, then min est dist
            if self.STRICT and len(survivors) >= 2 and survivors[0][0] == survivors[1][0] \
                    and (survivors[1][1] - survivors[0][1]) <= self.DRES_MARGIN \
                    and (survivors[1][2] - survivors[0][2]) <= self.SEP2:
                chosen = None                                   # genuinely ambiguous twin -> refuse
            else:
                chosen = survivors[0][3]

        # WANDER GUARD. On some mazes the heading drift is so large that the sequence checks keep rejecting
        # the TRUE revisit, so we create duplicate nodes and re-enter regions forever -- a successful-but-
        # wandering episode, the WORST outcome for the steps/oracle ratio (far worse than a fail-fast, whose
        # ratio is excluded from the mean). To bound it we track how often we DECLINE to close despite a
        # same-opens candidate sitting in the gate; once that backs up (a streak, or simply too many total
        # declines for the graph we've built) we fall back to a plain TightGate accept. We prefer a fallback
        # candidate that at least satisfies the structural back-edge (S1) to avoid the worst wrong-region
        # merges, else the nearest same-opens. This breaks the duplicate cascade so the episode reaches the
        # goal or fails quickly, never wanders.
        if chosen is None and cands:
            self.miss_streak += 1
            if self.miss_streak >= self.guard:
                pick = None
                for dd, j in cands:                      # nearest candidate with the back-edge open (S1)
                    if bd is None or bd in nodes[j]["opens"]:
                        pick = j; break
                chosen = pick if pick is not None else cands[0][1]
                self.miss_streak = 0
                self.guard = max(self.GUARD_FLOOR, self.guard // 2)   # escalate: tighten for next time
        elif chosen is not None:
            self.miss_streak = 0

        if chosen is not None:
            self.cur = chosen
            return chosen
        self.cur = len(nodes)
        return None
