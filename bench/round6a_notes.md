# Round 6-A: can the maze's Manhattan grid bound heading drift? (partly — mod 90° only)

R3/R5 showed HEADING drift is the dominant horn of the loop-closure vicious cycle (cmd-heading global error
p≈1.2; it loosens the metric gate). Tested whether the maze's axis-aligned walls give an observation-based
absolute heading reference.

**Finding (positive, but bounded):** the freq-4 Fourier phase of the 32-ray omni distance signal estimates
heading **mod 90°** with a circular std of **~20°, perfectly size-invariant** (19.7–21.0° from 13×13 to
81×81), zero bias. So the maze DOES contain a drift-free, size-invariant heading anchor — but only mod 90°.

**Finding (the wall):** the remaining **4-fold (quadrant) ambiguity is irreducible from local wall structure**
(the maze grid is 4-fold rotationally symmetric — the four orientations look identical locally). Resolving it
needs the quadrant carried by command integration, which random-walks (rot_noise=0.09 → uniformly random in
a few hundred steps). A complementary filter (cmd quadrant + grid mod-90) fails: even at rot_noise=0.02 the
fused absolute heading is 57–111° RMS with quadrant-correct fraction ~0.2–0.6 at large sizes — over a long
maze-covering trajectory, slow drift still randomizes the quadrant and the mod-90 anchor can't fix it.

**Conclusion:** heading mod 90° is boundable from observation (size-invariant); ABSOLUTE heading is not,
without a persistent global cue. The only persistent cue available is **loop closure** (recognizing a place
re-anchors both position AND heading) or a global landmark/boundary. So heading and loop closure are the same
problem: reliable place recognition re-anchors heading for free. Everything converges on **aliasing-robust
loop closure** as the single open core.

**Next (Round 6+):** stop treating heading/position separately. Build the SE(2) loop-closure back-end with
ROBUST (consistency-checked / RANSAC-style) data association: propose closures from the learned descriptor,
accept only the maximal geometrically-consistent set (a false closure between two look-alike places is
inconsistent with the rest of the trajectory and gets rejected). This is the standard robust-SLAM answer to
aliasing and is what R5's naive per-pair gating lacked. A reliable closure simultaneously fixes the √-floor
position drift AND the heading quadrant. Test the full pipeline (size-invariant local tracker + descriptor +
O(1) memory + robust SE(2) closure) for bounded error 13×13→105×105 vs the memorizing baselines.
