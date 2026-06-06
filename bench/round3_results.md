# Round 3: is the GLOBAL limit a MEMORY limit? (the compression question)

Motivation: R2 split size-OOD state-tracking into LOCAL (size-invariant, achieved) and GLOBAL metric
(grows). I called the global growth "pigeonhole / fixed-memory-limited." The user pushed back: what if a
*learned/importance-weighted compression* (weaken unimportant old memories, keep the retraceable ones)
breaks that limit — or does the RNN already compress? Round 3 tests this with training-free oracle
mechanism experiments (oracle place-identity, to isolate the MEMORY question from the recognition question).

Setup: an ORACLE DFS walk drives the agent to COVER the whole maze and genuinely revisit earlier cells
(loop closures) — the scripted forward-explorer spins in place (~6 cells), so it can never stress memory.
TRACKING is open-loop: the agent integrates only its commanded motion → drifting displacement estimate.

## R3-A — greedy snap-to-anchor with bounded memory (env.py loop=0.3, oracle place-id)
Loop closure = on revisiting a stored cell, snap estimate toward the stored (earlier, lower-drift) one.
- **FINAL error (end of walk):** `near` K=60 (keep low-drift near-start anchors) = `unbounded` = ~0.2, FLAT
  13×13→81×81, while `fifo` and `none` blow up. → importance-weighted O(1) memory bounds the *retraceable*
  (return-to-anchor) error; blind forgetting (fifo) fails. **The user's intuition holds for retraceable content.**
- **MEAN error over the WHOLE trajectory:** ALL schemes grow (even unbounded: +50 from 13→81). Greedy
  snap only re-zeros error *relative to the start*; in fresh, never-visited territory there is no anchor and
  greedy snap is not a global optimizer. → the honest metric is mean/everywhere error, and it needs a proper
  global estimator.

## R3-B — BATCH POSE-GRAPH least-squares, full vs compressed (the decisive test)
Nodes = cell-visits; ODOMETRY edges (telescoped dead-reckon between kept nodes) + LOOP-CLOSURE edges
(revisit ⇒ same place ⇒ relative position 0), jointly minimized (translation-only ⇒ x,y decouple into two
weighted-Laplacian solves). Compression = keep only K nodes, marginalize the rest. Fair eval: reconstruct
ALL original nodes (kept anchor's solved position + dead-reckon to it) → same eval set for every variant.

**Heading is a confound:** with commanded-heading integration (`--cmd_heading`), heading does a random walk
(rot_noise=0.09 persistent) and translation-only pose-graph CANNOT fix it → full graph p=1.18, loops p=1.26
(both blow up linearly). So we isolate the memory/compression question with ORACLE heading (translation
drift only — translation pose-graph is then the correct estimator).

Mean reconstructed error vs size, ORACLE heading (power-law exponent p; √-limit = 0.5):

| variant            | n6  | n12 | n20 | n28 | n40 | p     | memory          |
|--------------------|-----|-----|-----|-----|-----|-------|-----------------|
| full (all nodes)   |0.63 |1.07 |1.97 |2.98 |5.04 |**1.09**| N: 141→6397 (∝n²)|
| loops  K=10 (const)|0.39 |0.51 |0.83 |1.20 |1.98 |0.86   | fixed ~12       |
| loops  K=40 (const)|0.48 |0.59 |0.80 |1.14 |1.88 |0.69   | fixed ~42       |
| loops  K=80 (const)|0.63 |0.56 |0.80 |1.04 |1.83 |**0.54**| fixed ~82       |
| loops  K=160(const)|0.63 |0.65 |0.81 |0.99 |1.87 |**0.52**| fixed ~162      |
| fifo   K=80 (const)|0.51 |0.74 |1.40 |2.03 |2.98 |~0.9   | fixed ~82       |

## The findings (the honest answer to the compression question)
1. **Memory is NOT the binding constraint on global accuracy.** A CONSTANT-size (K~80–160, *independent of
   maze size*) importance-weighted pose-graph matches — and here beats — the unbounded O(n²) graph, hitting
   the √ floor (p≈0.5). Global metric structure is compressible to O(1) anchors because the task-relevant
   content (the pose-graph constraints tying the map together) is low-dimensional. **My earlier "pigeonhole /
   fixed-memory-limited" claim was wrong** — the user was right to push.
2. **The residual growth (~√size) is the FUNDAMENTAL single-gauge-anchor limit, not memory.** Error
   accumulates with constraint-distance to the one fixed reference; gauge freedom forbids more *absolute*
   anchors. Infinite memory (full) does NOT beat √ — it does worse here. Navigation doesn't need a globally
   accurate metric coordinate; it needs this √-consistent *topological/relational* map, which IS achieved.
3. **WHICH memories you keep matters: importance/spread ≫ fifo.** Blind recency-forgetting (fifo — what a
   plain RNN's fixed state effectively does) degrades with size; keeping a spatially/topologically
   representative anchor set stays flat. So "the RNN already compresses" is true but it compresses *badly*
   (indiscriminately); a structured importance-weighted memory is strictly better — confirming the user's
   "weaken unimportant, keep the retraceable" intuition.

## The redirection (what this means for the novel method)
- Memory compression per se is **already solved** (classical pose-graph sparsification / generic node
  removal, Kretzschmar–Stachniss 2011; the √ floor is Olson/Dellaert). R3 confirms it in this bench. So the
  novel-method effort should NOT go to "compress memory."
- The genuinely open, hard part — which R3 deliberately CHEATED with oracle place-identity and which R2-B
  failed at — is **learned, vocabulary-independent, aliasing-robust loop-closure RECOGNITION** (detect "I've
  been here" from observations when landmark ids alias), feeding a bounded importance-weighted memory. Plus
  **observation-based heading correction** (the R2 allo-canonical view), since heading is the dominant
  separate bottleneck translation memory can't touch.
- Round 4 target: a LEARNED memory-augmented tracker = allo-canonical observation encoder (heading-invariant
  place descriptor, R2) + bounded slot memory with learned write/importance + contrastive place recognition
  for loop closure. Train on 13×13, eval to 105×105; measure global error vs size against a plain-RNN
  baseline. The mechanism tests predict it CAN reach the √ floor with O(1) memory — IF the learned loop
  closure is reliable under aliasing. That reliability is the real research bet.
