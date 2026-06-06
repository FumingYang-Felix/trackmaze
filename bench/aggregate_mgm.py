"""Aggregate results_<arch>_<seed>.txt (lines: arch seed n local global) -> mean+/-std over seeds per
(arch, size). Prints LOCAL and GLOBAL OOD-error tables and the size-growth per arch."""
import glob, numpy as np
from collections import defaultdict

rows = defaultdict(lambda: defaultdict(list))   # arch -> n -> [(local,global), ...]
for f in glob.glob("results_*.txt"):
    for line in open(f):
        p = line.split()
        if len(p) == 5:
            arch, seed, n, lo, gl = p
            rows[arch][int(n)].append((float(lo), float(gl)))

archs = [a for a in ["mgm", "mgm_norot", "mgm_nomem", "gru", "transformer"] if a in rows]
sizes = sorted({n for a in rows for n in rows[a]})

def cell(arch, n, idx):
    v = [x[idx] for x in rows[arch].get(n, [])]
    return (np.mean(v), np.std(v), len(v)) if v else (np.nan, np.nan, 0)

for metric, idx in [("LOCAL (windowed rel-displacement = navigable accuracy)", 0), ("GLOBAL (final displacement)", 1)]:
    print(f"\n=== {metric} : mean+/-std over seeds, vs size (n6,12 = train; rest = OOD) ===")
    print(f"{'n':>4} {'grid':>9} | " + " | ".join(f"{a:>16}" for a in archs))
    for n in sizes:
        tag = "*tr*" if n in (6, 12) else ""
        seg = " | ".join(f"{cell(a,n,idx)[0]:6.2f}+-{cell(a,n,idx)[1]:4.2f}" for a in archs)
        print(f"{n:>4} {f'{2*n+1}x{2*n+1}':>9}{tag:>0} | {seg}")
    print("growth (smallest->largest eval size):")
    for a in archs:
        s0, s1 = sizes[0], sizes[-1]
        print(f"  {a:>12}: {cell(a,s0,idx)[0]:.2f} -> {cell(a,s1,idx)[0]:.2f}  ({cell(a,s1,idx)[0]-cell(a,s0,idx)[0]:+.2f})  (n_seeds={cell(a,s1,idx)[2]})")
