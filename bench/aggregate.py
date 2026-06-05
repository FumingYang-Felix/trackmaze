"""Aggregate res_*.json -> arch x size comparison, split by whether landmarks were provided.
The un-confounded re-anchoring test: does WITH-landmark beat NO-landmark? (positive benefit = it
actually uses landmarks to correct drift). Ceiling = oracle assoc ~0.4; floor = open-loop per size."""
import glob, json, os, numpy as np
from collections import defaultdict

CACHE = os.path.join(os.path.dirname(__file__), "_cache")

def get(r, n, k):
    bs = r["by_size"]; key = str(n) if str(n) in bs else n; return bs[key][k]

def main():
    runs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(CACHE, "res_*.json")))]
    if not runs: print("no results yet"); return
    sizes = sorted({int(n) for r in runs for n in r["by_size"]})
    ol = {n: np.mean([get(r, n, "open_final") for r in runs]) for n in sizes}
    grp = defaultdict(list)
    for r in runs: grp[(r["arch"], r.get("landmarks", True))].append(r)

    print(f"\n{'='*78}\nFINAL error (cells), mean±std over seeds   ceiling oracle~0.40")
    hdr = "arch / landmarks".ljust(22) + "".join(f"n={n}".rjust(13) for n in sizes); print(hdr); print("-"*len(hdr))
    for (arch, lm) in sorted(grp):
        rs = grp[(arch, lm)]; row = f"{arch} {'[+lm]' if lm else '[-lm]'}".ljust(22)
        for n in sizes:
            v = [get(r, n, "final") for r in rs]; row += f"{np.mean(v):.2f}±{np.std(v):.2f}".rjust(13)
        print(row)
    print("open-loop (floor)".ljust(22) + "".join(f"{ol[n]:.2f}".rjust(13) for n in sizes))

    # landmark benefit = no-lm final  -  with-lm final  (positive => landmarks actually help)
    archs = sorted({a for a,_ in grp})
    if any((a, True) in grp and (a, False) in grp for a in archs):
        print(f"\nLANDMARK BENEFIT  (no-lm minus with-lm; >0 = re-anchors, ~0 = ignores landmarks)")
        for a in archs:
            if (a, True) in grp and (a, False) in grp:
                w = {n: np.mean([get(r, n, "final") for r in grp[(a, True)]]) for n in sizes}
                wo = {n: np.mean([get(r, n, "final") for r in grp[(a, False)]]) for n in sizes}
                print(f"  {a.ljust(12)} " + " ".join(f"n{n}:{wo[n]-w[n]:+.2f}" for n in sizes))

if __name__ == "__main__":
    main()
