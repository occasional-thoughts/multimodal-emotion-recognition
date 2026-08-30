"""Collect results and test whether cross vs self is a REAL difference or noise.
Rajan et al. used a two-tailed t-test; we do the same so the comparison is like-for-like.

  python analyze.py --dir results
"""
import json, glob, argparse, collections
import numpy as np
from scipy import stats

p = argparse.ArgumentParser(); p.add_argument("--dir", default="results"); a = p.parse_args()

runs = collections.defaultdict(list)      # (fusion, features) -> list of per-fold WA/UA
for f in sorted(glob.glob(f"{a.dir}/*.json")):
    r = json.load(open(f))
    runs[(r["fusion"], r["features"])].append(r)

if not runs:
    raise SystemExit(f"no result files in {a.dir}/ - run train.py first")

print(f"{'fusion':12} {'features':11} {'WA %':>15} {'UA %':>15}  seeds")
print("-" * 62)
agg = {}
for k in sorted(runs):
    wa = np.concatenate([r["wa"] for r in runs[k]]) * 100   # every fold, every seed
    ua = np.concatenate([r["ua"] for r in runs[k]]) * 100
    agg[k] = (wa, ua)
    print(f"{k[0]:12} {k[1]:11} {wa.mean():7.2f} +/-{wa.std():5.2f} "
          f"{ua.mean():7.2f} +/-{ua.std():5.2f}  {len(runs[k])}")

print("\n" + "=" * 62)
print("CROSS vs SELF  (two-tailed t-test, as in Rajan et al.)")
print("=" * 62)
for feats in ["mfcc", "mfcc+gfcc"]:
    c, s = agg.get(("cross", feats)), agg.get(("self", feats))
    if not (c and s):
        print(f"  {feats:10} - need both cross and self runs"); continue
    for name, i in [("WA", 0), ("UA", 1)]:
        t, pv = stats.ttest_ind(c[i], s[i])
        d = c[i].mean() - s[i].mean()
        verdict = ("cross better" if d > 0 else "self better") if pv < 0.05 else "STATISTICALLY COMPARABLE"
        print(f"  {feats:10} {name}:  cross {c[i].mean():6.2f}  self {s[i].mean():6.2f}  "
              f"diff {d:+5.2f}  p={pv:.4f}  -> {verdict}")

print("\n" + "=" * 62)
print("DOES GFCC HELP?  (mfcc+gfcc vs mfcc)")
print("=" * 62)
for fu in ["cross", "self", "audio_only"]:
    m, g = agg.get((fu, "mfcc")), agg.get((fu, "mfcc+gfcc"))
    if not (m and g): continue
    t, pv = stats.ttest_ind(g[0], m[0])
    print(f"  {fu:11} WA: {m[0].mean():6.2f} -> {g[0].mean():6.2f}  "
          f"({g[0].mean()-m[0].mean():+5.2f})  p={pv:.4f}"
          f"{'  *significant*' if pv < 0.05 else ''}")
