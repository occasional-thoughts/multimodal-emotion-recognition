"""Collect results and test whether cross vs self is a REAL difference or noise.
Rajan et al. used a two-tailed t-test; we do the same so the comparison is like-for-like.

Runs that did not finish all 5 folds are reported but EXCLUDED from the statistics:
pooling a 1-fold run with a 5-fold run silently biases the mean toward whichever
fold happened to finish.

  python analyze.py --dir results
"""
import json, glob, argparse, collections
import numpy as np
from scipy import stats

p = argparse.ArgumentParser()
p.add_argument("--dir", default="results")
p.add_argument("--folds", type=int, default=5, help="folds a complete run must have")
a = p.parse_args()

runs, partial = collections.defaultdict(list), []
for f in sorted(glob.glob(f"{a.dir}/*.json")):
    r = json.load(open(f))
    if len(r["wa"]) >= a.folds:
        runs[(r["fusion"], r["features"])].append(r)
    else:
        partial.append(r)

if not runs:
    raise SystemExit(f"no complete result files in {a.dir}/ - run train.py first")

print(f"{'fusion':12} {'features':11} {'WA %':>15} {'UA %':>15}  seeds")
print("-" * 62)
agg, byseed = {}, {}
for k in sorted(runs):
    wa = np.concatenate([r["wa"] for r in runs[k]]) * 100   # every fold, every complete seed
    ua = np.concatenate([r["ua"] for r in runs[k]]) * 100
    agg[k] = (wa, ua)
    for r in runs[k]:
        byseed[(k[0], k[1], r["seed"])] = (np.array(r["wa"]) * 100, np.array(r["ua"]) * 100)
    print(f"{k[0]:12} {k[1]:11} {wa.mean():7.2f} +/-{wa.std():5.2f} "
          f"{ua.mean():7.2f} +/-{ua.std():5.2f}  {sorted(r['seed'] for r in runs[k])}")

if partial:
    print("\nEXCLUDED - incomplete runs (fewer than %d folds):" % a.folds)
    for r in partial:
        print(f"  {r['fusion']:11} {r['features']:10} seed {r['seed']}: "
              f"{len(r['wa'])}/{a.folds} folds, WA {np.mean(r['wa'])*100:.2f}")

print("\n" + "=" * 62)
print("CROSS vs SELF  (two-tailed t-test, as in Rajan et al.)")
print("within a seed, so training conditions are matched")
print("=" * 62)
seeds = sorted({k[2] for k in byseed})
for feats in ["mfcc", "mfcc+gfcc"]:
    for sd in seeds:
        c, s = byseed.get(("cross", feats, sd)), byseed.get(("self", feats, sd))
        if not (c and s):
            continue
        for name, i in [("WA", 0), ("UA", 1)]:
            t, pv = stats.ttest_ind(c[i], s[i])
            d = c[i].mean() - s[i].mean()
            verdict = ("cross better" if d > 0 else "self better") if pv < 0.05 else "STATISTICALLY COMPARABLE"
            print(f"  {feats:10} seed {sd} {name}:  cross {c[i].mean():6.2f}  self {s[i].mean():6.2f}  "
                  f"diff {d:+5.2f}  p={pv:.4f}  -> {verdict}")

print("\n" + "=" * 62)
print("DOES GFCC HELP?  (mfcc+gfcc vs mfcc, within a seed)")
print("=" * 62)
for fu in ["cross", "self", "audio_only"]:
    for sd in seeds:
        m, g = byseed.get((fu, "mfcc", sd)), byseed.get((fu, "mfcc+gfcc", sd))
        if not (m and g): continue
        t, pv = stats.ttest_ind(g[0], m[0])
        print(f"  {fu:11} seed {sd} WA: {m[0].mean():6.2f} -> {g[0].mean():6.2f}  "
              f"({g[0].mean()-m[0].mean():+5.2f})  p={pv:.4f}"
              f"{'  *significant*' if pv < 0.05 else ''}")

print("\n" + "=" * 62)
print("FUSION vs UNIMODAL  (best fusion vs each single stream)")
print("=" * 62)
best = max((k for k in agg), key=lambda k: agg[k][0].mean())
for uni in [("audio_only", "mfcc"), ("text_only", "mfcc")]:
    u = agg.get(uni)
    if not u: continue
    t, pv = stats.ttest_ind(agg[best][0], u[0])
    print(f"  {best[0]}/{best[1]} {agg[best][0].mean():.2f}  vs  {uni[0]} {u[0].mean():.2f}  "
          f"diff {agg[best][0].mean()-u[0].mean():+.2f}  p={pv:.5f}"
          f"{'  *significant*' if pv < 0.05 else ''}")
