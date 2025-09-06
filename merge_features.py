#!/usr/bin/env python3
import os, re, glob, csv, pandas as pd

RUNROOT = "/epoch/epoch1d/runmax"
rows = []

pat = re.compile(r"a0-(?P<a0>[\d\.]+)_ne-(?P<ne>[\d\.]+)_d-(?P<d>[\d\.]+)um$")

for run_dir in sorted(glob.glob(os.path.join(RUNROOT, "a0-*_*_d-*um"))):
    m = pat.search(run_dir)
    if not m:
        continue
    a0 = float(m.group("a0"))
    ne = float(m.group("ne"))
    d  = float(m.group("d"))

    sum_csv = os.path.join(run_dir, "tnsa1d_summary.csv")
    if not os.path.exists(sum_csv):
        continue
    with open(sum_csv, newline="") as f:
        r = next(csv.DictReader(f))    # 选取你需要的字段（可按需增减）
    rows.append({
        "run_dir": run_dir,
        "a0": a0, "n0_over_nc": ne, "d_um": d,
        "Ex_peak_max": float(r.get("Ex_peak_max", 0)),
        "t_at_Ex_peak": float(r.get("t_at_Ex_peak", 0)) if r.get("t_at_Ex_peak") else 0.0,
        "sheath_speed_mean": float(r.get("sheath_speed_mean", 0)),
        "sheath_width_mean": float(r.get("sheath_width_mean", 0)),
        "Phi_max": float(r.get("Phi_max", 0)),
    })

df = pd.DataFrame(rows)
out_csv = os.path.join(RUNROOT, "features.csv")
df.to_csv(out_csv, index=False)
print("✅ 写出:", out_csv, f"(共 {len(df)} 行)")
