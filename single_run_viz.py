#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量/单个 run 可视化：
  - number_density(x,t) 时空热图
  - 叠加鞘层轨迹 x_sheath(t)（优先用 tnsa1d_timeseries.csv；否则由 Ex 兜底推断）
支持：
  --all    扫描 root 下所有 run（含 job/*.sdf）
  --runs   指定若干 run 名（与 run 目录名一致）
"""
import os, argparse, glob, numpy as np, pandas as pd
import matplotlib.pyplot as plt

def _first_attr_case_insensitive(obj, substrings):
    subs = [s.lower() for s in substrings]
    for name in dir(obj):
        lname = name.lower()
        if any(s in lname for s in subs):
            return name
    return None

def _extract_x_from_grid(grid_obj):
    g = getattr(grid_obj, "data", None)
    if isinstance(g, np.ndarray):
        return g.squeeze()[:, 0] if g.ndim > 1 else g.squeeze()
    if hasattr(g, "__len__") and len(g) > 0:
        x0 = g[0]
        return x0 if isinstance(x0, np.ndarray) else np.array(x0)
    return np.array(g).squeeze()

def read_sdf_series(jobdir, species="auto", stride=1, max_frames=None, want_ex=True):
    import sdf
    files = sorted(glob.glob(os.path.join(jobdir, "*.sdf")))
    if not files: raise SystemExit(f"未找到 SDF: {jobdir}/*.sdf")
    if stride > 1: files = files[::stride]
    if max_frames: files = files[:max_frames]

    t_list, x_ref, nd_list, ex_list = [], None, [], []
    for f in files:
        d = sdf.read(f)
        # time
        t = None
        if hasattr(d, "Header") and isinstance(d.Header, dict):
            t = d.Header.get("time", None)
        if t is None and hasattr(d, "time"):
            t = float(d.time)

        # grid
        grid_name = (_first_attr_case_insensitive(d, ["grid_grid_mid"]) or
                     _first_attr_case_insensitive(d, ["grid_grid"]) or
                     _first_attr_case_insensitive(d, ["grid"]))
        if not grid_name: continue
        x = _extract_x_from_grid(getattr(d, grid_name))

        # number density（优先 Ion）
        nd_name = None
        if species.lower() == "ion":
            nd_name = (_first_attr_case_insensitive(d, ["derived_number_density_ion"]) or
                       _first_attr_case_insensitive(d, ["number_density_ion"]))
        elif species.lower() == "electron":
            nd_name = (_first_attr_case_insensitive(d, ["derived_number_density_electron"]) or
                       _first_attr_case_insensitive(d, ["number_density_electron"]))
        else:  # auto
            nd_name = (_first_attr_case_insensitive(d, ["derived_number_density_ion","number_density_ion"]) or
                       _first_attr_case_insensitive(d, ["derived_number_density_electron","number_density_electron"]))
        if not nd_name: continue
        nd = np.array(getattr(d, nd_name).data).squeeze()

        # Ex（兜底用）
        Ex = None
        if want_ex:
            ex_name = (_first_attr_case_insensitive(d, ["electric_field_ex"]) or
                       _first_attr_case_insensitive(d, ["fields_ex"]) or
                       _first_attr_case_insensitive(d, ["ex"]))
            if ex_name:
                Ex = np.array(getattr(d, ex_name).data).squeeze()

        if x_ref is None:
            x_ref = x
        t_list.append(float(t))
        nd_list.append((x, nd))
        ex_list.append((x, Ex) if Ex is not None else None)

    return np.array(t_list), x_ref, nd_list, ex_list

def build_heatmap(x_ref, nd_pairs, nx_plot=1200, log=False):
    xr = np.linspace(x_ref.min(), x_ref.max(), nx_plot)
    mat = []
    for x, nd in nd_pairs:
        mat.append(np.interp(xr, x, nd))
    mat = np.vstack(mat)
    if log:
        posmin = np.min(mat[mat > 0]) if np.any(mat > 0) else 1.0
        mat = np.log10(np.maximum(mat, posmin*1e-6))
    return xr, mat

def load_sheath_from_csv(rundir):
    csvp = os.path.join(rundir, "tnsa1d_timeseries.csv")
    if os.path.exists(csvp):
        df = pd.read_csv(csvp)
        if {"t","sheath_pos"}.issubset(df.columns):
            return df["t"].values, df["sheath_pos"].values
    return None, None

def sheath_from_ex(x_ref, ex_pairs, t_arr, x_rear=None):
    if not ex_pairs or ex_pairs[0] is None:
        return None, None
    x0, Ex0 = ex_pairs[0]
    if x_rear is None:
        x_rear = x0[int(np.argmax(np.abs(Ex0)))]
    xs = []
    for p in ex_pairs:
        if p is None:
            xs.append(np.nan); continue
        x, Ex = p
        m = x >= x_rear
        xv, Ev = x[m], Ex[m]
        j = int(np.argmax(np.abs(Ev)))
        xs.append(float(xv[j]))
    return t_arr, np.array(xs)

def plot_run(rundir, nx_plot=1200, stride=1, species="ion", log=True, overwrite=False):
    jobdir = os.path.join(rundir, "job")
    outdir = os.path.join(rundir, "plots_single")
    os.makedirs(outdir, exist_ok=True)
    outpng = os.path.join(outdir, "heatmap_density_sheath.png")
    if (not overwrite) and os.path.exists(outpng):
        print("[SKIP]", outpng)
        return outpng

    # 读 SDF
    t, x_ref, nd_pairs, ex_pairs = read_sdf_series(jobdir, species=species, stride=stride, want_ex=True)

    # 热图
    xr, mat = build_heatmap(x_ref, nd_pairs, nx_plot=nx_plot, log=log)

    # 鞘轨迹
    t_sheath, x_sheath = load_sheath_from_csv(rundir)
    if t_sheath is None or x_sheath is None:
        t_sheath, x_sheath = sheath_from_ex(x_ref, ex_pairs, t)
    # 作图（x→µm, t→fs）
    fig, ax = plt.subplots(figsize=(8, 4.8))
    im = ax.imshow(
        mat, aspect="auto", origin="lower",
        extent=[xr.min()*1e6, xr.max()*1e6, t.min()*1e15, t.max()*1e15],
        cmap="plasma"
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("log10(n)" if log else "n (arb.)")
    if x_sheath is not None:
        ax.plot(np.array(x_sheath)*1e6, np.array(t_sheath)*1e15, "w--", lw=2, label="sheath trajectory")
        ax.legend(loc="upper right")
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("t (fs)")
    ax.set_title(os.path.basename(rundir))
    fig.tight_layout()
    fig.savefig(outpng, dpi=180)
    plt.close(fig)
    print("✅", outpng)
    return outpng

def discover_runs(root):
    # 规则：包含 job/*.sdf 的目录即视为一个 run
    cands = sorted([d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d)])
    runs = []
    for rd in cands:
        if glob.glob(os.path.join(rd, "job", "*.sdf")):
            runs.append(rd)
    return runs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", help="runmax 下的若干 run 目录名（如 a0-10_ne-100_d-3.0um）")
    ap.add_argument("--all", action="store_true", help="处理 root 下所有包含 job/*.sdf 的 run")
    ap.add_argument("--root", default="/epoch/epoch1d/runmax", help="runmax 根目录")
    ap.add_argument("--nx_plot", type=int, default=1200)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--species", default="ion", choices=["ion","electron","auto"])
    ap.add_argument("--log", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    run_dirs = []
    if args.all:
        run_dirs = discover_runs(args.root)
    elif args.runs:
        run_dirs = [os.path.join(args.root, name) for name in args.runs]
    else:
        raise SystemExit("请使用 --all 或 --runs <names...>")

    if not run_dirs:
        raise SystemExit("未发现可处理的 run（检查是否存在 job/*.sdf）")

    for rundir in run_dirs:
        if not os.path.isdir(rundir):
            print("[WARN] 不存在：", rundir); continue
        try:
            plot_run(rundir, nx_plot=args.nx_plot, stride=args.stride,
                     species=args.species, log=args.log, overwrite=args.overwrite)
        except SystemExit as e:
            print("[SKIP]", rundir, "-", e)
        except Exception as e:
            print("[ERR ]", rundir, "-", e)

if __name__ == "__main__":
    main()
