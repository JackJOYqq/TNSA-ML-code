#!/usr/bin/env python3
# 对 runmax/features.csv 做跨 run 的 KMeans & XGBoost，可视化到 plots_cross/
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, GroupKFold, RandomizedSearchCV
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from xgboost import XGBRegressor
import xgboost as xgb
import scipy.stats as st


# ---------- 工具函数 ----------
def safeX(df, cols):
    """清洗特征矩阵：去除inf、前后向填充、均值填补、最后0填补"""
    X = (
        df[cols]
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
        .bfill()
    )
    for c in cols:
        if X[c].isna().any():
            X[c] = X[c].fillna(X[c].mean())
    return X.fillna(0.0).values


def choose_k(Xs, kmax=6):
    """用 silhouette 自动选择KMeans的k"""
    n = Xs.shape[0]
    if n < 5:
        return 2, None
    best = (2, -1.0)
    for k in range(2, min(kmax, n - 1) + 1):
        labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(Xs)
        sc = silhouette_score(Xs, labels)
        if sc > best[1]:
            best = (k, sc)
    return best


def fit_with_es(model, Xtr, ytr, Xva, yva):
    """
    适配 xgboost 3.0.4 的早停训练：
    1) 优先 callbacks=EarlyStopping(rounds=...)
    2) 回退 early_stopping_rounds=...
    3) 再不行则无早停
    —— 全程不传 eval_metric（使用回归默认 rmse）
    """
    # 1) callbacks（3.x 通常支持）
    try:
        model.fit(
            Xtr, ytr,
            eval_set=[(Xva, yva)],
            verbose=False,
            callbacks=[xgb.callback.EarlyStopping(rounds=80, save_best=True)]
        )
        return model
    except TypeError:
        pass
    except Exception:
        pass

    # 2) early_stopping_rounds 参数（有些构建支持）
    try:
        model.fit(
            Xtr, ytr,
            eval_set=[(Xva, yva)],
            early_stopping_rounds=80,
            verbose=False
        )
        return model
    except TypeError:
        pass
    except Exception:
        pass

    # 3) 无早停
    model.fit(
        Xtr, ytr,
        eval_set=[(Xva, yva)],
        verbose=False
    )
    return model



# ---------- 主程序 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="/epoch/epoch1d/runmax/features.csv")
    ap.add_argument("--outdir", default="/epoch/epoch1d/runmax/plots_cross")
    # 如果你的CSV里有 run 分组列（例如 run_id / deck），可以通过参数指定列名
    ap.add_argument("--group-col", default="", help="可选：分组列名（如 run_id），避免同一run泄漏")
    # Ex_peak_max 单位：si(V/m), gvperm(GV/m), norm(归一化 E/(m_e c ω / e))
    ap.add_argument("--ex-unit", choices=["si", "gvperm", "norm"], default="si",
                    help="Ex_peak_max 的单位：si=V/m；gvperm=GV/m；norm=归一化 E/(m_e c ω / e)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.features)

    # ====== 理论鞘场缩放 vs 仿真 Ex_peak_max 对比 ======
    # 物理常数（SI）
    EPS0 = 8.8541878128e-12
    KB   = 1.380649e-23
    ME   = 9.1093837015e-31
    ECH  = 1.602176634e-19
    C0   = 2.99792458e8
    PI   = np.pi

    # 你激光的中心波长（单位：um）。若不是 1.0um，请改成实际值（如 Ti:Sa 0.8um）
    laser_wavelength_um = 1.0
    lam = laser_wavelength_um * 1e-6     # m
    omega = 2*PI*C0/lam                  # rad/s

    # 临界密度（SI, m^-3）：n_c = eps0 * m_e * omega^2 / e^2
    n_c = EPS0 * ME * (omega**2) / (ECH**2)

    EX_SCALE = 1.0
    Ex_sim = df["Ex_peak_max"].values * EX_SCALE




    # 从表中读到的量：a0, n0_over_nc, Ex_peak_max（转为 SI 单位）
    Ex_sim = df["Ex_peak_max"].values * EX_SCALE
    a0_arr = df["a0"].values
    ne_over_nc = df["n0_over_nc"].values
    ne = ne_over_nc * n_c    # m^-3

    # 热电子温度（J）：T_h = m_e c^2 (sqrt(1+a0^2) - 1)
    Th_J = ME * (C0**2) * (np.sqrt(1.0 + a0_arr**2) - 1.0)

    # 理论鞘场（V/m）：E_th = sqrt(ne * kB * T_h / eps0)
    Th_J_clip = np.clip(Th_J, 1e-30, None)
    ne_clip   = np.clip(ne,   1e-6,  None)
    E_th = np.sqrt(ne_clip * Th_J_clip / EPS0)

    # 保存到表里，便于后续联动分析 / 导出
    df["E_sheath_theory"] = E_th

    # 误差指标（线性域 RMSE、相对误差中位数、Pearson & log-corr）
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_err = (Ex_sim - E_th) / np.where(E_th != 0, E_th, np.nan)
    rel_err = rel_err[~np.isnan(rel_err)]

    try:
        from scipy.stats import pearsonr
        corr = pearsonr(Ex_sim, E_th)[0]
    except Exception:
        corr = np.corrcoef(Ex_sim, E_th)[0, 1]

    Ex_clip = np.clip(Ex_sim, 1e-30, None)
    Eth_clip = np.clip(E_th,   1e-30, None)
    try:
        log_corr = np.corrcoef(np.log10(Eth_clip), np.log10(Ex_clip))[0, 1]
    except Exception:
        log_corr = np.nan

    rmse = float(np.sqrt(np.mean((Ex_sim - E_th)**2))) if len(Ex_sim) else np.nan
    med_rel = float(np.median(rel_err)) if len(rel_err) else np.nan

    print(f"[Sheath scaling]  corr(Ex_sim, E_th) = {corr:.3f}, log-corr = {log_corr:.3f}, "
          f"RMSE = {rmse:.3e} V/m, median rel.err = {med_rel:.3%}")

    # 可视化：仿真 vs 理论（对数轴）
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    lo = float(min(np.nanmin(Ex_clip), np.nanmin(Eth_clip)))
    hi = float(max(np.nanmax(Ex_clip), np.nanmax(Eth_clip)))
    lo = max(lo, 1e-2)  # 避免非正数上 log
    ax.scatter(E_th, Ex_sim, s=36, alpha=0.85)
    ax.plot([lo, hi], [lo, hi], linestyle="--")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("E_sheath (theory)  [V/m]")
    ax.set_ylabel("Ex_peak_max (simulation)  [V/m]")
    ax.set_title("Sheath field scaling: theory vs simulation")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "sheath_scaling_theory_vs_sim.png"), dpi=160)
    plt.close(fig)

    # 分组对比（厚度/密度分箱）
    try:
        df_bins = df.copy()
        df_bins["d_bin(um)"]  = pd.qcut(df_bins["d_um"], q=3, duplicates="drop")
        df_bins["ne_bin(nc)"] = pd.qcut(df_bins["n0_over_nc"], q=3, duplicates="drop")
        for col in ["d_bin(um)", "ne_bin(nc)"]:
            fig2, ax2 = plt.subplots(figsize=(6.4, 4.2))
            for key, sub in df_bins.groupby(col, observed=True):  # observed=True 抑制未来警告
                ax2.scatter(sub["E_sheath_theory"], sub["Ex_peak_max"], s=32, alpha=0.7, label=str(key))
            ax2.plot([lo, hi], [lo, hi], linestyle="--")
            ax2.set_xscale("log"); ax2.set_yscale("log")
            ax2.set_xlabel("E_sheath (theory)  [V/m]")
            ax2.set_ylabel("Ex_peak_max (simulation)  [V/m]")
            ax2.set_title(f"Sheath scaling by {col}")
            ax2.legend(fontsize=8)
            fig2.tight_layout()
            safe_name = col.replace('(', '_').replace(')', '_').replace(' ', '_')
            fig2.savefig(os.path.join(args.outdir, f"sheath_scaling_by_{safe_name}.png"), dpi=160)
            plt.close(fig2)
    except Exception:
        pass

    # ========== 1) KMeans（跨 run 相域）==========
    feats_km = ["Ex_peak_max", "sheath_speed_mean", "sheath_width_mean", "Phi_max", "a0", "n0_over_nc", "d_um"]
    X_km = safeX(df, feats_km)
    Xs = StandardScaler().fit_transform(X_km)
    k, sc = choose_k(Xs)
    km = KMeans(n_clusters=k, n_init=10, random_state=0)
    labels = km.fit_predict(Xs)
    df["cluster"] = labels

    # 2D 投影示意：Ex_peak_max vs Phi_max 着色为簇
    fig, ax = plt.subplots(figsize=(6, 5))
    sca = ax.scatter(df["Ex_peak_max"], df["Phi_max"], c=labels, s=40)
    ax.set_xlabel("Ex_peak_max")
    ax.set_ylabel("Phi_max")
    title_k = f"KMeans across runs (K={k}" + (f", silhouette={sc:.3f}" if sc is not None else "") + ")"
    ax.set_title(title_k)
    fig.colorbar(sca, ax=ax, label="Cluster")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "kmeans_runs.png"), dpi=160)
    plt.close(fig)

    # ========== 2) XGBoost（跨 run 回归 Phi_max）- 改进版 ==========
    # 特征/目标
    feats_xgb = ["Ex_peak_max", "sheath_speed_mean", "sheath_width_mean", "a0", "n0_over_nc", "d_um"]
    X = safeX(df, feats_xgb)
    y_raw = df["Phi_max"].values

    # 可选：目标对数化（Phi_max>0 时推荐）
    use_log_target = (np.nanmin(y_raw) > 0)
    y = np.log(y_raw) if use_log_target else y_raw.copy()

    # 分组（防止同一 run 泄漏）
    group_col = args.group_col.strip()
    groups = df[group_col].values if (group_col and group_col in df.columns) else None

    # 交叉验证
    n_samples = len(df)
    n_splits = 5 if n_samples >= 50 else max(3, min(5, n_samples // 10)) if n_samples >= 10 else 3
    cv = GroupKFold(n_splits=n_splits) if groups is not None else KFold(n_splits=n_splits, shuffle=True, random_state=0)

    # 物理单调约束（可按需要修改）
    # 顺序对应 feats_xgb: ["Ex_peak_max","sheath_speed_mean","sheath_width_mean","a0","n0_over_nc","d_um"]
    use_monotone = True
    monotone = (1, 0, -1, 0, 0, 0) if use_monotone else (0, 0, 0, 0, 0, 0)

    base = XGBRegressor(
        objective="reg:squarederror",
        random_state=0,
        tree_method="hist",
        monotone_constraints="(" + ",".join(map(str, monotone)) + ")",
    )

    # 超参搜索空间（不搜索 n_estimators，交给早停控制）
    param_dist = {
        "max_depth": st.randint(3, 7),
        "learning_rate": st.uniform(0.02, 0.10),
        "subsample": st.uniform(0.6, 0.4),          # 0.6 ~ 1.0
        "colsample_bytree": st.uniform(0.6, 0.4),   # 0.6 ~ 1.0
        "min_child_weight": st.randint(1, 8),
        "gamma": st.uniform(0.0, 0.4),
        "reg_lambda": st.uniform(0.5, 2.0),
        "reg_alpha": st.uniform(0.0, 0.5),
    }

    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_dist,
        n_iter=30,
        scoring="r2",
        cv=cv,
        n_jobs=-1,
        verbose=0,
        random_state=0,
        refit=True,
    )
    search.fit(X, y, groups=groups)  # 无分组时 groups=None 等价于普通fit

    # 最终模型：最佳超参 + 早停（适配 3.0.4）
    best_params = search.best_params_
    params = {**base.get_params(), **best_params}
    params["n_estimators"] = 5000
    params.pop("eval_metric", None)  # 保证不带 eval_metric
    model = XGBRegressor(**params)

    # 用 CV 的最后一折作为 early-stopping 的验证集
    splits = list(cv.split(X, y, groups)) if groups is not None else list(cv.split(X, y))
    tr_idx, va_idx = splits[-1]
    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xva, yva = X[va_idx], y[va_idx]

    model = fit_with_es(model, Xtr, ytr, Xva, yva)

    # 交叉验证评估（XGB）
    r2s, maes, rmses = [], [], []
    for tr, te in splits:
        m = XGBRegressor(**model.get_params())
        m = fit_with_es(m, X[tr], y[tr], X[te], y[te])
        yp = m.predict(X[te])
        if use_log_target:
            yp_lin = np.exp(yp)
            yte_lin = np.exp(y[te])
            r2s.append(r2_score(yte_lin, yp_lin))
            maes.append(mean_absolute_error(yte_lin, yp_lin))
            rmses.append(np.sqrt(mean_squared_error(yte_lin, yp_lin)))
        else:
            r2s.append(r2_score(y[te], yp))
            maes.append(mean_absolute_error(y[te], yp))
            rmses.append(np.sqrt(mean_squared_error(y[te], yp)))

    print(f"CV R²: mean={np.mean(r2s):.3f} ± {np.std(r2s):.3f}")
    print(f"CV MAE: mean={np.mean(maes):.3e} ± {np.std(maes):.3e}")
    print(f"CV RMSE: mean={np.mean(rmses):.3e} ± {np.std(rmses):.3e}")

    # 全量预测与可视化（XGB）
    yp_all = model.predict(X)
    if use_log_target:
        yp_all_lin = np.exp(yp_all)
        df["Phi_max_pred"] = yp_all_lin
        tru_all = y_raw
        pre_all = yp_all_lin
    else:
        df["Phi_max_pred"] = yp_all
        tru_all = y_raw
        pre_all = yp_all

    # 真值-预测（XGB）
    fig, ax = plt.subplots(figsize=(5, 5))
    lo2, hi2 = float(min(tru_all.min(), pre_all.min())), float(max(tru_all.max(), pre_all.max()))
    ax.scatter(tru_all, pre_all, s=40, alpha=0.85)
    ax.plot([lo2, hi2], [lo2, hi2], linestyle="--")
    ax.set_xlabel("True Phi_max")
    ax.set_ylabel("Pred Phi_max")
    ax.set_title(f"XGBoost (CV mean R²={np.mean(r2s):.3f})")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "xgb_runs_true_vs_pred.png"), dpi=160)
    plt.close(fig)

    # 残差图（XGB）
    resid = pre_all - tru_all
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.scatter(tru_all, resid, s=20, alpha=0.8)
    ax.axhline(0, linestyle="--")
    ax.set_xlabel("True Phi_max")
    ax.set_ylabel("Residual (Pred - True)")
    ax.set_title("Residuals vs True")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "xgb_runs_residuals.png"), dpi=160)
    plt.close(fig)

    # 原生 feature_importances_（XGB）
    try:
        imps = model.feature_importances_
        order = np.argsort(-imps)
        names = np.array(feats_xgb)[order]
        fig2, ax2 = plt.subplots(figsize=(5.8, 3.6))
        ax2.bar(range(len(imps)), imps[order])
        ax2.set_xticks(range(len(imps)))
        ax2.set_xticklabels(names, rotation=20)
        ax2.set_title("XGBoost feature importance (model-based)")
        fig2.tight_layout()
        fig2.savefig(os.path.join(args.outdir, "xgb_runs_feature_importance.png"), dpi=160)
        plt.close(fig2)
    except Exception:
        pass

    # 置换重要性（XGB）
    try:
        pi = permutation_importance(
            model, X, y, n_repeats=15, random_state=0
        )
        order = np.argsort(-pi.importances_mean)
        names = np.array(feats_xgb)[order]
        fig3, ax3 = plt.subplots(figsize=(5.8, 3.6))
        ax3.bar(range(len(names)), pi.importances_mean[order])
        ax3.set_xticks(range(len(names)))
        ax3.set_xticklabels(names, rotation=20)
        ax3.set_title("Permutation Importance")
        fig3.tight_layout()
        fig3.savefig(os.path.join(args.outdir, "xgb_perm_importance.png"), dpi=160)
        plt.close(fig3)
    except Exception:
        pass

    # PDP（XGB）
    try:
        key_feats = ["Ex_peak_max", "sheath_width_mean", "a0"]
        idxs = [feats_xgb.index(k) for k in key_feats if k in feats_xgb]
        if len(idxs) > 0:
            fig4 = plt.figure(figsize=(6.8, 6.0))
            PartialDependenceDisplay.from_estimator(
                model, X, idxs, feature_names=feats_xgb, kind="average", ax=None
            )
            plt.tight_layout()
            fig4.savefig(os.path.join(args.outdir, "xgb_pdp.png"), dpi=160)
            plt.close(fig4)
    except Exception:
        pass

    # ====== Baseline regressions: Linear vs Ridge（简化基线对比） ======
    def _eval_cv_reg(model_tpl, X_in, y_in, splits, use_log):
        r2s_b, maes_b, rmses_b = [], [], []
        for tr, te in splits:
            m = clone(model_tpl)
            m.fit(X_in[tr], y_in[tr])
            yp = m.predict(X_in[te])
            if use_log:
                yp_lin = np.exp(yp); yte_lin = np.exp(y_in[te])
                r2s_b.append(r2_score(yte_lin, yp_lin))
                maes_b.append(mean_absolute_error(yte_lin, yp_lin))
                rmses_b.append(np.sqrt(mean_squared_error(yte_lin, yp_lin)))
            else:
                r2s_b.append(r2_score(y_in[te], yp))
                maes_b.append(mean_absolute_error(y_in[te], yp))
                rmses_b.append(np.sqrt(mean_squared_error(y_in[te], yp)))
        return np.array(r2s_b), np.array(maes_b), np.array(rmses_b)

    # 1) 纯线性（仅用物理入参子集）
    feats_lin = ["a0", "n0_over_nc", "d_um"]
    X_lin = safeX(df, feats_lin)
    lin = LinearRegression()
    lin_r2, lin_mae, lin_rmse = _eval_cv_reg(lin, X_lin, y, splits, use_log_target)

    # 全量拟合并导出/画图（Linear）
    lin.fit(X_lin, y)
    lin_pred_all = lin.predict(X_lin)
    lin_pred_all = np.exp(lin_pred_all) if use_log_target else lin_pred_all
    df["Phi_max_pred_linear"] = lin_pred_all

    fig, ax = plt.subplots(figsize=(5,5))
    tru, pre = y_raw, lin_pred_all
    lo3, hi3 = float(min(np.nanmin(tru), np.nanmin(pre))), float(max(np.nanmax(tru), np.nanmax(pre)))
    ax.scatter(tru, pre, s=36, alpha=0.85)
    ax.plot([lo3, hi3], [lo3, hi3], linestyle="--")
    ax.set_xlabel("True Phi_max"); ax.set_ylabel("Pred Phi_max (Linear)")
    ax.set_title(f"Linear baseline (CV R²={np.mean(lin_r2):.3f}±{np.std(lin_r2):.3f})")
    fig.tight_layout(); fig.savefig(os.path.join(args.outdir, "baseline_linear_true_vs_pred.png"), dpi=160); plt.close(fig)

    resid_lin = pre - tru
    fig, ax = plt.subplots(figsize=(5.2,3.8))
    ax.scatter(tru, resid_lin, s=20, alpha=0.8)
    ax.axhline(0, linestyle="--")
    ax.set_xlabel("True Phi_max"); ax.set_ylabel("Residual (Pred - True)")
    ax.set_title("Residuals (Linear)")
    fig.tight_layout(); fig.savefig(os.path.join(args.outdir, "baseline_linear_residuals.png"), dpi=160); plt.close(fig)

    # 2) RidgeCV（用全部 XGB 特征 + 标准化）
    alphas = np.logspace(-4, 3, 20)
    ridge = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("ridge", RidgeCV(alphas=alphas))  # 去掉不支持的 store_cv_values
    ])
    X_ridge = safeX(df, feats_xgb)
    rid_r2, rid_mae, rid_rmse = _eval_cv_reg(ridge, X_ridge, y, splits, use_log_target)

    ridge.fit(X_ridge, y)
    rid_pred_all = ridge.predict(X_ridge)
    rid_pred_all = np.exp(rid_pred_all) if use_log_target else rid_pred_all
    df["Phi_max_pred_ridge"] = rid_pred_all

    fig, ax = plt.subplots(figsize=(5,5))
    tru, pre = y_raw, rid_pred_all
    lo4, hi4 = float(min(np.nanmin(tru), np.nanmin(pre))), float(max(np.nanmax(tru), np.nanmax(pre)))
    ax.scatter(tru, pre, s=36, alpha=0.85)
    ax.plot([lo4, hi4], [lo4, hi4], linestyle="--")
    ax.set_xlabel("True Phi_max"); ax.set_ylabel("Pred Phi_max (Ridge)")
    ax.set_title(f"Ridge baseline (CV R²={np.mean(rid_r2):.3f}±{np.std(rid_r2):.3f})")
    fig.tight_layout(); fig.savefig(os.path.join(args.outdir, "baseline_ridge_true_vs_pred.png"), dpi=160); plt.close(fig)

    resid_ridge = pre - tru
    fig, ax = plt.subplots(figsize=(5.2,3.8))
    ax.scatter(tru, resid_ridge, s=20, alpha=0.8)
    ax.axhline(0, linestyle="--")
    ax.set_xlabel("True Phi_max"); ax.set_ylabel("Residual (Pred - True)")
    ax.set_title("Residuals (Ridge)")
    fig.tight_layout(); fig.savefig(os.path.join(args.outdir, "baseline_ridge_residuals.png"), dpi=160); plt.close(fig)

        # 3) 只画 XGB vs Linear 的 CV R² 对比条形图（去掉 Ridge）
    try:
        xgb_r2_mean, xgb_r2_std = float(np.mean(r2s)), float(np.std(r2s))
    except Exception:
        xgb_r2_mean, xgb_r2_std = np.nan, np.nan

    labels_cmp = ["XGBoost", "Linear (a0,n0/nc,d)"]
    means_cmp  = np.array([xgb_r2_mean, float(np.mean(lin_r2))], dtype=float)
    errs_cmp   = np.array([xgb_r2_std,  float(np.std(lin_r2))], dtype=float)

    # 过滤 NaN/Inf，保证长度一致
    msk = np.isfinite(means_cmp) & np.isfinite(errs_cmp)
    labels_cmp = [lab for lab, ok in zip(labels_cmp, msk) if ok]
    means_cmp  = means_cmp[msk]
    errs_cmp   = errs_cmp[msk]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    xpos = np.arange(len(means_cmp))
    ax.bar(xpos, means_cmp, yerr=errs_cmp, capsize=5)
    ax.set_xticks(xpos); ax.set_xticklabels(labels_cmp, rotation=10)
    ax.set_ylabel("CV R² (mean ± std)")
    ax.set_title("Model comparison on Phi_max")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "model_cv_r2_comparison.png"), dpi=160)
    plt.close(fig)

    # 导出带 cluster/pred 的 features（包含新增列）
    out_csv = os.path.join(args.outdir, "features_with_cluster_pred.csv")
    df.to_csv(out_csv, index=False)

    print("✅ 输出：")
    print(" -", os.path.join(args.outdir, "sheath_scaling_theory_vs_sim.png"))
    print(" -", os.path.join(args.outdir, "kmeans_runs.png"))
    print(" -", os.path.join(args.outdir, "xgb_runs_true_vs_pred.png"))
    print(" -", os.path.join(args.outdir, "xgb_runs_residuals.png"))
    print(" -", os.path.join(args.outdir, "xgb_runs_feature_importance.png"))
    print(" -", os.path.join(args.outdir, "xgb_perm_importance.png"))
    print(" -", os.path.join(args.outdir, "xgb_pdp.png"))
    print(" -", os.path.join(args.outdir, "baseline_linear_true_vs_pred.png"))
    print(" -", os.path.join(args.outdir, "baseline_linear_residuals.png"))
    print(" -", os.path.join(args.outdir, "baseline_ridge_true_vs_pred.png"))
    print(" -", os.path.join(args.outdir, "baseline_ridge_residuals.png"))
    print(" -", os.path.join(args.outdir, "model_cv_r2_comparison.png"))
    print(" -", out_csv)


if __name__ == "__main__":
    main()
