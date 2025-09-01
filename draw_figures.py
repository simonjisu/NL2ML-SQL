import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
plt.style.use("ggplot")

def pick_first_existing(candidates, df):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def safe_log(series: pd.Series) -> pd.Series:
    s = series.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if (s > 0).all():
        return np.log(series.astype(float))
    if (s >= 0).all():
        return np.log1p(series.astype(float))
    shift = 1 - float(series.min())
    return np.log(series.astype(float) + shift)

def skyline(df: pd.DataFrame,
            perf_col: str,
            time_col: str,
            ascending: tuple[bool, bool] = (True, True)) -> pd.DataFrame:
    """
    Pareto-optimal rows for two criteria.
    ascending = (for time_col, for perf_col) — True=minimize, False=maximize
    """
    ordered = df.sort_values([time_col, perf_col],
                             ascending=list(ascending),
                             ignore_index=True)
    best_perf = np.inf if ascending[1] else -np.inf
    winners = []
    for _, row in ordered.iterrows():
        val = row[perf_col]
        better = (val <= best_perf) if ascending[1] else (val >= best_perf)
        if better:
            winners.append(row)
            best_perf = val
    return pd.DataFrame(winners)

def normalize_metric(
    df: pd.DataFrame,
    metric_col: str,
    dataset_col: str = "dataset",
    method: str = "minmax",   # "minmax" | "z" | "rank"
    log_pre: bool = False,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """
    Per-dataset normalization with natural orientation:
    - For lower-is-better metrics (rmse/mae/... and time): 0 = best, 1 = worst
    - For higher-is-better metrics (e.g., r2):             0 = worst, 1 = best

    'rank' returns [0,1] with 0 = best for lower-is-better metrics (e.g., rmse/time),
    and 1 = best for higher-is-better metrics (e.g., r2).
    """
    out = df.copy()
    s = out[metric_col].astype(float)

    name = metric_col.lower()
    lower_is_better = (
        name in {"rmse","mse","mae","mape","smape","rmsle","logloss","loss"} or
        any(k in name for k in ["time", "duration", "latency", "cost"])
    )

    if log_pre:
        s = np.log1p(np.clip(s, a_min=0, a_max=None))

    g = out.groupby(dataset_col)[s.name]

    if method.lower() == "minmax":
        gmin  = g.transform("min")
        gmax  = g.transform("max")
        denom = (gmax - gmin).replace(0, np.nan)
        mm = (s - gmin) / (denom + eps)       # 0 at min, 1 at max
        mm = mm.fillna(0.5)                   # constant group
        if lower_is_better:
            # Keep as-is: 0=best (min), 1=worst (max)
            out[f"{metric_col}_minmax"] = mm
        else:
            # Flip: higher-is-better -> 1=best, 0=worst
            out[f"{metric_col}_minmax"] = 1 - mm
        return out

    elif method.lower() == "z":
        # z is unbounded; if you need [0,1], prefer 'rank' or 'minmax'.
        gmean = g.transform("mean")
        gstd  = g.transform("std").replace(0, np.nan)
        z = (s - gmean) / (gstd + eps)
        # For lower-is-better we could negate, but it won’t be in [0,1].
        out[f"{metric_col}_z"] = -z if lower_is_better else z
        return out

    elif method.lower() == "rank":
        # Ranks: ascending=True means smaller is better.
        ranks  = g.transform(lambda col: col.rank(method="average", ascending=True))
        counts = g.transform("count")
        denom = (counts - 1).replace(0, np.nan)

        # Map to [0,1] with 0 at the best (smallest) and 1 at the worst (largest)
        rank01_small_best = (ranks - 1) / (denom + eps)   # 0 best, 1 worst

        if lower_is_better:
            # e.g., RMSE/TIME: already 0=best, 1=worst
            out[f"{metric_col}_rank"] = rank01_small_best
        else:
            # e.g., R^2: we want 1=best (larger), 0=worst → flip
            out[f"{metric_col}_rank"] = 1 - rank01_small_best
        return out

    else:
        raise ValueError("method must be one of {'minmax', 'z', 'rank'}")

def plot_mlselection():
    df = pd.read_csv("./exps/exp_mlselection.csv")
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'id'}, inplace=True)

    methods = ['minmax', 'z', 'rank']
    perf_cols = ['r2', 'rmse', 'mae']
    time_cols = ['fit_time', 'score_time']
    for metric in perf_cols:
        for method in methods:
            if metric in df.columns:
                df = normalize_metric(df, metric, dataset_col="dataset", method=method, log_pre=False)

    for time_col in time_cols:
        for method in methods:
            if time_col in df.columns:
                df = normalize_metric(df, time_col, dataset_col="dataset", method=method, log_pre=False)

    method = methods[2]
    perf_col = perf_cols[1] + '_' + method
    time_col = time_cols[0] + '_' + method
    ascending = (True, True)
    work = df[['id', perf_col, time_col]].copy().dropna(subset=[perf_col, time_col])
    work["perf"] = work[perf_col]
    work["time"] = work[time_col]

    sky = skyline(work, perf_col="perf", time_col="time", ascending=ascending)
    sky_labeled = sky.merge(df[["id"] + ["algorithm", "dataset", "model_id", "num_rows", "num_features"]].drop_duplicates("id"), on="id", how="left")

    base = sky_labeled["model_id"].astype(str) + "[" + sky_labeled["algorithm"].astype(str).str.replace('bayesian_ridge', 'bayes_ridge') + "] "
    sky_labeled['num_rows'] = sky_labeled['num_rows'] // 1000

    turn_off = False
    offset_xs = [-0.02, -0.02, -0.02, -0.03, -0.05, -0.02, 0.005, -0.16, -0.06] if not turn_off else list(range(len(sky_labeled)))
    offset_ys = [0.03, 0.02, 0.005, -0.01, 0.0, -0.07, -0.04, 0.01, 0.0] if not turn_off else list(range(len(sky_labeled)))
    sky_labeled["text"] = base # + pd.Series(newlines) + "(" + sky_labeled["num_rows"].astype(str) + "K, " + sky_labeled["num_features"].astype(str) + ")"
    sky_labeled['offset_xs'] = offset_xs
    sky_labeled['offset_ys'] = offset_ys

    non_sky = work[~work["id"].isin(sky_labeled["id"])].copy()
    colors = {"non-skyline": "#da3636", "skyline": "#1c81d3"}

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(non_sky["time"], non_sky["perf"], marker="x", alpha=0.5, s=80, color=colors["non-skyline"], label="Filtered neighbours")
    ax.scatter(sky_labeled["time"], sky_labeled["perf"], marker="o", alpha=0.9, s=100, color=colors["skyline"], label="Skyline")
    for _, r in sky_labeled.iterrows():
        ax.annotate(
            str(r["text"]),
            (r["time"], r["perf"]) if turn_off else (r["time"]+r["offset_xs"], r["perf"]+r["offset_ys"]),
            xytext=(5, 5), textcoords="offset points",
            fontsize=12, ha="left", va="bottom"
        )

    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_xlabel(f"Normalized Training Time (Lower Better)", fontsize=14)
    perf_col_labels = {'r2': '$R^2$', 'rmse': '$RMSE$', 'mae': '$MAE$'}
    ax.set_ylabel(f"Normalized {perf_col_labels[perf_col.split('_')[0]]} (Lower Better)", fontsize=14)
    ax.legend(loc="upper right")

    legend = ax.legend(fontsize=14, loc="upper right", frameon=True)
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("black")   # optional: thin black border
    plt.tight_layout()
    plt.show()

    fig.savefig("skyline_mlselection.pdf", dpi=300, bbox_inches='tight')

def plot_confusion_matrix(y_true, y_pred):
    cms = [
        np.array([[538, 62], [4, 596]]),   # GPT-4o-mini with CoT
        np.array([[535, 65], [7, 596]]),   # GPT-4o-mini with SC
        np.array([[252, 348], [267, 333]]),# Llama-3.1 8B with CoT
        np.array([[455, 145], [18, 582]])  # Llama-3.1 8B with SC
    ]

    titles = [
        "GPT-4o-mini with CoT",
        "GPT-4o-mini with MV",
        "Llama-3.1 8B with CoT",
        "Llama-3.1 8B with MV"
    ]

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    vmin = min(cm.min() for cm in cms)
    vmax = max(cm.max() for cm in cms)

    for ax, cm, title in zip(axes.ravel(), cms, titles):
        im = sns.heatmap(
            cm,
            annot=True, fmt="d", cmap="coolwarm",
            cbar=False,
            vmin=vmin, vmax=vmax,
            square=True, linewidths=0,
            annot_kws={"size": 12},
            ax=ax
        )
        ax.set_title(title, fontsize=14)
        ax.set_xticklabels(["True", "False"], fontsize=12)
        ax.set_yticklabels(["True", "False"], fontsize=12, rotation=0)

    axes[1, 0].set_xlabel("Prediction", fontsize=12)
    axes[1, 1].set_xlabel("Prediction", fontsize=12)
    axes[0, 0].set_ylabel("Target", fontsize=12)
    axes[1, 0].set_ylabel("Target", fontsize=12)
    plt.tight_layout(rect=[0, 0.15, 1, 1], w_pad=-10, h_pad=3)

    cbar = fig.colorbar(
        im.collections[0],
        ax=axes.ravel().tolist(),
        orientation="horizontal",
        fraction=0.05, pad=0.12
    )
    cbar.set_label("Counts", fontsize=12)
    cbar.ax.tick_params(labelsize=12)

    plt.savefig("confusion_matrices.pdf", dpi=300, bbox_inches="tight")
    plt.show()

def plot_intent_detection():
    df = pd.read_csv("full_test.csv") 
    df["scenario"] = df["scenario"].astype(str).str.replace(" ", "", regex=False)

    scenarios = ["Type1", "Type2"]
    model_order = list(df["llm_type"].drop_duplicates())
    x = np.arange(len(model_order))
    bar_w = 0.38

    colors = {
        "PM": "#8fd1f5",          # light blue
        "Cost-to-PM": "#1a71ac",  # dark blue
        "EM": "#f19f98",          # pastel red
        "Cost-to-EM": "#e41a1c",  # dark red
    }
    agg = (
        df.groupby(["scenario", "llm_type", "metric"], dropna=False)["value"]
        .mean()
        .rename("val")
        .reset_index()
    )

    # convenience getters
    def get_vals(scenario, metric):
        """Return values aligned to model_order; NaN -> np.nan (plotted as gaps)."""
        s = (
            agg[(agg["scenario"] == scenario) & (agg["metric"] == metric)]
            .set_index("llm_type")["val"]
            .reindex(model_order)
            .astype(float)
            .values
        )
        return s

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6, 8), sharex=False, sharey=True)

    def style_axes(ax):
        # grey panel, no gridlines
        ax.set_facecolor("#f0f0f0")
        ax.grid(False)
        # optional: lighten spines
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    # build legend handles (bars + lines) once
    legend_handles = [
        Patch(facecolor=colors["PM"], label="PM"),
        Patch(facecolor=colors["EM"], label="EM"),
        Line2D([0], [0], color=colors["Cost-to-PM"], linestyle="--", marker="o", label="Cost-to-PM"),
        Line2D([0], [0], color=colors["Cost-to-EM"], linestyle="--", marker="o", label="Cost-to-EM"),
    ]

    def annotate_bars(ax, xs, heights):
        for xi, h in zip(xs, heights):
            if np.isfinite(h):
                ax.annotate(f"{h:.2f}", (xi, h), ha="center", va="bottom",
                            fontsize=10, xytext=(0, 1), textcoords="offset points", zorder=10)

    def plot_scenario(ax, scenario):
        style_axes(ax)

        pm_vals = get_vals(scenario, "PM")
        em_vals = get_vals(scenario, "EM")

        pm_x = x - bar_w/2
        em_x = x + bar_w/2

        ax.bar(pm_x, pm_vals, width=bar_w, color=colors["PM"], label="PM")
        ax.bar(em_x, em_vals, width=bar_w, color=colors["EM"], label="EM")

        annotate_bars(ax, pm_x, pm_vals)
        annotate_bars(ax, em_x, em_vals)

        ax.set_ylabel("PM / EM", fontsize=12)
        ax.tick_params(axis="y", labelsize=11)

        ax.set_xticks(x)
        ax.set_xticklabels(['Llama-3.1 8B\n(Fine-tuned)', 'GPT o1-mini', 'GPT o3-mini'])
        ax.tick_params(axis="x", labelsize=11)

        ax_r = ax.twinx()
        style_axes(ax_r)

        cpm_vals = get_vals(scenario, "Cost-to-PM")
        cem_vals = get_vals(scenario, "Cost-to-EM")

        ax_r.plot(x, cpm_vals, linestyle="--", linewidth=2, marker="o", color=colors["Cost-to-PM"], alpha=0.8, zorder=1)
        ax_r.plot(x, cem_vals, linestyle="--", linewidth=2, marker="o", color=colors["Cost-to-EM"], alpha=0.8, zorder=1)

        ax_r.set_ylabel("Cost-to-PM / EM", fontsize=12)
        ax_r.tick_params(axis="y", labelsize=11)

        ax.set_title(f"Scenario {scenario.replace('Type','')}", fontsize=16)

    plot_scenario(axes[0], "Type1")
    plot_scenario(axes[1], "Type2")

    fig.legend(handles=legend_handles, loc="lower center", ncol=4, frameon=True)
    leg = fig.legends[0]
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor("black")

    plt.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.12, hspace=0.3)
    plt.show()
    fig.savefig("full_test.pdf", dpi=300, bbox_inches="tight")