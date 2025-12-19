"""
Utility functions for statistical visualization and model-comparison analysis.

This module includes:
- boxplots for model performance,
- correlations between complexity and performance,
- per-model correlation grids,
- scatterplots with regression/LOWESS trends,
- model ranking summaries,
- Friedman + Nemenyi statistical testing,
- repeated-measures ANOVA.

Designed for reproducible statistical analysis in the context of
interpretable ML model benchmarking.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Dict
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp
from statsmodels.stats.anova import AnovaRM


# ============================================================================
#                                INTERNAL UTILS
# ============================================================================

def _print_header(header: str):
    """Prints a formatted section header."""
    print("=" * 30)
    print(header)


# ============================================================================
#                     1) BOXPLOTS FOR MODEL PERFORMANCE
# ============================================================================

def plot_model_boxplots(df, metrics, yscale=None, rotation=30):
    """
    Draw boxplots of performance metrics across models.

    Parameters
    ----------
    df : DataFrame
        Must contain column 'Model' and the metrics.
    metrics : list[str]
        Metrics to visualize.
    yscale : {'log', None}
    rotation : int
        Rotation angle for x-axis labels.
    """
    n = len(metrics)
    ncols = 2
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows), sharey=False
    )
    axes = axes.ravel()

    for i, metric in enumerate(metrics):
        sns.boxplot(
            data=df,
            x="Model",
            y=metric,
            hue="Model",
            ax=axes[i],
            palette="Set2"
        )
        axes[i].set_title(f"{metric} per modello")
        axes[i].tick_params(axis="x", rotation=rotation)

        if yscale == "log":
            axes[i].set_yscale("log")

    # Remove unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


# ============================================================================
#                    2) COMPLEXITY ↔ PERFORMANCE CORRELATION
# ============================================================================

def correlation_complexity_performance(df, complexity_cols, performance_cols, method="spearman"):
    """
    Compute correlation matrix between complexity and performance metrics.

    Returns
    -------
    DataFrame  (index = complexity cols, columns = performance cols)
    """
    corr = pd.DataFrame(index=complexity_cols, columns=performance_cols, dtype=float)

    for c in complexity_cols:
        for p in performance_cols:
            corr.loc[c, p] = df[[c, p]].corr(method=method).iloc[0, 1]

    return corr


def plot_correlation_heatmap(corr_df):
    """
    Plot a heatmap for the complexity–performance correlation matrix.
    """
    plt.figure(figsize=(1.2 * len(corr_df.columns) + 2,
                        1.2 * len(corr_df.index) + 2))
    ax = sns.heatmap(
        corr_df.astype(float),
        annot=True,
        fmt=".2f",
        center=0.0,
        vmin=-1,
        vmax=1,
        linewidths=0.5
    )
    ax.set_title("Correlazioni complessità (righe) vs performance (colonne)")
    plt.tight_layout()
    _print_header("Correlazioni complessità ↔ performance")
    plt.show()
    return ax


# ============================================================================
#            2b) COMPLEXITY ↔ PERFORMANCE CORRELATION BY MODEL
# ============================================================================

def compute_corr_by_model(df, complexity_cols, performance_cols, method="spearman"):
    """
    Compute complexity–performance correlations separately for each model.

    Returns
    -------
    dict
        model_name -> DataFrame (complexity × performance)
    """
    out = {}

    for m, g in df.groupby("Model"):
        corr = pd.DataFrame(index=complexity_cols, columns=performance_cols, dtype=float)

        for c in complexity_cols:
            for p in performance_cols:
                corr.loc[c, p] = g[[c, p]].corr(method=method).iloc[0, 1]

        out[m] = corr

    return out


def plot_corr_grid(corr_by_model, ncols=3, suptitle="Correlazioni complessità vs performance — per modello"):
    """
    Plot a grid of per-model complexity–performance heatmaps.
    """
    models = list(corr_by_model.keys())
    n = len(models)
    nrows = int(np.ceil(n / ncols))

    first = next(iter(corr_by_model.values()))
    all_rows = list(first.index)
    all_cols = list(first.columns)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.9 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, m in zip(axes, models):
        cdf = corr_by_model[m].reindex(index=all_rows, columns=all_cols)
        sns.heatmap(
            cdf,
            ax=ax,
            vmin=-1.0,
            vmax=1.0,
            center=0.0,
            annot=True,
            fmt=".2f",
            cbar=False,
            linewidths=0.4
        )
        ax.set_title(m)
        ax.set_xlabel("Performance")
        ax.set_ylabel("Complessità")

    # Hide empty axes
    for ax in axes[len(models):]:
        ax.axis("off")

    if suptitle:
        fig.suptitle(suptitle, y=1.02, fontsize=12)

    fig.tight_layout()
    plt.show()
    return fig


# ============================================================================
#             3) SCATTER PLOT: PERFORMANCE vs COMPLEXITY
# ============================================================================

def scatter_perf_vs_complexity(
    df,
    perf,
    complexity,
    hue="Model",
    lowess=False,
    xscale=None,
    yscale=None
):
    """
    Scatterplot with trendline (linear or LOWESS) for performance vs complexity.

    Parameters
    ----------
    perf : str
        Column name of performance metric.
    complexity : str
        Column name of complexity metric.
    lowess : bool
        If True, per-model LOWESS curves via FacetGrid.
    """
    _print_header("Singola metrica performance vs singola metrica complessita'")

    if lowess:
        g = sns.FacetGrid(df, col=hue, col_wrap=3, sharex=True, sharey=False)
        g.map(
            sns.regplot,
            complexity,
            perf,
            scatter_kws=dict(alpha=0.5, s=20),
            lowess=True
        )

        for ax in g.axes.flatten():
            if xscale:
                ax.set_xscale(xscale)
            if yscale:
                ax.set_yscale(yscale)

        g.figure.subplots_adjust(top=0.85)
        g.figure.suptitle(f"{perf} vs {complexity} (LOWESS)")
        plt.show()
        return g

    else:
        g = sns.lmplot(
            data=df,
            x=complexity,
            y=perf,
            hue=hue,
            height=4,
            aspect=1.2,
            scatter_kws=dict(alpha=0.5, s=20)
        )
        ax = g.ax

        if xscale:
            ax.set_xscale(xscale)
        if yscale:
            ax.set_yscale(yscale)

        g.figure.subplots_adjust(top=0.9)
        g.figure.suptitle(f"{perf} vs {complexity} (regr. lineare)")
        plt.show()
        return g


# ============================================================================
#                5) MODEL RANKING SUMMARY AND VISUALIZATION
# ============================================================================

def model_ranking_summary(df, score, higher_is_better):
    """
    Compute per-dataset ranking of models and produce summary statistics.

    Returns
    -------
    summary : DataFrame
        Compact table with mean rank, std, wins, win_rate.
    ranked : DataFrame
        Per-dataset ranking values.
    """

    def _rank(group):
        ascending = not higher_is_better
        return group[score].rank(ascending=ascending, method="average")

    ranked = df.copy()
    ranked["rank"] = ranked.groupby("Dataset", group_keys=False).apply(_rank)

    summary = (
        ranked.groupby("Model")
        .agg(
            mean_rank=("rank", "mean"),
            std_rank=("rank", "std"),
            wins=("rank", lambda s: (s == 1).sum()),
            n=("rank", "count"),
        )
        .sort_values("mean_rank")
    )

    summary["win_rate"] = summary["wins"] / ranked["Dataset"].nunique()
    return summary, ranked


def plot_rank_bar(summary_df, title="Ranking medio per modello"):
    """
    Horizontal barplot of mean ranking per model.
    """
    ax = summary_df.sort_values("mean_rank").plot(
        y="mean_rank",
        kind="barh",
        xlim=(1, None),
        figsize=(6, 3.8),
        legend=False
    )
    ax.set_xlabel("Mean rank (↓ meglio)")
    ax.set_ylabel("Model")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()
    return ax


def stratified_model_ranking(df, complexity_metrics, evaluation_metric):
    """
    Compute ranking separately for datasets above and below the median
    of each complexity metric.
    """
    median_metrics = df[complexity_metrics].quantile(0.5)

    for metric in complexity_metrics:
        summary_ge, _ = model_ranking_summary(
            df[df[metric] >= median_metrics[metric]],
            score=evaluation_metric,
            higher_is_better=True
        )
        summary_l, _ = model_ranking_summary(
            df[df[metric] < median_metrics[metric]],
            score=evaluation_metric,
            higher_is_better=True
        )

        plot_rank_bar(summary_ge,
                      title=f"Ranking per modello per {metric} >= mediana")
        plot_rank_bar(summary_l,
                      title=f"Ranking per modello per {metric} < mediana")


def stratified_corr_by_model(df, complexity_cols, performance_cols):
    """
    Compute per-model correlations, stratified by high/low complexity.
    """
    median_metrics = df[complexity_cols].quantile(0.5)

    for metric in complexity_cols:
        corr_ge = compute_corr_by_model(
            df[df[metric] >= median_metrics[metric]],
            complexity_cols,
            performance_cols
        )
        corr_lt = compute_corr_by_model(
            df[df[metric] < median_metrics[metric]],
            complexity_cols,
            performance_cols
        )

        plot_corr_grid(
            corr_ge,
            ncols=3,
            suptitle=f"Correlazione complessita' vs performance per modello (≥ mediana {metric})"
        )
        plot_corr_grid(
            corr_lt,
            ncols=3,
            suptitle=f"Correlazione complessita' vs performance per modello (< mediana {metric})"
        )


# ============================================================================
#              6) FRIEDMAN TEST + POST-HOC NEMENYI SIGNIFICANCE
# ============================================================================

def friedman_and_nemenyi(ranked, alpha=0.05):
    """
    Perform Friedman test on model rankings and, if significant,
    post-hoc Nemenyi pairwise comparisons.
    """
    perf_ranks = ranked.pivot(index="Dataset", columns="Model", values="rank")

    stat, p_value = friedmanchisquare(
        *[perf_ranks[col].values for col in perf_ranks.columns]
    )

    print("=== Friedman test ===")
    print(f"Statistic: {stat:.4f}")
    print(f"P-value  : {p_value:.6f}")

    if p_value < alpha:
        print(f"\nDifferenze significative trovate (p < {alpha}). Post-hoc di Nemenyi:")
        nemenyi = sp.posthoc_nemenyi_friedman(perf_ranks.values)
        nemenyi.index = perf_ranks.columns
        nemenyi.columns = perf_ranks.columns
        print(nemenyi)
        report_significant_pairs(nemenyi, alpha)
        return nemenyi

    else:
        print(f"\nNessuna differenza significativa (p ≥ {alpha}).")
        return None


def report_significant_pairs(nemenyi, alpha=0.05):
    """
    Extract significantly different model pairs from Nemenyi matrix.
    """
    results = []
    models = nemenyi.columns

    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            p = nemenyi.iloc[i, j]
            if p < alpha:
                results.append((models[i], models[j], p))

    if results:
        print(f"Coppie con differenze significative (p < {alpha}):")
        for m1, m2, p in results:
            print(f" - {m1} vs {m2}: p = {p:.6f}")
    else:
        print(f"Nessuna coppia significativa con p < {alpha}.")

    return results


# ============================================================================
#                        7) REPEATED-MEASURES ANOVA
# ============================================================================

def repeated_measures_anova(df_long, score_col, subject_col="Dataset", within_col="Model"):
    """
    Perform repeated-measures ANOVA on long-format performance data.

    df_long must include columns:
    - subject_col  (e.g. Dataset)
    - within_col   (e.g. Model)
    - score_col    (metric of interest)
    """
    aov = AnovaRM(
        data=df_long,
        depvar=score_col,
        subject=subject_col,
        within=[within_col]
    ).fit()

    print("=== Repeated Measures ANOVA ===")
    print(aov.summary())
    return aov
