from __future__ import annotations

import argparse
import hashlib
import itertools
import random
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.text as mtext
sns.set_theme(style="whitegrid")


# ============================================================================ #
# CONFIGURATION
# ============================================================================ #

PLOT_LABELS: Dict[str, str] = {
    "TokenReplay_remaining_IGNORE": "TR-IGN",
    "TokenReplay_remaining_BRD": "TR-BRD",
    "TokenReplay_remaining_MFS": "TR-MFS",
    "TokenReplay_remaining_PM4PY": "TR-PM4PY",
    "BehaviorConformance": "Behavior",
    "PrefixAlignment_revert_20": "Prefix-20",
    "PrefixAlignment_revert_10": "Prefix-10",
    "PrefixAlignment_revert_5": "Prefix-5",
}

DEFAULT_RESULTS_DIR = Path("Tests_online/remaining_analysis")

N_FILES_FOR_EVENT_ANALYSIS: Optional[int] = None
RANDOM_SEED = 42
FIGURE_FORMATS = ["png", "svg", "pdf"]
SKIP_SCATTER = False  # set by --skip-scatter

# Subfolders (at any depth under the results dir) that are ignored entirely.
EXCLUDED_FOLDERS = {""}

EVENT_METRIC_COLUMNS = ["conformance_raw", "completeness_raw", "confidence_raw", "cost_raw"]
PERFORMANCE_METRICS = ["total_time_ms", "average_processing_time_per_event_ms", "total_space_used_mb"]

CONFIG_TYPE_PREFIXES: Dict[str, Tuple[str, ...]] = {
    "token_replay": ("TokenReplay",),
    "behavior": ("BehaviorConformance", "Behavior"),
    "prefix_alignment": ("PrefixAlignment",),
}

TOKEN_REPLAY_ORDER: Tuple[str, ...] = (
    "TokenReplay_remaining_IGNORE",
    "TokenReplay_remaining_BRD",
    "TokenReplay_remaining_MFS",
    "TokenReplay_remaining_PM4PY",
)

PREFIX_ALIGNMENT_ORDER: Tuple[str, ...] = (
    "PrefixAlignment_revert_20",
    "PrefixAlignment_revert_10",
    "PrefixAlignment_revert_5",
)

# Metric used for the cross-algorithm comparison:
#   token replay -> conformance_raw, behavior -> conformance_raw, prefix alignment -> cost_raw
ALGORITHM_METRIC: Dict[str, str] = {
    "behavior": "conformance_raw",
    "token_replay": "conformance_raw",
    "prefix_alignment": "cost_raw",
}

PERFECT_CONFORMANCE_THRESHOLD = 0.999
PERFECT_COST_THRESHOLD = 1e-9

SCATTER_MIN_SIZE = 4.0
SCATTER_MAX_SIZE = 120.0
SCATTER_SIZE_LEGEND_FULL = [500_000, 1_000_000, 1_500_000, 2_000_000]
SCATTER_SIZE_LEGEND_FILTERED = [10_000, 20_000, 30_000]
MAX_SCATTER_BINS = 30

MAX_SAFE_PATH_LENGTH = 200

CORRELATION_METHODS = ("spearman", "pearson")
METHOD_NAME = {"spearman": "Spearman", "pearson": "Pearson"}
METHOD_SYMBOL = {"spearman": "rho", "pearson": "r"}

# ============================================================================ #
# LABEL HELPERS
# ============================================================================ #
def _subsample(values: np.ndarray, n: Optional[int] = 200_000) -> np.ndarray:
    if n is None or len(values) <= n:
        return values
    return np.random.default_rng(RANDOM_SEED).choice(values, n, replace=False)


HIGHLIGHT_CONFIGS: Tuple[str, ...] = ("TokenReplay_remaining_BRD", "TokenReplay_remaining_MFS")
HIGHLIGHT_FONT_SCALE = 1.5   # size of highlighted labels relative to the other labels


def highlight_heatmap(ax, row_names, col_names, highlight: Tuple[str, ...] = HIGHLIGHT_CONFIGS) -> None:
    """Make the tick labels of the highlighted configurations bold and larger.
    Call AFTER set_xticklabels/set_yticklabels (or plt.setp) so the base size is already set."""
    for tick_labels, names in ((ax.get_yticklabels(), row_names),
                               (ax.get_xticklabels(), col_names)):
        for tick, name in zip(tick_labels, names):
            if name in highlight:
                tick.set_fontweight("bold")
                tick.set_fontsize(tick.get_fontsize() * HIGHLIGHT_FONT_SCALE)

def plot_label(configuration_name: str) -> str:
    return PLOT_LABELS.get(configuration_name, configuration_name)


def metric_label(metric: str) -> str:
    if metric == "conformance_raw":
        return "Conformance"
    if metric == "cost_raw":
        return "Alignment cost"
    return metric


def classify_configuration(configuration_name: str) -> str:
    for config_type, prefixes in CONFIG_TYPE_PREFIXES.items():
        if configuration_name.startswith(prefixes):
            return config_type
    return "unknown"


def comparison_metric_for_configuration(configuration_name: str) -> Optional[str]:
    return ALGORITHM_METRIC.get(classify_configuration(configuration_name))

FONT_SCALE = 1.35   # tweak here


def scale_figure_fonts(fig, scale: float = FONT_SCALE) -> None:
    seen = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            labels = axis.get_ticklabels()
            for t in labels:
                seen.add(id(t))
            if labels:  # tick_params so regenerated ticks keep the size
                axis.set_tick_params(labelsize=labels[0].get_fontsize() * scale)
    for t in fig.findobj(mtext.Text):
        if id(t) not in seen:
            t.set_fontsize(t.get_fontsize() * scale)
    try:
        rect = [0, 0, 1, 0.96] if getattr(fig, "_suptitle", None) is not None else None
        fig.tight_layout(rect=rect) if rect else fig.tight_layout()
    except Exception:
        pass


def save_figure_object(fig, path_without_ext: Path) -> None:
    scale_figure_fonts(fig)
    path_without_ext = windows_safe_path(path_without_ext, max_total_len=MAX_SAFE_PATH_LENGTH - 5)
    path_without_ext.parent.mkdir(parents=True, exist_ok=True)
    for ext in FIGURE_FORMATS:
        fig.savefig(f"{path_without_ext}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
# ============================================================================ #
# SMALL UTILITIES
# ============================================================================ #

def windows_safe_path(path: Path, max_total_len: int = MAX_SAFE_PATH_LENGTH) -> Path:
    full_str = str(path)
    if len(full_str) <= max_total_len:
        return path

    parent, suffix, stem = path.parent, path.suffix, path.stem
    digest = hashlib.sha1((stem + suffix).encode("utf-8")).hexdigest()[:10]
    budget = max(10, max_total_len - len(str(parent)) - len(suffix) - len(digest) - 3)
    shortened_stem = stem[:budget].rstrip("_")
    return parent / f"{shortened_stem}__{digest}{suffix}"


def safe_to_csv(obj, path: Path, **kwargs) -> None:
    path = windows_safe_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    obj.to_csv(path, **kwargs)


def save_figure_object(fig, path_without_ext: Path) -> None:
    path_without_ext = windows_safe_path(path_without_ext, max_total_len=MAX_SAFE_PATH_LENGTH - 5)
    path_without_ext.parent.mkdir(parents=True, exist_ok=True)
    for ext in FIGURE_FORMATS:
        fig.savefig(f"{path_without_ext}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_figure(path_without_ext: Path) -> None:
    save_figure_object(plt.gcf(), path_without_ext)


def is_perfect(values: pd.Series, metric: str) -> pd.Series:
    if metric == "cost_raw":
        return values <= PERFECT_COST_THRESHOLD
    return values >= PERFECT_CONFORMANCE_THRESHOLD


# def _subsample(values: np.ndarray, n: int = 200_000) -> np.ndarray:
#     if len(values) <= n:
#         return values
#     return np.random.default_rng(RANDOM_SEED).choice(values, n, replace=False)


# ============================================================================ #
# DISCOVERY AND LOADING
# ============================================================================ #

@dataclass
class ConfigSource:
    name: str
    events_path: Path
    summary_path: Optional[Path]


def discover_configurations(results_dir: Path) -> Dict[str, ConfigSource]:
    """
    Recursively find every <anything>/events*/<config>.csv under results_dir.
    The configuration name is the CSV stem. The matching file-level summary is
    looked up at <parent of events dir>/summaries/file/<config>.csv.
    per_file_checkpoint/ and analysis/ folders are ignored.
    """
    sources: Dict[str, ConfigSource] = {}

    for csv_path in sorted(results_dir.rglob("*.csv")):
        rel_parts = csv_path.relative_to(results_dir).parts

        if "per_file_checkpoint" in rel_parts or "analysis" in rel_parts:
            continue
        if any(part.lower() in EXCLUDED_FOLDERS for part in rel_parts[:-1]):
            continue
        if not csv_path.parent.name.lower().startswith("event"):
            continue

        name = csv_path.stem
        if name in sources:
            print(f"  [WARN] duplicate configuration '{name}' at {csv_path}; "
                  f"keeping {sources[name].events_path}")
            continue

        run_dir = csv_path.parent.parent
        summary = run_dir / "summaries" / "file" / f"{name}.csv"

        sources[name] = ConfigSource(
            name=name,
            events_path=csv_path,
            summary_path=summary if summary.exists() else None,
        )

    if not sources:
        raise RuntimeError(f"No events*/<config>.csv files found under {results_dir}")

    return sources


def _log_files_in(path: Path) -> set:
    files: set = set()
    for chunk in pd.read_csv(path, usecols=["log_file"], chunksize=500_000):
        files.update(chunk["log_file"].unique())
    return files


def sample_log_files(sources: Dict[str, ConfigSource], n_files: Optional[int]) -> List[str]:
    """Log files present in ALL configurations (read from the events CSVs)."""
    rng = random.Random(RANDOM_SEED)
    common: Optional[set] = None

    for name, src in sources.items():
        files_here = _log_files_in(src.events_path)
        print(f"  {name}: {len(files_here)} log files")
        common = files_here if common is None else common & files_here

    if not common:
        raise RuntimeError("No log files common to all discovered configurations.")

    common = sorted(common)
    if n_files is None or len(common) <= n_files:
        return common
    return sorted(rng.sample(common, n_files))


def load_events_for_files(source: ConfigSource, wanted_files: set, columns: List[str]) -> pd.DataFrame:
    wanted_cols = set(["log_file", "event_index", "case_id", "model_name"] + columns)
    chunks = []

    for chunk in pd.read_csv(source.events_path, usecols=lambda c: c in wanted_cols, chunksize=200_000):
        filtered = chunk[chunk["log_file"].isin(wanted_files)]
        if not filtered.empty:
            chunks.append(filtered)

    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


# ============================================================================ #
# CONFIGURATION ORDERING
# ============================================================================ #

def configuration_sort_key(configuration_name: str):
    """Behavior, then TokenReplay (explicit order), then PrefixAlignment (explicit order)."""
    config_type = classify_configuration(configuration_name)
    family_order = {"token_replay": 0, "behavior": 1, "prefix_alignment": 2, "unknown": 99}
    explicit = {
        "token_replay": TOKEN_REPLAY_ORDER,
        "prefix_alignment": PREFIX_ALIGNMENT_ORDER,
    }.get(config_type, ())

    if configuration_name in explicit:
        secondary = (0, explicit.index(configuration_name))
    else:
        secondary = (1, configuration_name)

    return (family_order.get(config_type, 99), secondary)


def ordered_configurations(configurations: List[str]) -> List[str]:
    return sorted(configurations, key=configuration_sort_key)


# ============================================================================ #
# STATISTICS
# ============================================================================ #


def correlation_pair(x: pd.Series, y: pd.Series, method: str = "spearman") -> Tuple[float, float, int]:
    paired = pd.concat([x, y], axis=1).dropna()

    if len(paired) < 3:
        return np.nan, np.nan, len(paired)
    if paired.iloc[:, 0].nunique() < 2 or paired.iloc[:, 1].nunique() < 2:
        return np.nan, np.nan, len(paired)

    func = stats.spearmanr if method == "spearman" else stats.pearsonr
    r, p = func(paired.iloc[:, 0], paired.iloc[:, 1])
    return float(r), float(p), len(paired)


def correlation_matrix_with_pvalues(
    df: pd.DataFrame,
    columns: List[str],
    method: str = "spearman",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    r_matrix = pd.DataFrame(np.nan, index=columns, columns=columns)
    p_matrix = pd.DataFrame(np.nan, index=columns, columns=columns)
    n_matrix = pd.DataFrame(np.nan, index=columns, columns=columns)

    for col_a, col_b in itertools.combinations(columns, 2):
        r, p, n = correlation_pair(df[col_a], df[col_b], method)
        for m_, val in ((r_matrix, r), (p_matrix, p), (n_matrix, n)):
            m_.loc[col_a, col_b] = val
            m_.loc[col_b, col_a] = val

    for col in columns:
        r_matrix.loc[col, col] = 1.0
        p_matrix.loc[col, col] = 0.0
        n_matrix.loc[col, col] = df[col].notna().sum()

    return r_matrix, p_matrix, n_matrix


def write_correlation_outputs(
    df: pd.DataFrame,
    columns: List[str],
    metric_map: Dict[str, str],
    out_dir: Path,
    prefix: str,
    title: str,
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    results = {}
    for method in CORRELATION_METHODS:
        r, p, n = correlation_matrix_with_pvalues(df, columns, method)
        safe_to_csv(r, out_dir / f"{prefix}{method}_r.csv")
        safe_to_csv(p, out_dir / f"{prefix}{method}_p.csv")
        safe_to_csv(n, out_dir / f"{prefix}{method}_n.csv")
        plot_correlation_heatmap(
            r, p, metric_map,
            f"{title} — {METHOD_NAME[method]} correlation",
            out_dir / f"{prefix}{method}_heatmap",
            method=method,
        )
        results[method] = (r, p)
    return results

def holm_bonferroni(p_values: List[float]) -> List[float]:
    n = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(n)
    running_max = 0.0

    for rank, idx in enumerate(order):
        corrected = (n - rank) * p_values[idx]
        running_max = max(running_max, corrected)
        adjusted[idx] = min(running_max, 1.0)

    return adjusted.tolist()


def kruskal_and_pairwise(df: pd.DataFrame, metric: str, group_col: str = "configuration"):
    groups = {name: group[metric].dropna().values for name, group in df.groupby(group_col)}
    groups = {name: values for name, values in groups.items() if len(values) >= 3}

    if len(groups) < 2:
        return None, pd.DataFrame()

    omnibus = stats.kruskal(*groups.values())
    pairs = list(itertools.combinations(groups.keys(), 2))

    raw_p, effect_sizes = [], []
    for name_a, name_b in pairs:
        a, b = groups[name_a], groups[name_b]
        u_stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        effect_sizes.append(1 - (2 * u_stat / (len(a) * len(b))))
        raw_p.append(p)

    adjusted_p = holm_bonferroni(raw_p)

    pairwise_df = pd.DataFrame({
        f"{group_col}_a": [pair[0] for pair in pairs],
        f"{group_col}_b": [pair[1] for pair in pairs],
        "p_value_raw": raw_p,
        "p_value_holm": adjusted_p,
        "rank_biserial_effect": effect_sizes,
        "significant_holm_0.05": [p < 0.05 for p in adjusted_p],
    }).sort_values("p_value_holm")

    return omnibus, pairwise_df


def format_p_value(p: float) -> str:
    """Used after a '<' in plots (p<0.001, p<1.2e-05, ...)."""
    if pd.isna(p):
        return ""
    if p == 0 or p < 1e-300:
        return "0.001"
    if p < 0.001:
        return f"{p:.1e}"
    return f"{p:.3f}"


# ============================================================================ #
# CROSS-ALGORITHM WIDE TABLE
# ============================================================================ #

def build_cross_algorithm_wide_table(
    events_by_config: Dict[str, pd.DataFrame],
) -> Tuple[Optional[pd.DataFrame], List[str], Dict[str, str]]:
    """One wide table with ALL configurations; each contributes its appropriate metric."""
    selected: Dict[str, str] = {}

    for configuration_name, df in events_by_config.items():
        metric = comparison_metric_for_configuration(configuration_name)
        if metric is None or metric not in df.columns or df[metric].isna().all():
            continue
        selected[configuration_name] = metric

    if len(selected) < 2:
        return None, [], {}

    wide = None
    for configuration_name, metric in selected.items():
        series = events_by_config[configuration_name][
            ["log_file", "event_index", metric]
        ].rename(columns={metric: configuration_name})

        wide = series if wide is None else wide.merge(
            series, on=["log_file", "event_index"], how="outer"
        )

    return wide, ordered_configurations(list(selected.keys())), selected


# ============================================================================ #
# SPEARMAN HEATMAP
# ============================================================================ #

def plot_correlation_heatmap(
    r_matrix: pd.DataFrame,
    p_matrix: pd.DataFrame,
    metric_map: Dict[str, str],
    title: str,
    path: Path,
    method: str = "spearman"
) -> None:

    configs = ordered_configurations(list(r_matrix.columns))
    r_matrix = r_matrix.loc[configs, configs]
    p_matrix = p_matrix.loc[configs, configs]

    show_annot = len(configs) <= 20
    labels = [plot_label(c) for c in configs]
    n = len(configs)
    size = max(7, min(15, 0.65 * n + 4))

    fig, ax = plt.subplots(figsize=(size, size))

    # Annotations are drawn manually: larger rho, smaller p-value.
    sns.heatmap(
        r_matrix.astype(float),
        annot=False,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        square=True,
        cbar_kws={"label": f"{METHOD_NAME[method]} {METHOD_SYMBOL[method]}"},
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )

    if show_annot:
        for row_idx, i in enumerate(configs):
            for col_idx, j in enumerate(configs):
                r = r_matrix.loc[i, j]
                if pd.isna(r):
                    continue

                text_color = "white" if abs(r) > 0.55 else "black"

                if i == j:
                    ax.text(col_idx + 0.5, row_idx + 0.5, "1.00",
                            ha="center", va="center", fontsize=8, color=text_color)
                    continue

                p = p_matrix.loc[i, j]
                ax.text(col_idx + 0.5, row_idx + 0.42, f"{r:.2f}",
                        ha="center", va="center", fontsize=12, color=text_color)
                ax.text(col_idx + 0.5, row_idx + 0.68, f"p<{format_p_value(p)}",
                        ha="center", va="center", fontsize=6, color=text_color)

    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
    ax.set_yticklabels(labels, rotation=0, fontsize=7)
    highlight_heatmap(ax, configs, configs)
    ax.set_title(title, fontsize=22)

    fig.tight_layout()
    save_figure_object(fig, path)


# ============================================================================ #
# SCATTER HELPERS
# ============================================================================ #

def _bin_edges_if_needed(values: np.ndarray, max_bins: int) -> Optional[np.ndarray]:
    unique_vals = np.unique(values)
    if len(unique_vals) <= max_bins:
        return None
    lo, hi = values.min(), values.max()
    if lo == hi:
        return None
    return np.linspace(lo, hi, max_bins + 1)


def _mid(v):
    return v.mid if hasattr(v, "mid") else v


def _aggregate_points(
    valid: pd.DataFrame,
    x: str,
    y: str,
    max_bins: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Merge points into (binned) unique x/y combinations with event counts."""
    x_edges = _bin_edges_if_needed(valid[x].to_numpy(), max_bins)
    y_edges = _bin_edges_if_needed(valid[y].to_numpy(), max_bins)

    if x_edges is None and y_edges is None:
        grouped = valid.groupby([x, y]).size().reset_index(name="event_count")
        return (
            grouped[x].to_numpy(dtype=float),
            grouped[y].to_numpy(dtype=float),
            grouped["event_count"].to_numpy(),
            False,
        )

    xb = pd.cut(valid[x], bins=x_edges, include_lowest=True) if x_edges is not None else valid[x]
    yb = pd.cut(valid[y], bins=y_edges, include_lowest=True) if y_edges is not None else valid[y]

    grouped = (
        valid.assign(_xb=xb, _yb=yb)
        .groupby(["_xb", "_yb"], observed=True)
        .size()
        .reset_index(name="event_count")
    )
    grouped = grouped[grouped["event_count"] > 0]

    return (
        grouped["_xb"].apply(_mid).astype(float).to_numpy(),
        grouped["_yb"].apply(_mid).astype(float).to_numpy(),
        grouped["event_count"].to_numpy(),
        True,
    )


def _size_for_count(counts: np.ndarray, legend_counts: List[int], min_size: float, max_size: float) -> np.ndarray:
    max_legend = max(legend_counts)
    counts = np.asarray(counts, dtype=float)
    sizes = max_size * np.sqrt(np.clip(counts, 0, max_legend) / max_legend)
    return np.clip(sizes, min_size, max_size)


def _size_legend_handles(legend_counts: List[int], min_size: float, max_size: float):
    handles = [
        plt.scatter(
            [], [],
            s=_size_for_count(np.array([c]), legend_counts, min_size, max_size)[0],
            alpha=0.55, edgecolors="none",
        )
        for c in legend_counts
    ]
    labels = [f"{c:,} events" for c in legend_counts]
    return handles, labels


def _fit_weighted_trend_line(x: np.ndarray, y: np.ndarray, weights: np.ndarray):
    if len(x) < 2 or np.ptp(x) == 0:
        return None
    try:
        slope, intercept = np.polyfit(x, y, deg=1, w=weights)
    except (np.linalg.LinAlgError, ValueError):
        return None

    x_line = np.array([x.min(), x.max()])
    y_line = slope * x_line + intercept
    direction = "increasing" if slope > 0 else "decreasing" if slope < 0 else "flat"
    return x_line, y_line, direction


# ============================================================================ #
# SINGLE SCATTER
# ============================================================================ #

def plot_scatter_sized_by_count(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    path: Path,
    legend_counts: List[int],
    rho: Optional[float] = None,
    p_value: Optional[float] = None,
    max_bins: int = MAX_SCATTER_BINS,
    min_size: float = SCATTER_MIN_SIZE,
    max_size: float = SCATTER_MAX_SIZE,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
) -> None:

    if SKIP_SCATTER:
        return

    valid = df[[x, y]].dropna()
    if len(valid) < 5:
        return

    plot_x, plot_y, counts, binned = _aggregate_points(valid, x, y, max_bins)
    merge_note = f"; values merged into up to {max_bins}×{max_bins} bins" if binned else ""
    sizes = _size_for_count(counts, legend_counts, min_size, max_size)

    fig, ax = plt.subplots(figsize=(7.0, 7.0))
    ax.scatter(plot_x, plot_y, s=sizes, alpha=0.55, edgecolors="none")

    handles, labels = _size_legend_handles(legend_counts, min_size, max_size)

    trend = _fit_weighted_trend_line(plot_x, plot_y, counts)
    if trend is not None:
        x_line, y_line, direction = trend
        trend_handle, = ax.plot(x_line, y_line, color="red", linestyle="--", linewidth=1.8)
        handles.append(trend_handle)
        labels.append(f"trend line ({direction})")

    ax.set_xlabel(xlabel if xlabel else plot_label(x), fontsize=9)
    ax.set_ylabel(ylabel if ylabel else plot_label(y), fontsize=9)
    ax.tick_params(axis="both", labelsize=8)

    if rho is not None and p_value is not None:
        title = f"{title}\nSpearman rho={rho:.3f}, p<{format_p_value(p_value)}"

    ax.set_title(f"{title}\n(n={len(valid):,} events{merge_note})", fontsize=22)
    ax.set_box_aspect(1)

    ax.legend(
        handles, labels,
        title="Dot size = event count",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
        frameon=True,
        labelspacing=1.4,
        fontsize=7,
        title_fontsize=7.5,
    )

    fig.tight_layout()
    save_figure_object(fig, path)


# ============================================================================ #
# REFERENCE CONFIG VS TOKEN REPLAY SUBPLOTS
# ============================================================================ #

def plot_reference_vs_token_replay_subplots(
    wide: pd.DataFrame,
    reference_config: str,
    token_configs: List[str],
    r_matrix: pd.DataFrame,
    p_matrix: pd.DataFrame,
    output_path: Path,
    suptitle: Optional[str] = None,
) -> None:
    """One subplot per TokenReplay configuration: reference metric vs TokenReplay conformance."""

    if not token_configs:
        return

    n = len(token_configs)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.5 * nrows), squeeze=False)
    axes = axes.flatten()

    reference_label = plot_label(reference_config)

    for idx, token_config in enumerate(token_configs):
        ax = axes[idx]

        sub = wide[[reference_config, token_config]].dropna()
        if len(sub) < 5:
            ax.axis("off")
            continue

        tmp = sub.rename(columns={reference_config: "_x", token_config: "_y"})
        plot_x, plot_y, counts, _ = _aggregate_points(tmp, "_x", "_y", MAX_SCATTER_BINS)

        sizes = _size_for_count(counts, SCATTER_SIZE_LEGEND_FULL, SCATTER_MIN_SIZE, SCATTER_MAX_SIZE)
        ax.scatter(plot_x, plot_y, s=sizes, alpha=0.55, edgecolors="none")

        trend = _fit_weighted_trend_line(plot_x, plot_y, counts)
        if trend is not None:
            x_line, y_line, _ = trend
            ax.plot(x_line, y_line, color="red", linestyle="--", linewidth=1.7)

        r = r_matrix.loc[reference_config, token_config]
        p = p_matrix.loc[reference_config, token_config]

        ax.set_title(f"{plot_label(token_config)}\nrho={r:.3f}, p<{format_p_value(p)}", fontsize=9)
        ax.set_xlabel(reference_label, fontsize=8)
        ax.set_ylabel(plot_label(token_config), fontsize=8)
        ax.set_ylim(0, 1)
        ax.tick_params(labelsize=7)
        ax.set_box_aspect(1)

    for idx in range(len(token_configs), len(axes)):
        axes[idx].axis("off")

    fig.suptitle(
        suptitle if suptitle is not None else f"{reference_label} vs TokenReplay configurations",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure_object(fig, output_path)


# ============================================================================ #
# PERFECT AGREEMENT
# ============================================================================ #

def perfect_agreement_table(
    wide: pd.DataFrame,
    configs_with_metric: List[str],
    metric_map: Dict[str, str],
) -> pd.DataFrame:

    rows = []

    for config_a, config_b in itertools.combinations(configs_with_metric, 2):
        metric_a, metric_b = metric_map[config_a], metric_map[config_b]

        sub = wide[[config_a, config_b]].dropna()
        n = len(sub)
        if n == 0:
            continue

        a_perfect = is_perfect(sub[config_a], metric_a)
        b_perfect = is_perfect(sub[config_b], metric_b)

        both = int((a_perfect & b_perfect).sum())
        only_a = int((a_perfect & ~b_perfect).sum())
        only_b = int((~a_perfect & b_perfect).sum())
        neither = int((~a_perfect & ~b_perfect).sum())

        rows.append({
            "config_a": config_a,
            "config_b": config_b,
            "metric_a": metric_a,
            "metric_b": metric_b,
            "n_joint_events": n,
            "both_perfect": both,
            "pct_both_perfect": 100.0 * both / n,
            "only_config_a_perfect": only_a,
            "only_config_b_perfect": only_b,
            "neither_perfect": neither,
        })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("pct_both_perfect", ascending=False)


def plot_perfect_confusion_matrix(
    config_a: str, config_b: str,
    both: int, only_a: int, only_b: int, neither: int,
    metric_a: str, metric_b: str,
    path: Path,
) -> None:

    matrix = np.array([[both, only_a], [only_b, neither]])
    total = matrix.sum()
    pct_both = 100.0 * both / total if total else 0.0

    fig, ax = plt.subplots(figsize=(5.2, 4.4))

    sns.heatmap(
        matrix, annot=True, fmt="d", cmap="Greens",
        xticklabels=[f"{plot_label(config_b)}: perfect", f"{plot_label(config_b)}: not perfect"],
        yticklabels=[f"{plot_label(config_a)}: perfect", f"{plot_label(config_a)}: not perfect"],
        cbar_kws={"label": "event count"},
        ax=ax,
    )

    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=7)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=7)
    highlight_heatmap(ax, [config_a, config_a], [config_b, config_b])

    ax.set_title(
        f"Perfect agreement\n"
        f"{plot_label(config_a)} ({metric_label(metric_a)}) vs "
        f"{plot_label(config_b)} ({metric_label(metric_b)})\n"
        f"{pct_both:.1f}% of {total:,} joint events perfect on both",
        fontsize=9,
    )

    fig.tight_layout()
    save_figure_object(fig, path)


# ============================================================================ #
# CROSS-ALGORITHM CORRELATION
# ============================================================================ #

def _both_prefix(config_a: str, config_b: str) -> bool:
    return (classify_configuration(config_a) == "prefix_alignment"
            and classify_configuration(config_b) == "prefix_alignment")


def cross_algorithm_event_correlation(events_by_config: Dict[str, pd.DataFrame], output_dir: Path) -> None:

    print("\n=== Cross-algorithm event-level correlation ===")

    out_dir = output_dir / "cross_algorithm_event_correlation"
    out_dir.mkdir(parents=True, exist_ok=True)

    wide, configs, metric_map = build_cross_algorithm_wide_table(events_by_config)

    if wide is None or len(configs) < 2:
        print("  Not enough configurations for cross-algorithm analysis.")
        return

    print(f"  {len(configs)} configurations, {len(wide):,} joined events")
    print("  Metric assignment:")
    for config in configs:
        print(f"    {config}: {metric_map[config]}")

    # Spearman matrices
    corr = write_correlation_outputs(
        wide, configs, metric_map, out_dir,
        prefix="all_configurations__", title="All configurations",
    )
    r_matrix, p_matrix = corr["spearman"]   # scatter plots / subplots keep using Spearman

    # Main heatmap
    plot_correlation_heatmap(
        r_matrix, p_matrix, metric_map,
        "All configurations — Spearman correlation",
        out_dir / "all_configurations__spearman_heatmap",
    )

    behavior_configs = [c for c in configs if classify_configuration(c) == "behavior"]
    token_configs = [c for c in configs if classify_configuration(c) == "token_replay"]

    if behavior_configs and token_configs:
        plot_reference_vs_token_replay_subplots(
            wide, behavior_configs[0], token_configs, r_matrix, p_matrix,
            out_dir / "behavior_vs_token_replay__subplots",
            suptitle="Behavior vs TokenReplay configurations",
        )

    prefix20 = [c for c in configs if c == "PrefixAlignment_revert_20"]
    if prefix20 and token_configs:
        plot_reference_vs_token_replay_subplots(
            wide, prefix20[0], token_configs, r_matrix, p_matrix,
            out_dir / "prefix20_vs_token_replay__subplots",
            suptitle="Prefix_20 (cost) vs TokenReplay configurations",
        )

    # All pairwise scatter plots (prefix-vs-prefix skipped)
    if not SKIP_SCATTER:
        all_pairs = list(itertools.combinations(configs, 2))
        print(f"  Scatter plots: evaluating {len(all_pairs)} configuration pairs")

        for config_a, config_b in all_pairs:
            if _both_prefix(config_a, config_b):
                continue

            r = r_matrix.loc[config_a, config_b]
            p = p_matrix.loc[config_a, config_b]
            if pd.isna(r):
                continue

            metric_a, metric_b = metric_map[config_a], metric_map[config_b]

            plot_scatter_sized_by_count(
                wide, config_a, config_b,
                f"{plot_label(config_a)} vs {plot_label(config_b)}",
                out_dir / f"scatter_all__{config_a}__vs__{config_b}",
                legend_counts=SCATTER_SIZE_LEGEND_FULL,
                rho=r, p_value=p,
                xlabel=f"{plot_label(config_a)} ({metric_label(metric_a)})",
                ylabel=f"{plot_label(config_b)} ({metric_label(metric_b)})",
            )

    # Perfect agreement
    agreement_table = perfect_agreement_table(wide, configs, metric_map)

    if agreement_table.empty:
        return

    keep = ~agreement_table.apply(lambda row: _both_prefix(row["config_a"], row["config_b"]), axis=1)
    agreement_table = agreement_table[keep]

    safe_to_csv(agreement_table, out_dir / "perfect_agreement_summary.csv", index=False)

    confusion_dir = out_dir / "perfect_confusion_matrices"
    non_perfect_dir = out_dir / "non_perfect_scatter"

    for _, row in agreement_table.iterrows():
        config_a, config_b = row["config_a"], row["config_b"]
        metric_a, metric_b = row["metric_a"], row["metric_b"]

        plot_perfect_confusion_matrix(
            config_a, config_b,
            int(row["both_perfect"]), int(row["only_config_a_perfect"]),
            int(row["only_config_b_perfect"]), int(row["neither_perfect"]),
            metric_a, metric_b,
            confusion_dir / f"{config_a}__vs__{config_b}",
        )

        if SKIP_SCATTER:
            continue

        sub = wide[[config_a, config_b]].dropna()
        a_perfect = is_perfect(sub[config_a], metric_a)
        b_perfect = is_perfect(sub[config_b], metric_b)
        disagreement_subset = sub[~(a_perfect & b_perfect)]

        plot_scatter_sized_by_count(
            disagreement_subset, config_a, config_b,
            f"{plot_label(config_a)} vs {plot_label(config_b)} — at least one event is not perfect",
            non_perfect_dir / f"{config_a}__vs__{config_b}",
            legend_counts=SCATTER_SIZE_LEGEND_FILTERED,
            rho=r_matrix.loc[config_a, config_b],
            p_value=p_matrix.loc[config_a, config_b],
            xlabel=f"{plot_label(config_a)} ({metric_label(metric_a)})",
            ylabel=f"{plot_label(config_b)} ({metric_label(metric_b)})",
        )


# ============================================================================ #
# RAW FITNESS DISTRIBUTIONS FOR THE TOKEN-REPLAY (REMAINING) VARIANTS
# ============================================================================ #


from matplotlib.patches import Rectangle

HIGHLIGHT_CONFIGS: Tuple[str, ...] = ("TokenReplay_remaining_BRD", "TokenReplay_remaining_MFS")


def _matrix_heatmap(matrix: pd.DataFrame, title: str, cbar_label: str, path: Path,
                    fmt: str = ".3f", cmap: str = "viridis",
                    title_size: int = 16, label_size: int = 15, annot_size: int = 14) -> None:
    cols = list(matrix.columns)
    n = len(cols)
    labels = [plot_label(c) for c in cols]

    fig, ax = plt.subplots(figsize=(1.5 * n + 4, 1.3 * n + 3.5))
    sns.heatmap(matrix.astype(float), annot=True, fmt=fmt, cmap=cmap, square=True,
                annot_kws={"fontsize": annot_size},
                cbar_kws={"label": cbar_label},
                xticklabels=labels, yticklabels=labels, ax=ax)

    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=label_size)
    ax.set_yticklabels(labels, rotation=0, fontsize=label_size)
    ax.set_title(title, fontsize=title_size)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=label_size - 2)
    cbar.set_label(cbar_label, fontsize=label_size)

    highlight_heatmap(ax, cols, cols)

    fig.tight_layout()
    save_figure_object(fig, path)


# Variants whose fitness is bounded to [0.5, 1]; mapped linearly to [0, 1] for plotting.
HALF_RANGE_RESCALE: Tuple[str, ...] = ("TokenReplay_remaining_IGNORE",)
HALF_RANGE_LOW = 0.5


def rescale_for_plot(config: str, v: np.ndarray) -> np.ndarray:
    if config in HALF_RANGE_RESCALE:
        return (v - HALF_RANGE_LOW) / (1.0 - HALF_RANGE_LOW)   # = 2v - 1
    return v

def remaining_variant_distributions(events_by_config: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    """
    Compare the REAL (raw) fitness values of the TokenReplay variants.
    Equal Spearman correlation does not imply equal values: rank correlation
    ignores scale, shifts and ties, so the values are compared directly.
    """
    print("\n=== Raw fitness distributions (TokenReplay variants) ===")

    metric = "conformance_raw"
    configs = ordered_configurations([
        c for c in events_by_config
        if classify_configuration(c) == "token_replay" and metric in events_by_config[c].columns
    ])

    if len(configs) < 2:
        print("  Fewer than 2 TokenReplay variants; skipping.")
        return

    out_dir = output_dir / "remaining_raw_fitness_distributions"
    out_dir.mkdir(parents=True, exist_ok=True)

    values = {c: events_by_config[c][metric].dropna().to_numpy(dtype=float) for c in configs}
    palette = dict(zip(configs, sns.color_palette("deep", len(configs))))

    # 1. Summary statistics
    rows = []
    for c in configs:
        v = values[c]
        q = np.quantile(v, [0.05, 0.25, 0.5, 0.75, 0.95])
        rows.append({
            "configuration": c, "label": plot_label(c), "n": len(v),
            "mean": v.mean(), "std": v.std(), "min": v.min(),
            "q05": q[0], "q25": q[1], "median": q[2], "q75": q[3], "q95": q[4],
            "max": v.max(),
            f"pct_perfect(>={PERFECT_CONFORMANCE_THRESHOLD})":
                100.0 * np.mean(v >= PERFECT_CONFORMANCE_THRESHOLD),
            "pct_zero": 100.0 * np.mean(v <= 1e-9),
            "n_unique_values": len(np.unique(v)),
        })
    summary = pd.DataFrame(rows)
    safe_to_csv(summary, out_dir / "raw_fitness_summary.csv", index=False)
    print(summary[["label", "n", "mean", "median", "std", "min"]].to_string(index=False))

    # 2. Histogram grid (shared bins and axes)
    lo = min(v.min() for v in values.values())
    hi = max(v.max() for v in values.values())
    bins = np.linspace(lo, hi, 51)

    ncols = min(2, len(configs))
    nrows = int(np.ceil(len(configs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.2 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    axes = axes.flatten()
    for ax, c in zip(axes, configs):
        counts, _ = np.histogram(values[c], bins=bins)
        ax.bar(bins[:-1], counts / counts.sum(), width=np.diff(bins), align="edge",
               color=palette[c], alpha=0.85)
        ax.axvline(np.median(values[c]), color="red", linestyle="--", linewidth=1.2)
        ax.set_title(f"{plot_label(c)}  (mean={values[c].mean():.3f}, "
                     f"median={np.median(values[c]):.3f})", fontsize=9)
        ax.set_xlabel("Raw fitness", fontsize=8)
        ax.set_ylabel("Fraction of events", fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes[len(configs):]:
        ax.axis("off")
    fig.suptitle("Distribution of raw fitness per TokenReplay variant (red = median)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure_object(fig, out_dir / "raw_fitness_histograms")

    # 3. Overlaid ECDF
    qs = np.linspace(0, 1, 1001)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for c in configs:
        ax.plot(np.quantile(values[c], qs), qs, label=plot_label(c), color=palette[c], linewidth=1.8)
    ax.set_xlabel("Raw fitness", fontsize=9)
    ax.set_ylabel("Cumulative fraction of events", fontsize=9)
    ax.set_title("ECDF of raw fitness (TokenReplay variants)", fontsize=11)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure_object(fig, out_dir / "raw_fitness_ecdf")

    # 4. Violin plot (IGN [0.5,1] mapped linearly to [0,1] for comparability)
    violin_parts, scale_rows = [], []
    for c in configs:
        v = values[c]
        rescaled = c in HALF_RANGE_RESCALE
        v_plot = rescale_for_plot(c, v)
        scale_rows.append({"configuration": c, "label": plot_label(c),
                           "raw_min": float(v.min()), "raw_max": float(v.max()),
                           "rescaled_0.5_1_to_0_1": rescaled})
        violin_parts.append(pd.DataFrame({
            "configuration": plot_label(c) + ("*" if rescaled else ""),
            metric: _subsample(v_plot,n = None),
        }))
    safe_to_csv(pd.DataFrame(scale_rows), out_dir / "violin_rescaling_info.csv", index=False)

    long = pd.concat(violin_parts, ignore_index=True)
    order = [plot_label(c) + ("*" if c in HALF_RANGE_RESCALE else "") for c in configs]
    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(configs)), 5))
    sns.violinplot(data=long, x="configuration", y=metric, order=order,
                   cut=0, inner="quartile", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fitness")
    ax.set_xlabel("")
    ax.set_title("Fitness distribution (violin)\n"
                 "* rescaled from [0.5, 1] to [0, 1]", fontsize=10)
    fig.tight_layout()
    save_figure_object(fig, out_dir / "raw_fitness_violin")


    # 5. Paired, event-by-event differences
    frames = [
        events_by_config[c][["log_file", "event_index", metric]].rename(columns={metric: c})
        for c in configs
    ]
    wide = reduce(lambda a, b: a.merge(b, on=["log_file", "event_index"], how="inner"), frames).dropna()
    print(f"  {len(wide):,} events present in all variants")

    pair_rows = []
    mad = pd.DataFrame(0.0, index=configs, columns=configs)
    pct_diff = pd.DataFrame(0.0, index=configs, columns=configs)
    ks = pd.DataFrame(0.0, index=configs, columns=configs)

    for a, b in itertools.combinations(configs, 2):
        d = wide[a].to_numpy() - wide[b].to_numpy()
        abs_d = np.abs(d)
        ks_stat = stats.ks_2samp(_subsample(wide[a].to_numpy(),n = None), _subsample(wide[b].to_numpy(),n = None)).statistic
        row = {
            "config_a": a, "config_b": b, "n_events": len(d),
            "mean_diff_a_minus_b": d.mean(),
            "mean_abs_diff": abs_d.mean(),
            "median_abs_diff": np.median(abs_d),
            "max_abs_diff": abs_d.max(),
            "pct_events_identical(<1e-9)": 100.0 * np.mean(abs_d < 1e-9),
            "pct_events_different": 100.0 * np.mean(abs_d >= 1e-9),
            "pct_a_higher": 100.0 * np.mean(d > 1e-9),
            "pct_b_higher": 100.0 * np.mean(d < -1e-9),
            "ks_statistic": ks_stat,
        }
        pair_rows.append(row)
        for m_, val in ((mad, row["mean_abs_diff"]),
                        (pct_diff, row["pct_events_different"]),
                        (ks, ks_stat)):
            m_.loc[a, b] = val
            m_.loc[b, a] = val

    safe_to_csv(pd.DataFrame(pair_rows), out_dir / "paired_value_differences.csv", index=False)

    _matrix_heatmap(mad, "Mean |fitness difference| per event", "mean abs diff",
                    out_dir / "paired_mean_abs_difference_heatmap", fmt=".4f")
    _matrix_heatmap(pct_diff, "% of events where fitness differs", "% events",
                    out_dir / "paired_pct_events_different_heatmap", fmt=".1f", title_size=22)
    _matrix_heatmap(ks, "Two-sample KS distance between fitness distributions", "KS D",
                    out_dir / "ks_distance_heatmap", fmt=".3f")

        # 5b. Same comparison with IGN rescaled from [0.5, 1] to [0, 1]
    wide_scaled = wide.copy()
    for c in configs:
        wide_scaled[c] = rescale_for_plot(c, wide[c].to_numpy(dtype=float))

    DIFF_TOL = 1e-9   # raise (e.g. 1e-3) to ignore tiny numerical differences
    pct_diff_scaled = pd.DataFrame(0.0, index=configs, columns=configs)
    scaled_rows = []

    for a, b in itertools.combinations(configs, 2):
        abs_d = np.abs(wide_scaled[a].to_numpy() - wide_scaled[b].to_numpy())
        pct = 100.0 * np.mean(abs_d >= DIFF_TOL)
        pct_diff_scaled.loc[a, b] = pct
        pct_diff_scaled.loc[b, a] = pct
        scaled_rows.append({
            "config_a": a, "config_b": b, "n_events": len(abs_d),
            "pct_events_different_scaled": pct,
            "mean_abs_diff_scaled": abs_d.mean(),
        })

    safe_to_csv(pd.DataFrame(scaled_rows), out_dir / "paired_value_differences_scaled.csv", index=False)

    _matrix_heatmap(
        pct_diff_scaled,
        "% of events where fitness differs\n(TR-IGN rescaled from [0.5, 1] to [0, 1])",
        "% events",
        out_dir / "paired_pct_events_different_scaled_heatmap",
        fmt=".1f",
        title_size=22
    )

    # 5c. Fraction of events with estimated remaining tokens r > 0 (inferred from IGN - variant)
    ign = "TokenReplay_remaining_IGNORE"
    if ign in configs:
        R_TOL = 1e-9
        others = [c for c in configs if c != ign]
        rem_rows, implied = [], {}

        for c in others:
            diff = wide[ign].to_numpy(dtype=float) - wide[c].to_numpy(dtype=float)
            r_pos = diff > R_TOL
            implied[c] = 2.0 * diff[r_pos]                      # implied r/p where r > 0
            rem_rows.append({
                "configuration": c, "label": plot_label(c), "n_events": len(diff),
                "pct_events_r_gt_0": 100.0 * r_pos.mean(),
                "pct_events_variant_above_ign": 100.0 * np.mean(diff < -R_TOL),  # sanity: should be ~0
                "mean_implied_r_over_p_when_r_gt_0": implied[c].mean() if r_pos.any() else 0.0,
                "median_implied_r_over_p_when_r_gt_0": np.median(implied[c]) if r_pos.any() else 0.0,
                "mean_fitness_drop_vs_ign_all_events": diff.mean(),
            })

        rem_df = pd.DataFrame(rem_rows)
        safe_to_csv(rem_df, out_dir / "estimated_remaining_tokens_vs_ign.csv", index=False)
        print("\n  Events with estimated remaining tokens r > 0 (inferred from IGN - variant):")
        print(rem_df[["label", "n_events", "pct_events_r_gt_0",
                      "pct_events_variant_above_ign",
                      "mean_implied_r_over_p_when_r_gt_0"]].to_string(index=False))

        # Bar chart: % of events with r > 0
        fig, ax = plt.subplots(figsize=(max(5, 1.6 * len(others)), 4.5))
        bars = ax.bar([plot_label(c) for c in others], rem_df["pct_events_r_gt_0"],
                      color=[palette[c] for c in others])
        for bar, val in zip(bars, rem_df["pct_events_r_gt_0"]):
            ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.1f}%",
                    ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("% of events")
        ax.set_ylim(0, max(5.0, rem_df["pct_events_r_gt_0"].max() * 1.15))
        ax.set_title("Events with estimated remaining tokens r > 0\n(variant fitness < IGN fitness)", fontsize=10)
        fig.tight_layout()
        save_figure_object(fig, out_dir / "pct_events_remaining_gt_0")

        # Histogram: size of the penalty (implied r/p) when r > 0
        with_r = [c for c in others if len(implied[c]) > 0]
        if with_r:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            bins = np.linspace(0, 1, 41)
            for c in with_r:
                ax.hist(implied[c], bins=bins, alpha=0.55, label=plot_label(c),
                        color=palette[c], weights=np.full(len(implied[c]), 1.0 / len(implied[c])))
            ax.set_xlabel("Implied r/p (events with r > 0)")
            ax.set_ylabel("Fraction of those events")
            ax.set_title("Size of the remaining-token penalty", fontsize=10)
            ax.legend(fontsize=8)
            fig.tight_layout()
            save_figure_object(fig, out_dir / "implied_remaining_ratio_hist")
    # 6. Per-model boxplots
    if all("model_name" in events_by_config[c].columns for c in configs):
        parts = []
        for c in configs:
            df = events_by_config[c][["model_name", metric]].dropna()
            if len(df) > 400_000:
                df = df.sample(400_000, random_state=RANDOM_SEED)
            parts.append(df.assign(variant=plot_label(c)))
        per_model = pd.concat(parts, ignore_index=True)

        models = sorted(per_model["model_name"].unique())
        fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(models) * len(configs) / 2), 5.5))
        sns.boxplot(data=per_model, x="model_name", y=metric, hue="variant",
                    order=models, hue_order=[plot_label(c) for c in configs],
                    showfliers=False, ax=ax)
        ax.set_xlabel("Model")
        ax.set_ylabel("Raw fitness")
        ax.set_title("Raw fitness per model and TokenReplay variant", fontsize=11)
        plt.setp(ax.get_xticklabels(), rotation=60, ha="right", fontsize=7)
        ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
        fig.tight_layout()
        save_figure_object(fig, out_dir / "raw_fitness_per_model_boxplot")

        per_model_mean = per_model.groupby(["model_name", "variant"])[metric].mean().unstack("variant")
        safe_to_csv(per_model_mean, out_dir / "raw_fitness_mean_per_model.csv")


# ============================================================================ #
# PER-MODEL CORRELATION
# ============================================================================ #

def per_model_event_correlation(events_by_config: Dict[str, pd.DataFrame], output_dir: Path) -> None:

    print("\n=== Per-model event-level correlation ===")

    out_dir = output_dir / "per_model_event_correlation"

    models: set = set()
    for df in events_by_config.values():
        if "model_name" in df.columns:
            models.update(df["model_name"].dropna().unique())

    if not models:
        print("  No model_name column found.")
        return

    for model_name in sorted(models):
        model_out_dir = out_dir / str(model_name)
        model_out_dir.mkdir(parents=True, exist_ok=True)

        model_events = {}
        for config_name, df in events_by_config.items():
            if "model_name" not in df.columns:
                continue
            sub = df[df["model_name"] == model_name]
            if not sub.empty:
                model_events[config_name] = sub

        wide, configs, metric_map = build_cross_algorithm_wide_table(model_events)
        if wide is None or len(configs) < 2:
            continue

        print(f"  model={model_name}: {len(configs)} configurations, {len(wide):,} joined events")

        corr = write_correlation_outputs(
            wide, configs, metric_map, model_out_dir,
            prefix="", title=str(model_name),
        )
        r_matrix, p_matrix = corr["spearman"]

        plot_correlation_heatmap(
            r_matrix, p_matrix, metric_map,
            f"{model_name} — all configurations — Spearman correlation",
            model_out_dir / "spearman_heatmap",
        )

        behavior_configs = [c for c in configs if classify_configuration(c) == "behavior"]
        token_configs = [c for c in configs if classify_configuration(c) == "token_replay"]

        if behavior_configs and token_configs:
            plot_reference_vs_token_replay_subplots(
                wide, behavior_configs[0], token_configs, r_matrix, p_matrix,
                model_out_dir / "behavior_vs_token_replay__subplots",
                suptitle=f"{model_name}: Behavior vs TokenReplay configurations",
            )

        prefix20 = [c for c in configs if c == "PrefixAlignment_revert_20"]
        if prefix20 and token_configs:
            plot_reference_vs_token_replay_subplots(
                wide, prefix20[0], token_configs, r_matrix, p_matrix,
                model_out_dir / "prefix20_vs_token_replay__subplots",
                suptitle=f"{model_name}: Prefix_20 (cost) vs TokenReplay configurations",
            )

        if SKIP_SCATTER:
            continue

        for config_a, config_b in itertools.combinations(configs, 2):
            if _both_prefix(config_a, config_b):
                continue

            r = r_matrix.loc[config_a, config_b]
            p = p_matrix.loc[config_a, config_b]
            if pd.isna(r):
                continue

            metric_a, metric_b = metric_map[config_a], metric_map[config_b]

            plot_scatter_sized_by_count(
                wide, config_a, config_b,
                f"{model_name}\n{plot_label(config_a)} vs {plot_label(config_b)}",
                model_out_dir / f"scatter_all__{config_a}__vs__{config_b}",
                legend_counts=SCATTER_SIZE_LEGEND_FULL,
                rho=r, p_value=p,
                xlabel=f"{plot_label(config_a)} ({metric_label(metric_a)})",
                ylabel=f"{plot_label(config_b)} ({metric_label(metric_b)})",
            )


# ============================================================================ #
# WITHIN-CONFIGURATION CORRELATION
# ============================================================================ #

def within_configuration_event_correlation(events_by_config: Dict[str, pd.DataFrame], output_dir: Path) -> None:

    print("\n=== Within-configuration event-level correlation ===")

    out_dir = output_dir / "within_configuration_event_correlation"
    out_dir.mkdir(parents=True, exist_ok=True)

    for configuration_name, df in events_by_config.items():
        columns = [c for c in EVENT_METRIC_COLUMNS if c in df.columns and df[c].notna().any()]
        if len(columns) < 2:
            continue

        print(f"  {configuration_name}: metrics={columns}, n={len(df):,}")

        corr = write_correlation_outputs(
            df, columns, {c: c for c in columns}, out_dir,
            prefix=f"{configuration_name}__",
            title=f"{plot_label(configuration_name)} — within-configuration",
        )
        r_matrix, p_matrix = corr["spearman"]

        plot_correlation_heatmap(
            r_matrix, p_matrix, {c: c for c in columns},
            f"{plot_label(configuration_name)} — within-configuration Spearman correlation",
            out_dir / f"{configuration_name}__spearman_heatmap",
        )

        if SKIP_SCATTER:
            continue

        for col_a, col_b in itertools.combinations(columns, 2):
            r = r_matrix.loc[col_a, col_b]
            p = p_matrix.loc[col_a, col_b]
            if pd.isna(r):
                continue

            plot_scatter_sized_by_count(
                df, col_a, col_b,
                f"{plot_label(configuration_name)}: {col_a} vs {col_b}",
                out_dir / f"{configuration_name}__{col_a}__vs__{col_b}",
                legend_counts=SCATTER_SIZE_LEGEND_FULL,
                rho=r, p_value=p,
                xlabel=col_a, ylabel=col_b,
            )


# ============================================================================ #
# PERFORMANCE PLOTS
# ============================================================================ #

def plot_performance_summary(file_df: pd.DataFrame, metric: str, title: str, path: Path) -> None:
    """Median + interquartile range + individual file-level observations."""

    data = file_df[["configuration", metric]].dropna()
    if data.empty:
        return

    order = sorted(data["configuration"].unique(), key=configuration_sort_key)
    positions = np.arange(len(order))

    medians, q1s, q3s = [], [], []
    for config in order:
        vals = data.loc[data["configuration"] == config, metric].to_numpy()
        medians.append(np.median(vals))
        q1s.append(np.percentile(vals, 25))
        q3s.append(np.percentile(vals, 75))

    fig, ax = plt.subplots(figsize=(max(8, 0.75 * len(order)), 5.5))

    ax.vlines(positions, q1s, q3s, linewidth=7, alpha=0.65)
    ax.scatter(positions, medians, s=35, zorder=3, label="Median")

    for i, config in enumerate(order):
        vals = data.loc[data["configuration"] == config, metric].to_numpy()
        jitter = np.random.default_rng(RANDOM_SEED + i).uniform(-0.08, 0.08, len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=7, alpha=0.25, edgecolors="none")

    ax.set_xticks(positions)
    ax.set_xticklabels([plot_label(c) for c in order], rotation=55, ha="right", fontsize=8)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8)

    fig.tight_layout()
    save_figure_object(fig, path)


def performance_comparison(
    file_df: pd.DataFrame,
    events_by_config: Dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:

    print("\n=== Performance comparison ===")

    out_dir = output_dir / "performance"
    out_dir.mkdir(parents=True, exist_ok=True)

    for metric in PERFORMANCE_METRICS:
        if metric not in file_df.columns or file_df[metric].isna().all():
            continue

        plot_performance_summary(
            file_df, metric,
            f"{metric.replace('_', ' ').title()} by configuration",
            out_dir / f"{metric}__summary_plot",
        )

        omnibus, pairwise_df = kruskal_and_pairwise(file_df, metric)
        if omnibus is None:
            continue

        print(f"  {metric}: Kruskal-Wallis H={omnibus.statistic:.2f}, "
              f"p<{format_p_value(omnibus.pvalue)}")

        safe_to_csv(pairwise_df, out_dir / f"{metric}__pairwise_tests.csv", index=False)

    available_metrics = [m for m in PERFORMANCE_METRICS if m in file_df.columns]
    if available_metrics:
        summary = file_df.groupby("configuration")[available_metrics].agg(["mean", "median", "std"])
        safe_to_csv(summary, out_dir / "performance_summary_by_configuration.csv")

    # Event-level processing-time violin
    parts = [df.assign(configuration=name) for name, df in events_by_config.items() if not df.empty]
    if not parts:
        return

    combined_events = pd.concat(parts, ignore_index=True)
    if "total_time_ms" not in combined_events.columns:
        return

    data = combined_events[["configuration", "total_time_ms"]].dropna()
    order = sorted(data["configuration"].unique(), key=configuration_sort_key)

    fig, ax = plt.subplots(figsize=(max(8, 0.7 * len(order)), 5))
    sns.violinplot(data=data, x="configuration", y="total_time_ms",
                   order=order, cut=0, inner="quartile", ax=ax)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([plot_label(c) for c in order], rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("Processing time per event (ms)")
    ax.set_xlabel("Configuration")
    ax.set_title("Distribution of event processing time")

    fig.tight_layout()
    save_figure_object(fig, out_dir / "event_level_total_time_ms__violin")


# ============================================================================ #
# MODEL VS CONFIGURATION DIFFERENCES
# ============================================================================ #

def model_configuration_differences(file_df: pd.DataFrame, output_dir: Path) -> None:

    print("\n=== Model vs configuration differences ===")

    out_dir = output_dir / "model_vs_configuration"
    out_dir.mkdir(parents=True, exist_ok=True)

    working = file_df.copy()
    working["configuration_type"] = working["configuration"].map(classify_configuration)

    # Behavior / TokenReplay -> mean_conformance ; PrefixAlignment -> mean_cost
    for config_type, metric_name in [("conformance", "mean_conformance"), ("cost", "mean_cost")]:

        if metric_name not in working.columns:
            continue

        allowed_types = ["behavior", "token_replay"] if config_type == "conformance" else ["prefix_alignment"]

        subset = working[working["configuration_type"].isin(allowed_types)].copy()
        if subset.empty or subset["model_name"].nunique() < 2:
            continue

        pivot = subset.pivot_table(index="model_name", columns="configuration",
                                   values=metric_name, aggfunc="mean")

        safe_to_csv(pivot, out_dir / f"{metric_name}__model_by_configuration_means.csv")

        if pivot.shape[0] >= 2 and pivot.shape[1] >= 2:
            pivot = pivot[sorted(pivot.columns, key=configuration_sort_key)]

            fig, ax = plt.subplots(figsize=(max(7, 0.8 * pivot.shape[1]), max(4, 0.55 * pivot.shape[0])))
            sns.heatmap(
                pivot, annot=(pivot.size <= 150), fmt=".2f", cmap="viridis",
                cbar_kws={"label": metric_name},
                xticklabels=[plot_label(c) for c in pivot.columns],
                ax=ax,
            )
            ax.set_title(f"{metric_name} — mean by model and configuration")
            plt.setp(ax.get_xticklabels(), rotation=60, ha="right", fontsize=8)
            plt.setp(ax.get_yticklabels(), fontsize=8)
            highlight_heatmap(ax, list(pivot.index), list(pivot.columns))
            fig.tight_layout()
            save_figure_object(fig, out_dir / f"{metric_name}__model_by_configuration_heatmap")

        # Per model
        per_model_rows = []
        for model_name, model_group in subset.groupby("model_name"):
            omnibus, pairwise_df = kruskal_and_pairwise(model_group, metric_name, group_col="configuration")
            if omnibus is None:
                continue

            per_model_rows.append({
                "model_name": model_name,
                "n_configurations": model_group["configuration"].nunique(),
                "n_files": len(model_group),
                "kruskal_H": omnibus.statistic,
                "kruskal_p": omnibus.pvalue,
                "significant_pairs_holm_0.05":
                    int(pairwise_df["significant_holm_0.05"].sum()) if not pairwise_df.empty else 0,
                "total_pairs": len(pairwise_df),
            })

            if not pairwise_df.empty:
                safe_to_csv(
                    pairwise_df,
                    out_dir / f"{metric_name}__by_model__{model_name}__configuration_pairwise.csv",
                    index=False,
                )

        if per_model_rows:
            safe_to_csv(
                pd.DataFrame(per_model_rows).sort_values("kruskal_p"),
                out_dir / f"{metric_name}__per_model_configuration_kruskal.csv",
                index=False,
            )

        # Per configuration
        per_config_rows = []
        for configuration_name, config_group in subset.groupby("configuration"):
            omnibus, pairwise_df = kruskal_and_pairwise(config_group, metric_name, group_col="model_name")
            if omnibus is None:
                continue

            per_config_rows.append({
                "configuration": configuration_name,
                "n_models": config_group["model_name"].nunique(),
                "n_files": len(config_group),
                "kruskal_H": omnibus.statistic,
                "kruskal_p": omnibus.pvalue,
                "significant_pairs_holm_0.05":
                    int(pairwise_df["significant_holm_0.05"].sum()) if not pairwise_df.empty else 0,
                "total_pairs": len(pairwise_df),
            })

            if not pairwise_df.empty:
                safe_to_csv(
                    pairwise_df,
                    out_dir / f"{metric_name}__by_configuration__{configuration_name}__model_pairwise.csv",
                    index=False,
                )

        if per_config_rows:
            safe_to_csv(
                pd.DataFrame(per_config_rows).sort_values("kruskal_p"),
                out_dir / f"{metric_name}__per_configuration_model_kruskal.csv",
                index=False,
            )


# ============================================================================ #
# MAIN
# ============================================================================ #

def main() -> None:
    global SKIP_SCATTER

    parser = argparse.ArgumentParser(
        description="Event-level statistical comparison of conformance-checking configurations."
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--skip-scatter", action="store_true",
                        help="Skip all scatter plots (heatmaps/tables/distributions only; much faster).")
    args = parser.parse_args()

    SKIP_SCATTER = args.skip_scatter
    results_dir = args.results_dir
    output_dir = results_dir / "analysis_20"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover configurations (recursive)
    sources = discover_configurations(results_dir)
    configurations = ordered_configurations(list(sources.keys()))

    print(f"Discovered {len(configurations)} configuration(s) under {results_dir}:")
    for c in configurations:
        print(f"  {c} -> {plot_label(c)}   [{sources[c].events_path}]")

    # Common log files
    print("\nLog files per configuration:")
    sampled_files = sample_log_files(sources, N_FILES_FOR_EVENT_ANALYSIS)
    print(f"\nUsing {len(sampled_files)} log file(s) common to all configurations.")
    wanted_files = set(sampled_files)

    # Load events
    events_by_config: Dict[str, pd.DataFrame] = {}
    for c in configurations:
        df = load_events_for_files(sources[c], wanted_files, EVENT_METRIC_COLUMNS + ["total_time_ms"])
        if not df.empty:
            events_by_config[c] = df
            print(f"  {c}: {len(df):,} events loaded")

    # Event-level analyses
    cross_algorithm_event_correlation(events_by_config, output_dir)   # main Spearman heatmap
    remaining_variant_distributions(events_by_config, output_dir)     # raw fitness distributions
    per_model_event_correlation(events_by_config, output_dir)
    within_configuration_event_correlation(events_by_config, output_dir)

    # File-level data (only configurations that have a summaries/file CSV)
    parts = [pd.read_csv(sources[c].summary_path)
             for c in configurations if sources[c].summary_path is not None]
    file_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    if not file_df.empty:
        model_configuration_differences(file_df, output_dir)
        performance_comparison(file_df, events_by_config, output_dir)
    else:
        print("\nNo summaries/file CSVs found; skipping file-level analyses.")

    print("\nDone.")
    print(f"All tables/figures (.png/.svg/.pdf) written under:\n{output_dir}")


if __name__ == "__main__":
    main()