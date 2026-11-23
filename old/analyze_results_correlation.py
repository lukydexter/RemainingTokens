from __future__ import annotations

import argparse
import hashlib
import itertools
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

sns.set_theme(style="whitegrid")

PLOT_LABELS: Dict[str, str] = {
    "TokenReplay_remaining_MFS_silent_IGNORE": "TBF-MFS-IGN",
    "TokenReplay_remaining_BRD_silent_LTIt": "TBF-BRD-LTIt",
    "TokenReplay_remaining_BRD_silent_IGNORE": "TBF-BRD-IGN",
    "TokenReplay_remaining_MFS_silent_LTIt": "TBF-MFS-LTIt",
    "TokenReplay_remaining_IGNORE_silent_IGNORE": "TBF-IGN-IGN",
    "TokenReplay_remaining_IGNORE_silent_LTIt": "TBF-IGN-LTIt",
    "BehaviorConformance": "Behavior",
    "PrefixAlignment_revert_5": "Prefix_5",
    "PrefixAlignment_revert_10": "Prefix_10",
    "PrefixAlignment_revert_20": "Prefix_20",
}

# ---------------------------------------------------------------------------- #
# CONFIGURATION
# ---------------------------------------------------------------------------- #

N_FILES_FOR_EVENT_ANALYSIS = None
RANDOM_SEED = 42
FIGURE_FORMATS = ["png", "svg", "pdf"]
DEFAULT_RESULTS_DIR = Path("Tests_online/full_TR_conformance_results_speed")

EVENT_METRIC_COLUMNS = ["conformance_raw", "completeness_raw", "confidence_raw", "cost_raw"]
PERFORMANCE_METRICS = ["total_time_ms", "average_processing_time_per_event_ms", "total_space_used_mb"]

CONFIG_TYPE_PREFIXES: Dict[str, Tuple[str, ...]] = {
    "token_replay": ("TokenReplay",),
    "behavior": ("BehaviorConformance", "Behavior"),
    "prefix_alignment": ("PrefixAlignment",),
}

# revert_20, then revert_10, then revert_5 -- not alphabetical
PREFIX_ALIGNMENT_ORDER: Tuple[str, ...] = (
    "PrefixAlignment_revert_20", "PrefixAlignment_revert_10", "PrefixAlignment_revert_5",
)

# token replay -> conformance_raw, behavior -> conformance_raw, prefix alignment -> cost_raw
# (PrefixAlignment conformance_raw is intentionally NOT used in the cross-algorithm comparison)
ALGORITHM_METRIC: Dict[str, str] = {
    "behavior": "conformance_raw", "token_replay": "conformance_raw", "prefix_alignment": "cost_raw",
}

PERFECT_CONFORMANCE_THRESHOLD = 0.999
PERFECT_COST_THRESHOLD = 1e-9

SCATTER_MIN_SIZE = 4.0
SCATTER_MAX_SIZE = 120.0
SCATTER_SIZE_LEGEND_FULL = [500_000, 1_000_000, 1_500_000, 2_000_000]
SCATTER_SIZE_LEGEND_FILTERED = [10_000, 20_000, 30_000]
MAX_SCATTER_BINS = 30

NOTABLE_PAIRS_PER_METRIC_PER_MODEL = 2
MAX_SAFE_PATH_LENGTH = 200

CORRELATION_MAGNITUDE_BINS = [
    (0.1, "negligible"), (0.3, "weak"), (0.5, "moderate"),
    (0.7, "strong"), (0.9, "very strong"), (1.01, "near-perfect"),
]

# Number of models per combined heatmap-grid image, for the multi-model summary figures.
MODEL_HEATMAP_GRID_SIZES: Tuple[int, ...] = (6, 4)


# ============================================================================ #
# LABEL HELPERS
# ============================================================================ #

def plot_label(configuration_name: str) -> str:
    return PLOT_LABELS.get(configuration_name, configuration_name)


def metric_label(metric: str) -> str:
    if metric == "conformance_raw":
        return "Conformance"
    if metric == "cost_raw":
        return "Alignment cost"
    return metric


def comparison_metric_for_configuration(configuration_name: str) -> Optional[str]:
    return ALGORITHM_METRIC.get(classify_configuration(configuration_name))


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
    return parent / f"{stem[:budget].rstrip('_')}__{digest}{suffix}"


def safe_to_csv(obj, path: Path, **kwargs) -> None:
    path = windows_safe_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    obj.to_csv(path, **kwargs)


def save_figure(path_without_ext: Path) -> None:
    path_without_ext = windows_safe_path(path_without_ext, max_total_len=MAX_SAFE_PATH_LENGTH - 5)
    path_without_ext.parent.mkdir(parents=True, exist_ok=True)
    for ext in FIGURE_FORMATS:
        plt.savefig(f"{path_without_ext}.{ext}", dpi=150, bbox_inches="tight")
    plt.close()


def save_figure_object(fig, path_without_ext: Path) -> None:
    path_without_ext = windows_safe_path(path_without_ext, max_total_len=MAX_SAFE_PATH_LENGTH - 5)
    path_without_ext.parent.mkdir(parents=True, exist_ok=True)
    for ext in FIGURE_FORMATS:
        fig.savefig(f"{path_without_ext}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def chunk_list(items: List, size: int) -> List[List]:
    """Split a list into consecutive chunks of at most `size` items."""
    return [items[i:i + size] for i in range(0, len(items), size)]


def discover_configurations(results_dir: Path) -> List[str]:
    events_dir = results_dir / "events"
    if not events_dir.exists():
        raise RuntimeError(f"No events/ directory found under {results_dir}")
    return sorted(p.stem for p in events_dir.glob("*.csv"))


def classify_configuration(configuration_name: str) -> str:
    for config_type, prefixes in CONFIG_TYPE_PREFIXES.items():
        if configuration_name.startswith(prefixes):
            return config_type
    return "unknown"


def is_perfect(values: pd.Series, metric: str) -> pd.Series:
    if metric == "cost_raw":
        return values <= PERFECT_COST_THRESHOLD
    return values >= PERFECT_CONFORMANCE_THRESHOLD


def sample_log_files(results_dir: Path, configurations: List[str], n_files: Optional[int]) -> List[str]:
    rng = random.Random(RANDOM_SEED)
    common_files: Optional[set] = None

    for configuration_name in configurations:
        path = results_dir / "summaries" / "file" / f"{configuration_name}.csv"
        if not path.exists():
            continue
        files_here = set(pd.read_csv(path, usecols=["log_file"])["log_file"])
        common_files = files_here if common_files is None else common_files & files_here

    if not common_files:
        raise RuntimeError("Could not find any log files common to the discovered configurations.")

    common_files = sorted(common_files)
    if n_files is None or len(common_files) <= n_files:
        return common_files
    return sorted(rng.sample(common_files, n_files))


def load_events_for_files(results_dir: Path, configuration_name: str, wanted_files: set, columns: List[str]) -> pd.DataFrame:
    path = results_dir / "events" / f"{configuration_name}.csv"
    if not path.exists():
        return pd.DataFrame()

    all_wanted_cols = list(dict.fromkeys(["log_file", "event_index", "case_id", "model_name"] + columns))
    chunks = []
    for chunk in pd.read_csv(path, usecols=lambda c: c in all_wanted_cols, chunksize=200_000):
        filtered = chunk[chunk["log_file"].isin(wanted_files)]
        if not filtered.empty:
            chunks.append(filtered)

    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


# ============================================================================ #
# CONFIGURATION ORDERING
# ============================================================================ #

def configuration_sort_key(configuration_name: str):
    """Required order: 1) Behavior, 2) TokenReplay, 3) PrefixAlignment. Alphabetical within family."""
    config_type = classify_configuration(configuration_name)
    family_order = {"behavior": 0, "token_replay": 1, "prefix_alignment": 2, "unknown": 99}

    if config_type == "prefix_alignment":
        if configuration_name in PREFIX_ALIGNMENT_ORDER:
            secondary_key = (0, PREFIX_ALIGNMENT_ORDER.index(configuration_name))
        else:
            secondary_key = (1, configuration_name)
    else:
        secondary_key = (0, configuration_name)

    return (family_order.get(config_type, 99), secondary_key)


def ordered_configurations(configurations: List[str]) -> List[str]:
    return sorted(configurations, key=configuration_sort_key)


# ============================================================================ #
# STATISTICS
# ============================================================================ #

def spearman_pair(x: pd.Series, y: pd.Series) -> Tuple[float, float, int]:
    paired = pd.concat([x, y], axis=1).dropna()
    if len(paired) < 3 or paired.iloc[:, 0].nunique() < 2 or paired.iloc[:, 1].nunique() < 2:
        return np.nan, np.nan, len(paired)
    r, p = stats.spearmanr(paired.iloc[:, 0], paired.iloc[:, 1])
    return float(r), float(p), len(paired)


def spearman_matrix_with_pvalues(df: pd.DataFrame, columns: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    r_matrix = pd.DataFrame(np.nan, index=columns, columns=columns)
    p_matrix = pd.DataFrame(np.nan, index=columns, columns=columns)
    n_matrix = pd.DataFrame(np.nan, index=columns, columns=columns)

    for col_a, col_b in itertools.combinations(columns, 2):
        r, p, n = spearman_pair(df[col_a], df[col_b])
        r_matrix.loc[col_a, col_b] = r_matrix.loc[col_b, col_a] = r
        p_matrix.loc[col_a, col_b] = p_matrix.loc[col_b, col_a] = p
        n_matrix.loc[col_a, col_b] = n_matrix.loc[col_b, col_a] = n

    for col in columns:
        r_matrix.loc[col, col] = 1.0
        p_matrix.loc[col, col] = 0.0
        n_matrix.loc[col, col] = df[col].notna().sum()

    return r_matrix, p_matrix, n_matrix


def holm_bonferroni(p_values: List[float]) -> List[float]:
    n = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        running_max = max(running_max, (n - rank) * p_values[idx])
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


# ============================================================================ #
# P-VALUE FORMATTING
# ============================================================================ #

def format_p_value(p: float) -> str:
    """scipy can legitimately return p=0 below floating-point precision; show that as <1e-300."""
    if pd.isna(p):
        return ""
    if p == 0 or p < 1e-300:
        return "0.001"
    if p < 0.001:
        return "0.001"
    return f"{p:.3f}"


def format_rho_p(r: float, p: float) -> str:
    if pd.isna(r):
        return ""
    return f"rho={r:.2f}\np<{format_p_value(p)}"


def _magnitude_label(r: float) -> str:
    abs_r = abs(r)
    for threshold, label in CORRELATION_MAGNITUDE_BINS:
        if abs_r < threshold:
            return label
    return CORRELATION_MAGNITUDE_BINS[-1][1]


def _sign_word(r: float) -> str:
    return "positive" if r > 0 else "negative" if r < 0 else "no"


# ============================================================================ #
# BUILD CROSS-ALGORITHM WIDE TABLE
# ============================================================================ #

def build_cross_algorithm_wide_table(events_by_config: Dict[str, pd.DataFrame]) -> Tuple[Optional[pd.DataFrame], List[str], Dict[str, str]]:
    """
    Build one wide table containing ALL configurations, each contributing its own metric:
    Behavior/TokenReplay -> conformance_raw, PrefixAlignment -> cost_raw.
    """
    selected = {}
    for configuration_name, df in events_by_config.items():
        metric = comparison_metric_for_configuration(configuration_name)
        if metric is None or metric not in df.columns or df[metric].isna().all():
            continue
        selected[configuration_name] = metric

    if len(selected) < 2:
        return None, [], {}

    wide = None
    for configuration_name, metric in selected.items():
        series = events_by_config[configuration_name][["log_file", "event_index", metric]].rename(
            columns={metric: configuration_name}
        )
        wide = series if wide is None else wide.merge(series, on=["log_file", "event_index"], how="outer")

    configs = ordered_configurations(list(selected.keys()))
    return wide, configs, selected


# ============================================================================ #
# HEATMAP
# ============================================================================ #

def _draw_heatmap_on_ax(ax, r_matrix: pd.DataFrame, p_matrix: pd.DataFrame, configs: List[str], title: str, annot: bool = True) -> None:
    """Draw one Spearman-correlation heatmap into an existing Axes (used standalone and in grids)."""
    labels = [plot_label(c) for c in configs]

    sns.heatmap(
        r_matrix.astype(float), annot=False, cmap="RdBu_r", vmin=-1, vmax=1, square=True,
        cbar_kws={"label": "Spearman rho"}, xticklabels=labels, yticklabels=labels, ax=ax,
    )

    if annot:
        for row_idx, i in enumerate(configs):
            for col_idx, j in enumerate(configs):
                r = r_matrix.loc[i, j]
                if pd.isna(r):
                    continue
                text_color = "white" if abs(r_matrix.values[row_idx, col_idx]) > 0.55 else "black"
                if i == j:
                    ax.text(col_idx + 0.5, row_idx + 0.5, "1.00", ha="center", va="center", fontsize=7, color=text_color)
                    continue
                p = p_matrix.loc[i, j]
                ax.text(col_idx + 0.5, row_idx + 0.42, f"{r:.2f}", ha="center", va="center", fontsize=10, color=text_color)
                ax.text(col_idx + 0.5, row_idx + 0.68, f"p<{format_p_value(p)}", ha="center", va="center", fontsize=5.5, color=text_color)

    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
    ax.set_yticklabels(labels, rotation=0, fontsize=7)
    ax.set_title(title, fontsize=10)


def plot_correlation_heatmap(r_matrix: pd.DataFrame, p_matrix: pd.DataFrame, metric_map: Dict[str, str], title: str, path: Path) -> None:
    configs = ordered_configurations(list(r_matrix.columns))
    r_matrix, p_matrix = r_matrix.loc[configs, configs], p_matrix.loc[configs, configs]

    n = len(configs)
    size = max(7, min(15, 0.65 * n + 4))
    fig, ax = plt.subplots(figsize=(size, size))

    _draw_heatmap_on_ax(ax, r_matrix, p_matrix, configs, title, annot=n <= 20)
    fig.tight_layout()
    save_figure_object(fig, path)


def plot_model_heatmap_grid(
    model_data: Dict[str, Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]],
    model_names: List[str],
    ncols: int,
    nrows: int,
    suptitle: str,
    path: Path,
) -> None:
    """Render several models' correlation heatmaps as subplots in a single combined figure."""
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 6.2 * nrows), squeeze=False)
    axes = axes.flatten()

    for idx, model_name in enumerate(model_names):
        ax = axes[idx]
        if model_name not in model_data:
            ax.axis("off")
            continue
        r_matrix, p_matrix, _metric_map = model_data[model_name]
        configs = ordered_configurations(list(r_matrix.columns))
        r_sub, p_sub = r_matrix.loc[configs, configs], p_matrix.loc[configs, configs]
        _draw_heatmap_on_ax(ax, r_sub, p_sub, configs, str(model_name), annot=len(configs) <= 12)

    for idx in range(len(model_names), len(axes)):
        axes[idx].axis("off")

    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
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


def _size_for_count(counts: np.ndarray, legend_counts: List[int], min_size: float, max_size: float) -> np.ndarray:
    max_legend = max(legend_counts)
    counts = np.asarray(counts, dtype=float)
    sizes = max_size * np.sqrt(np.clip(counts, 0, max_legend) / max_legend)
    return np.clip(sizes, min_size, max_size)


def _size_legend_handles(legend_counts: List[int], min_size: float, max_size: float) -> Tuple[List, List[str]]:
    handles = [
        plt.scatter([], [], s=_size_for_count(np.array([count]), legend_counts, min_size, max_size)[0], alpha=0.55, edgecolors="none")
        for count in legend_counts
    ]
    labels = [f"{count:,} events" for count in legend_counts]
    return handles, labels


def _fit_weighted_trend_line(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, str]]:
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
    df: pd.DataFrame, x: str, y: str, title: str, path: Path, legend_counts: List[int],
    rho: Optional[float] = None, p_value: Optional[float] = None, max_bins: int = MAX_SCATTER_BINS,
    min_size: float = SCATTER_MIN_SIZE, max_size: float = SCATTER_MAX_SIZE,
    xlabel: Optional[str] = None, ylabel: Optional[str] = None,
) -> None:
    valid = df[[x, y]].dropna()
    if len(valid) < 5:
        return

    x_edges = _bin_edges_if_needed(valid[x].to_numpy(), max_bins)
    y_edges = _bin_edges_if_needed(valid[y].to_numpy(), max_bins)

    if x_edges is None and y_edges is None:
        grouped = valid.groupby([x, y]).size().reset_index(name="event_count")
        plot_x, plot_y, merge_note = grouped[x], grouped[y], ""
    else:
        x_binned = pd.cut(valid[x], bins=x_edges, include_lowest=True) if x_edges is not None else valid[x]
        y_binned = pd.cut(valid[y], bins=y_edges, include_lowest=True) if y_edges is not None else valid[y]
        grouped = valid.assign(_xb=x_binned, _yb=y_binned).groupby(["_xb", "_yb"], observed=True).size().reset_index(name="event_count")
        grouped = grouped[grouped["event_count"] > 0]
        plot_x = grouped["_xb"].apply(lambda v: v.mid if hasattr(v, "mid") else v).astype(float)
        plot_y = grouped["_yb"].apply(lambda v: v.mid if hasattr(v, "mid") else v).astype(float)
        merge_note = f"; values merged into up to {max_bins}\u00d7{max_bins} bins"

    counts = grouped["event_count"].to_numpy()
    sizes = _size_for_count(counts, legend_counts, min_size, max_size)

    fig, ax = plt.subplots(figsize=(7.0, 7.0))
    ax.scatter(plot_x, plot_y, s=sizes, alpha=0.55, edgecolors="none")

    trend_line = _fit_weighted_trend_line(plot_x.to_numpy(), plot_y.to_numpy(), counts)
    handles, labels = _size_legend_handles(legend_counts, min_size, max_size)

    if trend_line is not None:
        x_line, y_line, direction = trend_line
        trend_handle, = ax.plot(x_line, y_line, color="red", linestyle="--", linewidth=1.8)
        handles.append(trend_handle)
        labels.append(f"trend line ({direction})")

    ax.set_xlabel(xlabel if xlabel else plot_label(x), fontsize=9)
    ax.set_ylabel(ylabel if ylabel else plot_label(y), fontsize=9)
    ax.tick_params(axis="both", labelsize=8)

    if rho is not None and p_value is not None:
        title = f"{title}\nSpearman rho={rho:.3f}, p<{format_p_value(p_value)}"

    ax.set_title(f"{title}\n(n={len(valid):,} events{merge_note})", fontsize=10)
    ax.set_box_aspect(1)
    ax.legend(handles, labels, title="Dot size = event count", loc="upper left", bbox_to_anchor=(1.02, 1.0),
               borderaxespad=0, frameon=True, labelspacing=1.4, fontsize=7, title_fontsize=7.5)

    fig.tight_layout()
    save_figure_object(fig, path)


# ============================================================================ #
# REFERENCE-CONFIG VS TOKEN REPLAY SUBPLOTS
# ============================================================================ #

def _reorder_middle_pair_for_3x2_grid(configs: List[str], ncols: int, nrows: int) -> List[str]:
    """
    For a 3-column x 2-row subplot grid, swap the two middle subplots
    (index 1 = top-middle, index 4 = bottom-middle) so their order is inverted.
    No-op for any other grid shape.
    """
    if ncols == 3 and nrows == 2 and len(configs) == 6:
        configs = list(configs)
        configs[1], configs[4] = configs[4], configs[1]
    return configs


def plot_reference_vs_token_replay_subplots(
    wide: pd.DataFrame, reference_config: str, token_configs: List[str],
    r_matrix: pd.DataFrame, p_matrix: pd.DataFrame, output_path: Path, suptitle: Optional[str] = None,
) -> None:
    """
    One subplot per TokenReplay configuration, each comparing reference_config's metric
    against that TokenReplay configuration's conformance. Dot size encodes event count;
    trend line is red. `reference_config` may be Behavior or a PrefixAlignment config
    (cost_raw instead of conformance_raw) -- plotting logic is identical either way.
    """
    if not token_configs:
        return

    n = len(token_configs)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    token_configs = _reorder_middle_pair_for_3x2_grid(token_configs, ncols, nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.5 * nrows), squeeze=False)
    axes = axes.flatten()
    reference_label = plot_label(reference_config)

    for idx, token_config in enumerate(token_configs):
        ax = axes[idx]
        sub = wide[[reference_config, token_config]].dropna()
        if len(sub) < 5:
            ax.axis("off")
            continue

        x, y = sub[reference_config], sub[token_config]
        x_edges = _bin_edges_if_needed(x.to_numpy(), MAX_SCATTER_BINS)
        y_edges = _bin_edges_if_needed(y.to_numpy(), MAX_SCATTER_BINS)

        if x_edges is None and y_edges is None:
            grouped = sub.groupby([reference_config, token_config]).size().reset_index(name="event_count")
            plot_x = grouped[reference_config].to_numpy()
            plot_y = grouped[token_config].to_numpy()
        else:
            xb = pd.cut(x, bins=x_edges, include_lowest=True) if x_edges is not None else x
            yb = pd.cut(y, bins=y_edges, include_lowest=True) if y_edges is not None else y
            grouped = sub.assign(_xb=xb, _yb=yb).groupby(["_xb", "_yb"], observed=True).size().reset_index(name="event_count")
            plot_x = grouped["_xb"].apply(lambda v: v.mid if hasattr(v, "mid") else v).astype(float).to_numpy()
            plot_y = grouped["_yb"].apply(lambda v: v.mid if hasattr(v, "mid") else v).astype(float).to_numpy()

        counts = grouped["event_count"].to_numpy()
        sizes = _size_for_count(counts, SCATTER_SIZE_LEGEND_FULL, SCATTER_MIN_SIZE, SCATTER_MAX_SIZE)
        ax.scatter(plot_x, plot_y, s=sizes, alpha=0.55, edgecolors="none")

        trend_line = _fit_weighted_trend_line(plot_x, plot_y, counts)
        if trend_line is not None:
            x_line, y_line, _ = trend_line
            ax.plot(x_line, y_line, color="red", linestyle="--", linewidth=1.7)

        r = r_matrix.loc[reference_config, token_config]
        p = p_matrix.loc[reference_config, token_config]

        ax.set_title(f"{plot_label(token_config)}\nrho={r:.3f}, p<{format_p_value(p)}", fontsize=9)
        ax.set_xlabel(reference_label, fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_ylabel(plot_label(token_config), fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_box_aspect(1)

    for idx in range(len(token_configs), len(axes)):
        axes[idx].axis("off")

    fig.suptitle(suptitle if suptitle is not None else f"{reference_label} vs TokenReplay configurations", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure_object(fig, output_path)


# ============================================================================ #
# PERFECT AGREEMENT
# ============================================================================ #

def perfect_agreement_table(wide: pd.DataFrame, configs_with_metric: List[str], metric_map: Dict[str, str]) -> pd.DataFrame:
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
            "config_a": config_a, "config_b": config_b, "metric_a": metric_a, "metric_b": metric_b,
            "n_joint_events": n, "both_perfect": both, "pct_both_perfect": 100.0 * both / n,
            "only_config_a_perfect": only_a, "only_config_b_perfect": only_b, "neither_perfect": neither,
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("pct_both_perfect", ascending=False)


def plot_perfect_confusion_matrix(
    config_a: str, config_b: str, both: int, only_a: int, only_b: int, neither: int,
    metric_a: str, metric_b: str, path: Path,
) -> None:
    matrix = np.array([[both, only_a], [only_b, neither]])
    total = matrix.sum()
    pct_both = 100.0 * both / total if total else 0.0

    plt.figure(figsize=(5.2, 4.4))
    sns.heatmap(
        matrix, annot=True, fmt="d", cmap="Greens",
        xticklabels=[f"{plot_label(config_b)}: perfect", f"{plot_label(config_b)}: not perfect"],
        yticklabels=[f"{plot_label(config_a)}: perfect", f"{plot_label(config_a)}: not perfect"],
        cbar_kws={"label": "event count"},
    )
    plt.xticks(rotation=20, ha="right", fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.title(
        f"Perfect agreement\n{plot_label(config_a)} ({metric_label(metric_a)}) vs "
        f"{plot_label(config_b)} ({metric_label(metric_b)})\n"
        f"{pct_both:.1f}% of {total:,} joint events perfect on both", fontsize=9,
    )
    plt.tight_layout()
    save_figure(path)


# ============================================================================ #
# CROSS-ALGORITHM CORRELATION
# ============================================================================ #

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

    r_matrix, p_matrix, n_matrix = spearman_matrix_with_pvalues(wide, configs)
    safe_to_csv(r_matrix, out_dir / "all_configurations__spearman_r.csv")
    safe_to_csv(p_matrix, out_dir / "all_configurations__spearman_p.csv")
    safe_to_csv(n_matrix, out_dir / "all_configurations__spearman_n.csv")

    plot_correlation_heatmap(r_matrix, p_matrix, metric_map, "All configurations \u2014 Spearman correlation",
                              out_dir / "all_configurations__spearman_heatmap")

    behavior_configs = [c for c in configs if classify_configuration(c) == "behavior"]
    token_configs = [c for c in configs if classify_configuration(c) == "token_replay"]

    if behavior_configs and token_configs:
        plot_reference_vs_token_replay_subplots(
            wide, behavior_configs[0], token_configs, r_matrix, p_matrix,
            out_dir / "behavior_vs_token_replay__subplots", suptitle="Behavior vs TokenReplay configurations",
        )

    prefix20_configs = [c for c in configs if c == "PrefixAlignment_revert_20"]
    if prefix20_configs and token_configs:
        plot_reference_vs_token_replay_subplots(
            wide, prefix20_configs[0], token_configs, r_matrix, p_matrix,
            out_dir / "prefix20_vs_token_replay__subplots", suptitle="Prefix_20 (cost) vs TokenReplay configurations",
        )

    # All pairwise scatter plots. PrefixAlignment vs Behavior/TokenReplay is allowed
    # (it contributes cost_raw); PrefixAlignment vs PrefixAlignment is skipped.
    all_pairs = list(itertools.combinations(configs, 2))
    print(f"  Scatter plots: evaluating {len(all_pairs)} configuration pairs")

    for config_a, config_b in all_pairs:
        if classify_configuration(config_a) == "prefix_alignment" and classify_configuration(config_b) == "prefix_alignment":
            continue

        r, p = r_matrix.loc[config_a, config_b], p_matrix.loc[config_a, config_b]
        if pd.isna(r):
            continue

        metric_a, metric_b = metric_map[config_a], metric_map[config_b]
        xlabel = f"{plot_label(config_a)} ({metric_label(metric_a)})"
        ylabel = f"{plot_label(config_b)} ({metric_label(metric_b)})"

        plot_scatter_sized_by_count(
            wide, config_a, config_b, f"{plot_label(config_a)} vs {plot_label(config_b)}",
            out_dir / f"scatter_all__{config_a}__vs__{config_b}",
            legend_counts=SCATTER_SIZE_LEGEND_FULL, rho=r, p_value=p, xlabel=xlabel, ylabel=ylabel,
        )

    # Perfect agreement tables / confusion matrices / non-perfect scatters (PrefixAlignment-vs-PrefixAlignment excluded)
    agreement_table = perfect_agreement_table(wide, configs, metric_map)
    if agreement_table.empty:
        return

    agreement_table = agreement_table[~(
        (agreement_table["config_a"].map(classify_configuration) == "prefix_alignment")
        & (agreement_table["config_b"].map(classify_configuration) == "prefix_alignment")
    )]
    safe_to_csv(agreement_table, out_dir / "perfect_agreement_summary.csv", index=False)

    confusion_dir = out_dir / "perfect_confusion_matrices"
    non_perfect_dir = out_dir / "non_perfect_scatter"

    for _, row in agreement_table.iterrows():
        config_a, config_b = row["config_a"], row["config_b"]

        plot_perfect_confusion_matrix(
            config_a, config_b, int(row["both_perfect"]), int(row["only_config_a_perfect"]),
            int(row["only_config_b_perfect"]), int(row["neither_perfect"]), row["metric_a"], row["metric_b"],
            confusion_dir / f"{config_a}__vs__{config_b}",
        )

        sub = wide[[config_a, config_b]].dropna()
        metric_a, metric_b = row["metric_a"], row["metric_b"]
        a_perfect = is_perfect(sub[config_a], metric_a)
        b_perfect = is_perfect(sub[config_b], metric_b)
        disagreement_subset = sub[~(a_perfect & b_perfect)]

        r, p = r_matrix.loc[config_a, config_b], p_matrix.loc[config_a, config_b]
        plot_scatter_sized_by_count(
            disagreement_subset, config_a, config_b,
            f"{plot_label(config_a)} vs {plot_label(config_b)} \u2014 at least one event is not perfect",
            non_perfect_dir / f"{config_a}__vs__{config_b}",
            legend_counts=SCATTER_SIZE_LEGEND_FILTERED, rho=r, p_value=p,
            xlabel=f"{plot_label(config_a)} ({metric_label(metric_a)})",
            ylabel=f"{plot_label(config_b)} ({metric_label(metric_b)})",
        )


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

    # Collected while looping, so we can also render combined multi-model heatmap grids afterwards.
    model_heatmap_data: Dict[str, Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]] = {}

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

        r_matrix, p_matrix, n_matrix = spearman_matrix_with_pvalues(wide, configs)
        safe_to_csv(r_matrix, model_out_dir / "spearman_r.csv")
        safe_to_csv(p_matrix, model_out_dir / "spearman_p.csv")
        safe_to_csv(n_matrix, model_out_dir / "spearman_n.csv")

        model_heatmap_data[str(model_name)] = (r_matrix, p_matrix, metric_map)

        plot_correlation_heatmap(r_matrix, p_matrix, metric_map,
                                  f"{model_name} \u2014 all configurations \u2014 Spearman correlation",
                                  model_out_dir / "spearman_heatmap")

        behavior_configs = [c for c in configs if classify_configuration(c) == "behavior"]
        token_configs = [c for c in configs if classify_configuration(c) == "token_replay"]

        if behavior_configs and token_configs:
            plot_reference_vs_token_replay_subplots(
                wide, behavior_configs[0], token_configs, r_matrix, p_matrix,
                model_out_dir / "behavior_vs_token_replay__subplots",
                suptitle=f"{model_name}: Behavior vs TokenReplay configurations",
            )

        prefix20_configs = [c for c in configs if c == "PrefixAlignment_revert_20"]
        if prefix20_configs and token_configs:
            plot_reference_vs_token_replay_subplots(
                wide, prefix20_configs[0], token_configs, r_matrix, p_matrix,
                model_out_dir / "prefix20_vs_token_replay__subplots",
                suptitle=f"{model_name}: Prefix_20 (cost) vs TokenReplay configurations",
            )

        # Per-model scatter plots (includes PrefixAlignment vs Behavior/TokenReplay; excludes PA-vs-PA)
        for config_a, config_b in itertools.combinations(configs, 2):
            if classify_configuration(config_a) == "prefix_alignment" and classify_configuration(config_b) == "prefix_alignment":
                continue

            r, p = r_matrix.loc[config_a, config_b], p_matrix.loc[config_a, config_b]
            if pd.isna(r):
                continue

            metric_a, metric_b = metric_map[config_a], metric_map[config_b]
            plot_scatter_sized_by_count(
                wide, config_a, config_b, f"{model_name}\n{plot_label(config_a)} vs {plot_label(config_b)}",
                model_out_dir / f"scatter_all__{config_a}__vs__{config_b}",
                legend_counts=SCATTER_SIZE_LEGEND_FULL, rho=r, p_value=p,
                xlabel=f"{plot_label(config_a)} ({metric_label(metric_a)})",
                ylabel=f"{plot_label(config_b)} ({metric_label(metric_b)})",
            )

    # ------------------------------------------------------------------------ #
    # Combined multi-model heatmap-grid summary images: N per grid, for each
    # configured group size (e.g. groups of 6 -> 2 images, groups of 4 -> 3 images
    # for 12 models). Grid shape is chosen to roughly match the group size.
    # ------------------------------------------------------------------------ #
    model_names_sorted = list(model_heatmap_data.keys())
    if not model_names_sorted:
        return

    grid_shape_by_size = {6: (3, 2), 4: (2, 2)}

    for group_size in MODEL_HEATMAP_GRID_SIZES:
        ncols, nrows = grid_shape_by_size.get(group_size, (group_size, 1))
        groups = chunk_list(model_names_sorted, group_size)

        for group_idx, group in enumerate(groups, start=1):
            plot_model_heatmap_grid(
                model_heatmap_data, group, ncols=ncols, nrows=nrows,
                suptitle=f"Model heatmaps \u2014 group {group_idx} of {len(groups)} ({group_size} per image)",
                path=out_dir / f"model_heatmaps_grid{group_size}__part{group_idx}",
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

        r_matrix, p_matrix, n_matrix = spearman_matrix_with_pvalues(df, columns)
        safe_to_csv(r_matrix, out_dir / f"{configuration_name}__spearman_r.csv")
        safe_to_csv(p_matrix, out_dir / f"{configuration_name}__spearman_p.csv")
        safe_to_csv(n_matrix, out_dir / f"{configuration_name}__spearman_n.csv")

        metric_map = {c: c for c in columns}
        plot_correlation_heatmap(
            r_matrix, p_matrix, metric_map,
            f"{plot_label(configuration_name)} \u2014 within-configuration Spearman correlation",
            out_dir / f"{configuration_name}__spearman_heatmap",
        )

        for col_a, col_b in itertools.combinations(columns, 2):
            r, p = r_matrix.loc[col_a, col_b], p_matrix.loc[col_a, col_b]
            if pd.isna(r):
                continue

            plot_scatter_sized_by_count(
                df, col_a, col_b, f"{plot_label(configuration_name)}: {col_a} vs {col_b}",
                out_dir / f"{configuration_name}__{col_a}__vs__{col_b}",
                legend_counts=SCATTER_SIZE_LEGEND_FULL, rho=r, p_value=p, xlabel=col_a, ylabel=col_b,
            )


# ============================================================================ #
# PERFORMANCE PLOTS
# ============================================================================ #

def plot_performance_summary(file_df: pd.DataFrame, metric: str, title: str, path: Path) -> None:
    """Median + IQR + individual file-level observations (clearer than a log-scale boxplot)."""
    data = file_df[["configuration", metric]].dropna()
    if data.empty:
        return

    order = sorted(data["configuration"].unique(), key=configuration_sort_key)
    positions = np.arange(len(order))
    medians, q1s, q3s = [], [], []

    for config in order:
        values = data.loc[data["configuration"] == config, metric].to_numpy()
        medians.append(np.median(values))
        q1s.append(np.percentile(values, 25))
        q3s.append(np.percentile(values, 75))

    fig, ax = plt.subplots(figsize=(max(8, 0.75 * len(order)), 5.5))
    ax.vlines(positions, q1s, q3s, linewidth=7, alpha=0.65)
    ax.scatter(positions, medians, s=35, zorder=3, label="Median")

    for i, config in enumerate(order):
        values = data.loc[data["configuration"] == config, metric].to_numpy()
        jitter = np.random.default_rng(RANDOM_SEED + i).uniform(-0.08, 0.08, len(values))
        ax.scatter(np.full(len(values), i) + jitter, values, s=7, alpha=0.25, edgecolors="none")

    labels = [plot_label(c) for c in order]
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8)

    fig.tight_layout()
    save_figure_object(fig, path)


def performance_comparison(file_df: pd.DataFrame, events_by_config: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    print("\n=== Performance comparison ===")
    out_dir = output_dir / "performance"
    out_dir.mkdir(parents=True, exist_ok=True)

    for metric in PERFORMANCE_METRICS:
        if metric not in file_df.columns or file_df[metric].isna().all():
            continue

        plot_performance_summary(file_df, metric, f"{metric.replace('_', ' ').title()} by configuration",
                                  out_dir / f"{metric}__summary_plot")

        omnibus, pairwise_df = kruskal_and_pairwise(file_df, metric)
        if omnibus is None:
            continue

        print(f"  {metric}: Kruskal-Wallis H={omnibus.statistic:.2f}, p<{format_p_value(omnibus.pvalue)}")
        safe_to_csv(pairwise_df, out_dir / f"{metric}__pairwise_tests.csv", index=False)

    available_metrics = [m for m in PERFORMANCE_METRICS if m in file_df.columns]
    if available_metrics:
        summary = file_df.groupby("configuration")[available_metrics].agg(["mean", "median", "std"])
        safe_to_csv(summary, out_dir / "performance_summary_by_configuration.csv")

    combined_events = pd.concat(
        [df.assign(configuration=name) for name, df in events_by_config.items() if not df.empty],
        ignore_index=True,
    )

    if not combined_events.empty and "total_time_ms" in combined_events.columns:
        data = combined_events[["configuration", "total_time_ms"]].dropna()
        order = sorted(data["configuration"].unique(), key=configuration_sort_key)

        plt.figure(figsize=(max(8, 0.7 * len(order)), 5))
        sns.violinplot(data=data, x="configuration", y="total_time_ms", order=order, cut=0, inner="quartile")
        plt.xticks(range(len(order)), [plot_label(c) for c in order], rotation=55, ha="right", fontsize=8)
        plt.ylabel("Processing time per event (ms)")
        plt.xlabel("Configuration")
        plt.title("Distribution of event processing time")
        plt.tight_layout()
        save_figure(out_dir / "event_level_total_time_ms__violin")


# ============================================================================ #
# MODEL VS CONFIGURATION DIFFERENCES
# ============================================================================ #

def model_configuration_differences(file_df: pd.DataFrame, output_dir: Path) -> None:
    print("\n=== Model vs configuration differences ===")
    out_dir = output_dir / "model_vs_configuration"
    out_dir.mkdir(parents=True, exist_ok=True)

    working = file_df.copy()
    working["configuration_type"] = working["configuration"].map(classify_configuration)

    # Same metric routing as the event-level cross-algorithm analysis:
    # Behavior/TokenReplay -> mean_conformance, PrefixAlignment -> mean_cost.
    for config_type, metric_name in [("conformance", "mean_conformance"), ("cost", "mean_cost")]:
        if metric_name not in working.columns:
            continue

        allowed_types = ["behavior", "token_replay"] if config_type == "conformance" else ["prefix_alignment"]
        subset = working[working["configuration_type"].isin(allowed_types)].copy()
        if subset.empty or subset["model_name"].nunique() < 2:
            continue

        pivot = subset.pivot_table(index="model_name", columns="configuration", values=metric_name, aggfunc="mean")
        safe_to_csv(pivot, out_dir / f"{metric_name}__model_by_configuration_means.csv")

        if pivot.shape[0] >= 2 and pivot.shape[1] >= 2:
            ordered_columns = sorted(pivot.columns, key=configuration_sort_key)
            pivot = pivot[ordered_columns]

            plt.figure(figsize=(max(7, 0.8 * pivot.shape[1]), max(4, 0.55 * pivot.shape[0])))
            sns.heatmap(pivot, annot=(pivot.size <= 150), fmt=".2f", cmap="viridis", cbar_kws={"label": metric_name},
                        xticklabels=[plot_label(c) for c in pivot.columns])
            plt.title(f"{metric_name} \u2014 mean by model and configuration")
            plt.xticks(rotation=60, ha="right", fontsize=8)
            plt.yticks(fontsize=8)
            plt.tight_layout()
            save_figure(out_dir / f"{metric_name}__model_by_configuration_heatmap")

        # Per model
        per_model_rows = []
        for model_name, model_group in subset.groupby("model_name"):
            omnibus, pairwise_df = kruskal_and_pairwise(model_group, metric_name, group_col="configuration")
            if omnibus is None:
                continue

            per_model_rows.append({
                "model_name": model_name, "n_configurations": model_group["configuration"].nunique(),
                "n_files": len(model_group), "kruskal_H": omnibus.statistic, "kruskal_p": omnibus.pvalue,
                "significant_pairs_holm_0.05": int(pairwise_df["significant_holm_0.05"].sum()) if not pairwise_df.empty else 0,
                "total_pairs": len(pairwise_df),
            })

            if not pairwise_df.empty:
                safe_to_csv(pairwise_df, out_dir / f"{metric_name}__by_model__{model_name}__configuration_pairwise.csv", index=False)

        if per_model_rows:
            safe_to_csv(pd.DataFrame(per_model_rows).sort_values("kruskal_p"),
                        out_dir / f"{metric_name}__per_model_configuration_kruskal.csv", index=False)

        # Per configuration
        per_config_rows = []
        for configuration_name, config_group in subset.groupby("configuration"):
            omnibus, pairwise_df = kruskal_and_pairwise(config_group, metric_name, group_col="model_name")
            if omnibus is None:
                continue

            per_config_rows.append({
                "configuration": configuration_name, "n_models": config_group["model_name"].nunique(),
                "n_files": len(config_group), "kruskal_H": omnibus.statistic, "kruskal_p": omnibus.pvalue,
                "significant_pairs_holm_0.05": int(pairwise_df["significant_holm_0.05"].sum()) if not pairwise_df.empty else 0,
                "total_pairs": len(pairwise_df),
            })

            if not pairwise_df.empty:
                safe_to_csv(pairwise_df, out_dir / f"{metric_name}__by_configuration__{configuration_name}__model_pairwise.csv", index=False)

        if per_config_rows:
            safe_to_csv(pd.DataFrame(per_config_rows).sort_values("kruskal_p"),
                        out_dir / f"{metric_name}__per_configuration_model_kruskal.csv", index=False)


# ============================================================================ #
# MAIN
# ============================================================================ #

def main() -> None:
    parser = argparse.ArgumentParser(description="Event-level statistical comparison of conformance-checking configurations.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()

    results_dir = args.results_dir
    output_dir = results_dir / "analysis_final"
    output_dir.mkdir(parents=True, exist_ok=True)

    configurations = ordered_configurations(discover_configurations(results_dir))
    print(f"Discovered {len(configurations)} configuration(s) under {results_dir}/events/")
    print("\nPlot label mapping:")
    for config in configurations:
        print(f"  {config} -> {plot_label(config)}")

    sampled_files = sample_log_files(results_dir, configurations, N_FILES_FOR_EVENT_ANALYSIS)
    file_desc = "all" if N_FILES_FOR_EVENT_ANALYSIS is None else str(len(sampled_files))
    print(f"\nUsing {file_desc} log file(s) ({len(sampled_files)} total) for event-level analysis.")
    wanted_files = set(sampled_files)

    events_by_config: Dict[str, pd.DataFrame] = {}
    for configuration_name in configurations:
        df = load_events_for_files(results_dir, configuration_name, wanted_files, EVENT_METRIC_COLUMNS + ["total_time_ms"])
        if not df.empty:
            events_by_config[configuration_name] = df
            print(f"  {configuration_name}: {len(df):,} events loaded")

    cross_algorithm_event_correlation(events_by_config, output_dir)
    per_model_event_correlation(events_by_config, output_dir)
    within_configuration_event_correlation(events_by_config, output_dir)

    file_df_parts = []
    for configuration_name in configurations:
        path = results_dir / "summaries" / "file" / f"{configuration_name}.csv"
        if path.exists():
            file_df_parts.append(pd.read_csv(path))

    file_df = pd.concat(file_df_parts, ignore_index=True) if file_df_parts else pd.DataFrame()

    if not file_df.empty:
        model_configuration_differences(file_df, output_dir)
        performance_comparison(file_df, events_by_config, output_dir)

    print("\nDone.")
    print(f"All tables/figures (.png/.svg/.pdf) written under:\n{output_dir}")


if __name__ == "__main__":
    main()