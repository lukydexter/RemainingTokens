from __future__ import annotations

import math
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =============================================================================
# Configuration
# =============================================================================

RESULTS_DIR = Path("Tests_online/stress-test/results_stress_test")
PLOTS_DIR = RESULTS_DIR / "analysis_final"
SUMMARY_DIR = PLOTS_DIR / "summary"
CHAR_DIR = PLOTS_DIR / "characteristics"

CHECKPOINT_INTERVAL = 10_000  # events between saved rows
TOTAL_EVENTS = 2_000_000

# Manual override for how configurations are labelled everywhere (plots,
# tables, legends). Keyed by the CSV filename stem (without ".csv"). Any
# configuration not listed here falls back to its filename with underscores
# turned into spaces.
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

for directory in (SUMMARY_DIR, CHAR_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# Columns every result file must contain to be usable at all.
CORE_COLUMNS = {"event", "total time", "local avg time", "Total space used (MB)"}

# Every other numeric column we know how to parse if present. Anything not
# listed here is left untouched (so extra columns don't break loading).
NUMERIC_COLUMNS = [
    "event", "total time", "Average processing time per event", "local avg time",
    "Average time (last 10000 events)", "Local average time (last 10000 events)",
    "Min processing time (ms)", "Max processing time (ms)",
    "Min processing time last 10000 (ms)", "Q1 processing time last 10000 (ms)",
    "Median processing time last 10000 (ms)", "Q3 processing time last 10000 (ms)",
    "Max processing time last 10000 (ms)",
    "Local min time (ms)", "Local max time (ms)",
    "Local min time last 10000 (ms)", "Local Q1 time last 10000 (ms)",
    "Local median time last 10000 (ms)", "Local Q3 time last 10000 (ms)",
    "Local max time last 10000 (ms)",
    "Total space used (MB)",
    "Min processing memory delta (MB)", "Max processing memory delta (MB)",
    "Min processing memory delta last 10000 (MB)", "Q1 processing memory delta last 10000 (MB)",
    "Median processing memory delta last 10000 (MB)", "Q3 processing memory delta last 10000 (MB)",
    "Max processing memory delta last 10000 (MB)",
]

TIME_BAND_COLUMNS = [
    "Local min time last 10000 (ms)", "Local Q1 time last 10000 (ms)",
    "Local median time last 10000 (ms)", "Local Q3 time last 10000 (ms)",
    "Local max time last 10000 (ms)",
]

MEMORY_BAND_COLUMNS = [
    "Min processing memory delta last 10000 (MB)", "Q1 processing memory delta last 10000 (MB)",
    "Median processing memory delta last 10000 (MB)", "Q3 processing memory delta last 10000 (MB)",
    "Max processing memory delta last 10000 (MB)",
]


# =============================================================================
# Loading results
# =============================================================================

def configuration_name_from_file(path: Path) -> str:
    """some_configuration.csv -> 'some configuration', unless overridden in PLOT_LABELS."""
    if path.stem in PLOT_LABELS:
        return PLOT_LABELS[path.stem]
    return path.stem.replace("_", " ")


def slugify(name: str) -> str:
    """'some configuration' -> 'some_configuration' (safe for filenames)."""
    return name.strip().lower().replace(" ", "_")


def load_results(results_dir: Path) -> pd.DataFrame:
    """Load all stress-test result CSVs, keeping every recognised numeric column."""
    files = sorted(results_dir.glob("*.csv"))
    files = [p for p in files if p.name != "comparison_summary.csv"]
    if not files:
        raise FileNotFoundError(f"No result CSV files found in {results_dir}")

    frames = []
    for path in files:
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print(f"Skipping {path.name}: could not read CSV ({exc})")
            continue

        missing = CORE_COLUMNS - set(df.columns)
        if missing:
            print(f"Skipping {path.name}: missing columns {sorted(missing)}")
            continue

        df = df.copy()
        df["configuration"] = configuration_name_from_file(path)
        frames.append(df)

    if not frames:
        raise ValueError("No valid stress-test result CSV files were found.")

    results = pd.concat(frames, ignore_index=True)

    for column in NUMERIC_COLUMNS:
        if column in results.columns:
            results[column] = pd.to_numeric(results[column], errors="coerce")

    results.dropna(subset=list(CORE_COLUMNS), inplace=True)
    results.sort_values(["configuration", "event"], inplace=True)
    results.reset_index(drop=True, inplace=True)
    return results


# =============================================================================
# Summary
# =============================================================================

def create_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Take the final checkpoint for every configuration."""
    final_rows = results.sort_values(["configuration", "event"]).groupby("configuration", as_index=False).tail(1).copy()

    columns = {
        "configuration": "Configuration",
        "total time": "Total time (ms)",
        "local avg time": "Avg time per event (ms)",
        "Total space used (MB)": "Peak memory (MB)",
    }
    # Extra columns are added to the summary only if the CSVs actually have them.
    optional = {
        "Local median time last 10000 (ms)": "Median time, last 10k (ms)",
        "Max processing memory delta last 10000 (MB)": "Max memory delta, last 10k (MB)",
    }
    for source, label in optional.items():
        if source in results.columns:
            columns[source] = label

    summary = final_rows[list(columns.keys())].rename(columns=columns)

    # Average time of the (last-10k-window) averages: mean, across every
    # checkpoint of a run, of the windowed average metrics. This smooths out
    # single-checkpoint noise better than just looking at the last checkpoint.
    windowed_average_columns = {
        "Local average time (last 10000 events)": "Avg of local avg time, last 10k (ms)",
        "Average time (last 10000 events)": "Avg of avg time, last 10k (ms)",
    }
    for source, label in windowed_average_columns.items():
        if source in results.columns:
            means = results.groupby("configuration")[source].mean()
            summary[label] = summary["Configuration"].map(means)

    summary.sort_values("Avg time per event (ms)", inplace=True)
    summary.reset_index(drop=True, inplace=True)
    return summary


def save_summary(summary: pd.DataFrame) -> None:
    output_path = SUMMARY_DIR / "comparison_summary.csv"
    summary.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")


def print_summary(summary: pd.DataFrame) -> None:
    display = summary.copy()
    for column in display.columns:
        if column != "Configuration":
            display[column] = display[column].map(lambda v: f"{v:,.3f}")

    print("\n" + "=" * 105)
    print("STRESS TEST COMPARISON")
    print("=" * 105 + "\n")
    print(display.to_string(index=False))
    print()


# =============================================================================
# Figure saving / styling
# =============================================================================

def save_figure(fig: plt.Figure, filename: str, category: Path = SUMMARY_DIR) -> None:
    """Save a figure as PNG, SVG and PDF inside the given category folder."""
    base_path = category / filename
    for suffix, kwargs in ((".png", {"dpi": 300}), (".svg", {}), (".pdf", {})):
        path = base_path.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", **kwargs)
        print(f"Saved: {path}")
    plt.close(fig)


def setup_plot(title: str, xlabel: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=150)
    ax.set_title(title, fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel(xlabel, fontsize=11, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=11, labelpad=8)
    ax.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax


# =============================================================================
# Summary plots (compare configurations against each other)
# =============================================================================

def plot_metric_over_events(results: pd.DataFrame, column: str, title: str, ylabel: str, filename: str, log_scale: bool = False) -> None:
    """Generic line-per-configuration plot of `column` vs event count."""
    if log_scale:
        title, ylabel, filename = title + " (Log Scale)", ylabel + " [log]", filename + "_log"

    fig, ax = setup_plot(title=title, xlabel="Number of events processed", ylabel=ylabel)
    for configuration, group in results.groupby("configuration"):
        group = group.sort_values("event")
        ax.plot(group["event"], group[column], marker="o", markersize=3, linewidth=2, label=configuration)

    if log_scale:
        ax.set_yscale("log")
        ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.25)

    ax.legend(title="Configuration", frameon=False, loc="best")
    fig.tight_layout()
    save_figure(fig, filename, category=SUMMARY_DIR)


def plot_final_bar(summary: pd.DataFrame, column: str, title: str, ylabel: str, filename: str, value_fmt: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=150)
    bars = ax.bar(summary["Configuration"], summary[column])
    ax.set_title(title, fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Configuration")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, value in zip(bars, summary[column]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), value_fmt.format(value), ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    save_figure(fig, filename, category=SUMMARY_DIR)


def plot_comparison_table(summary: pd.DataFrame) -> None:
    table_data = summary.copy()
    for column in table_data.columns:
        if column != "Configuration":
            table_data[column] = table_data[column].map(lambda v: f"{v:,.3f}")

    height = max(3.5, 2.0 + 0.6 * len(table_data))
    width = max(15, 1.9 * len(table_data.columns))
    fig, ax = plt.subplots(figsize=(width, height), dpi=150)
    ax.axis("off")

    # Wrap long headers onto two lines so columns stay narrow and legible.
    wrapped_labels = [label.replace(" ", "\n", 1) if len(label) > 18 else label for label in table_data.columns]

    table = ax.table(cellText=table_data.values, colLabels=wrapped_labels, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width(col=list(range(len(table_data.columns))))
    table.scale(1, 2.2)
    for cell in table.get_celld().values():
        cell.set_linewidth(0.5)
    for column_index in range(len(table_data.columns)):
        table[0, column_index].set_text_props(fontweight="bold")

    ax.set_title("Stress Test Configuration Comparison", fontsize=17, fontweight="bold", pad=20)
    fig.tight_layout()
    save_figure(fig, "comparison_table", category=SUMMARY_DIR)


# =============================================================================
# Characteristic plots (per-configuration distribution over checkpoints)
# =============================================================================

def plot_band_grid(results: pd.DataFrame, band_columns: list[str], title: str, ylabel: str, filename: str, color: str = "C0", log_scale: bool = False) -> None:
    """
    Faceted grid with one subplot per configuration, showing the windowed
    min-max range, Q1-Q3 band and median line for `band_columns`
    (must be [min, q1, median, q3, max], in that order).
    """
    if not all(col in results.columns for col in band_columns):
        print(f"Skipping '{title}': required columns not present in results.")
        return
    if log_scale:
        title, filename = title + " (Log Scale)", filename + "_log"

    col_min, col_q1, col_median, col_q3, col_max = band_columns
    configurations = sorted(results["configuration"].unique())
    ncols = min(3, len(configurations))
    nrows = math.ceil(len(configurations) / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4 * nrows), dpi=150, sharex=True)
    axes = np.atleast_1d(axes).flatten()

    for ax, configuration in zip(axes, configurations):
        group = results[results["configuration"] == configuration].sort_values("event")
        ax.fill_between(group["event"], group[col_min], group[col_max], alpha=0.15, color=color, label="Min-Max")
        ax.fill_between(group["event"], group[col_q1], group[col_q3], alpha=0.35, color=color, label="Q1-Q3")
        ax.plot(group["event"], group[col_median], color=color, linewidth=1.8, label="Median")
        ax.set_title(configuration, fontsize=10)
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.3)
        if log_scale:
            ax.set_yscale("log")

    for ax in axes[len(configurations):]:
        ax.set_axis_off()

    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.supxlabel("Number of events processed")
    fig.supylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, filename, category=CHAR_DIR)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("\n" + "=" * 105)
    print("STRESS TEST RESULT ANALYSIS")
    print("=" * 105 + "\n")
    print(f"Results directory:\n  {RESULTS_DIR}")
    print(f"Output directory:\n  {PLOTS_DIR}")
    print(f"Checkpoint interval: every {CHECKPOINT_INTERVAL:,} events (up to {TOTAL_EVENTS:,} total)\n")

    print("Loading results...")
    results = load_results(RESULTS_DIR)
    print(f"Loaded {len(results):,} checkpoint rows from {results['configuration'].nunique()} configurations.")

    summary = create_summary(results)
    print_summary(summary)
    save_summary(summary)
    plot_comparison_table(summary)

    # --- Summary plots: compare configurations against each other ----------
    plot_metric_over_events(results, "total time", "Total Processing Time", "Total time (ms)", "total_processing_time")
    plot_metric_over_events(results, "total time", "Total Processing Time", "Total time (ms)", "total_processing_time", log_scale=True)
    plot_metric_over_events(results, "local avg time", "Average Processing Time per Event", "Avg time per event (ms)", "average_processing_time")
    plot_metric_over_events(results, "local avg time", "Average Processing Time per Event", "Avg time per event (ms)", "average_processing_time", log_scale=True)
    if "Local average time (last 10000 events)" in results.columns:
        plot_metric_over_events(results, "Local average time (last 10000 events)", "Average Time (last 10k events)", "Avg time (ms)", "windowed_average_time_local")
        plot_metric_over_events(results, "Local average time (last 10000 events)", "Average Time (last 10k events)", "Avg time (ms)", "windowed_average_time_local", log_scale=True)
    if "Average time (last 10000 events)" in results.columns:
        plot_metric_over_events(results, "Average time (last 10000 events)", "Average Time (last 10k events)", "Avg time (ms)", "windowed_average_time_global")
        plot_metric_over_events(results, "Average time (last 10000 events)", "Average Time (last 10k events)", "Avg time (ms)", "windowed_average_time_global", log_scale=True)
    plot_metric_over_events(results, "Total space used (MB)", "Peak Memory Usage", "Peak memory usage (MB)", "memory_usage")

    plot_final_bar(summary, "Avg time per event (ms)", "Final Average Processing Time per Event", "Avg time per event (ms)", "final_average_time_comparison", "{:.4f}")
    plot_final_bar(summary, "Peak memory (MB)", "Final Peak Memory Usage", "Peak memory (MB)", "final_memory_comparison", "{:.2f}")

    # --- Characteristic plots: distribution over checkpoints, per config ---
    plot_band_grid(results, TIME_BAND_COLUMNS, "Local Processing Time Distribution (last 10k events)", "Time (ms)", "time_characteristics")
    plot_band_grid(results, TIME_BAND_COLUMNS, "Local Processing Time Distribution (last 10k events)", "Time (ms)", "time_characteristics", log_scale=True)
    plot_band_grid(results, MEMORY_BAND_COLUMNS, "Memory Delta Distribution (last 10k events)", "Memory delta (MB)", "memory_delta_characteristics", color="C1")

    print("\n" + "=" * 105)
    print("ANALYSIS COMPLETED")
    print("=" * 105 + "\n")
    print(f"Summary plots:        {SUMMARY_DIR}")
    print(f"Characteristic plots: {CHAR_DIR}\n")


if __name__ == "__main__":
    main()