from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Set

import matplotlib.pyplot as plt
import pandas as pd

# =============================================================================
# Configuration
# =============================================================================

STRESS_DIR = Path("Tests_online/stress-test")

# (directory, allowed CSV stems or None for "all files in the directory")
SOURCES: list[tuple[Path, Optional[Set[str]]]] = [
    # The 4 token replay "remaining" variations
    (STRESS_DIR / "results_stress_remaining", None),
    # Behavior + prefix alignments only (old token replay files here are ignored)
    (
        STRESS_DIR / "results_beamline",
        {
            "BehaviorConformance",
            "PrefixAlignment_revert_5",
            "PrefixAlignment_revert_10",
            "PrefixAlignment_revert_20",
        },
    ),
]

PLOTS_DIR = STRESS_DIR / "analysis_remaining"

# Labels used in plots/tables, keyed by CSV filename stem.
PLOT_LABELS: Dict[str, str] = {
    "TokenReplay_remaining_PM4PY": "TBF-PM4PY",
    "TokenReplay_remaining_IGNORE": "TBF-IGN",
    "TokenReplay_remaining_MFS": "TBF-MFS",
    "TokenReplay_remaining_BRD": "TBF-BRD",
    "BehaviorConformance": "Behavior",
    "PrefixAlignment_revert_5": "Prefix_5",
    "PrefixAlignment_revert_10": "Prefix_10",
    "PrefixAlignment_revert_20": "Prefix_20",
}

FIG_SIZE = (11, 4.5)  # less tall than before (was 11 x 6.5)

CORE_COLUMNS = {"event", "total time", "local avg time", "Total space used (MB)"}

NUMERIC_COLUMNS = [
    "event",
    "total time",
    "Average processing time per event",
    "local avg time",
    "Average time (last 10000 events)",
    "Local average time (last 10000 events)",
    "Total space used (MB)",
]

PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Loading
# =============================================================================

def load_results() -> pd.DataFrame:
    frames = []
    for directory, allowed in SOURCES:
        files = sorted(directory.glob("*.csv"))
        if not files:
            print(f"No CSV files in {directory}")
        for path in files:
            if path.name == "comparison_summary.csv":
                continue
            if allowed is not None and path.stem not in allowed:
                continue
            try:
                df = pd.read_csv(path)
            except Exception as exc:
                print(f"Skipping {path.name}: {exc}")
                continue
            missing = CORE_COLUMNS - set(df.columns)
            if missing:
                print(f"Skipping {path.name}: missing {sorted(missing)}")
                continue
            df["configuration"] = PLOT_LABELS.get(path.stem, path.stem.replace("_", " "))
            frames.append(df)

    if not frames:
        raise FileNotFoundError("No usable result CSV files found.")

    results = pd.concat(frames, ignore_index=True)
    for column in NUMERIC_COLUMNS:
        if column in results.columns:
            results[column] = pd.to_numeric(results[column], errors="coerce")
    results.dropna(subset=list(CORE_COLUMNS), inplace=True)
    return results.sort_values(["configuration", "event"]).reset_index(drop=True)


# =============================================================================
# Summary
# =============================================================================

def create_summary(results: pd.DataFrame) -> pd.DataFrame:
    final = results.groupby("configuration", as_index=False).tail(1)
    summary = final[
        ["configuration", "total time", "local avg time", "Total space used (MB)"]
    ].rename(
        columns={
            "configuration": "Configuration",
            "total time": "Total time (ms)",
            "local avg time": "Avg time per event (ms)",
            "Total space used (MB)": "Peak memory (MB)",
        }
    )
    return summary.sort_values("Avg time per event (ms)").reset_index(drop=True)


# =============================================================================
# Plots
# =============================================================================

def save_figure(fig: plt.Figure, filename: str) -> None:
    for suffix, kwargs in ((".png", {"dpi": 300}), (".svg", {}), (".pdf", {})):
        path = (PLOTS_DIR / filename).with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", **kwargs)
        print(f"Saved: {path}")
    plt.close(fig)


def plot_metric_over_events(
    results: pd.DataFrame,
    column: str,
    title: str,
    ylabel: str,
    filename: str,
    log_scale: bool = False,
) -> None:
    if column not in results.columns:
        print(f"Skipping '{title}': column '{column}' not found.")
        return

    if log_scale:
        title, ylabel, filename = title + " (Log Scale)", ylabel + " [log]", filename + "_log"

    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=150)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    ax.set_xlabel("Number of events processed", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for configuration, group in results.groupby("configuration"):
        ax.plot(
            group["event"],
            group[column],
            marker="o",
            markersize=3,
            linewidth=2,
            label=configuration,
        )

    if log_scale:
        ax.set_yscale("log")
        ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.25)

    ax.legend(title="Configuration", frameon=False, loc="best")
    fig.tight_layout()
    save_figure(fig, filename)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    results = load_results()
    print(
        f"Loaded {len(results):,} checkpoint rows from "
        f"{results['configuration'].nunique()} configurations."
    )

    summary = create_summary(results)
    summary.to_csv(PLOTS_DIR / "comparison_summary.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))

    time_plots = [
        ("total time", "Total Processing Time", "Total time (ms)", "total_processing_time"),
        ("local avg time", "Average Processing Time per Event", "Avg time per event (ms)", "average_processing_time"),
        ("Local average time (last 10000 events)", "Average Time (last 10k events)", "Avg time (ms)", "windowed_average_time"),
        ("Average time (last 10000 events)", "Average Time (last 10k events, total)", "Avg time (ms)", "windowed_average_time_global"),
    ]
    for column, title, ylabel, filename in time_plots:
        plot_metric_over_events(results, column, title, ylabel, filename)
        plot_metric_over_events(results, column, title, ylabel, filename, log_scale=True)

    plot_metric_over_events(
        results, "Total space used (MB)", "Peak Memory Usage", "Peak memory usage (MB)", "memory_usage"
    )

    print(f"\nPlots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()