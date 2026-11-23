"""Visualization helpers for process mining and benchmark outputs."""

from .petri_net_visualization import ModelVisualizer

__all__ = ["ModelVisualizer", "BenchmarkVisualizer"]


def __getattr__(name: str):
	if name == "BenchmarkVisualizer":
		from .benchmark_visualization import BenchmarkVisualizer

		return BenchmarkVisualizer
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
