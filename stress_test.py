from __future__ import annotations

import argparse
import csv
import io
import itertools
import math
import re
import statistics
import tracemalloc
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from time import perf_counter_ns
from typing import Dict, Iterator, List, Optional
from xml.etree import ElementTree as ET

from pm4py.objects.petri_net.importer import importer as pnml_importer
from pybeamline.bevent import BEvent
from pybeamline.stream.base_source import BaseSource
from pybeamline.stream.stream import Stream

from token_replay_new_tests import (
    Model,
    build_configurations,
    create_checker,
    process_event,
)


STRESS_DIR = Path("Tests_online/stress-test")
MODEL_PATH = STRESS_DIR / "model.pnml"
STREAM_PATH = STRESS_DIR / "stream"
RESULTS_DIR = STRESS_DIR / "results_stress_remaining"

CHECKPOINT_SIZE = 10_000

# Fallback case id used only when a record genuinely carries no case
# information of its own (e.g. a CSV without a case column).
UNKNOWN_CASE_ID = "unassigned-case"

# Columns that may hold the activity name in a CSV/TSV stream.
EVENT_COLUMNS = (
    "E:concept:name",
    "concept:name",
    "event",
    "activity",
    "name",
)

# Columns that may hold the case/trace id in a CSV/TSV stream.
CASE_COLUMNS = (
    "T:concept:name",
    "case:concept:name",
    "case_id",
    "case",
    "trace_id",
    "trace",
)

RESULT_COLUMNS = [
    "event",
    "total time",
    "Average processing time per event",
    "local avg time",
    "Average time (last 10000 events)",
    "Local average time (last 10000 events)",
    # Processing time (our outer perf_counter_ns() timer), overall since run start
    "Min processing time (ms)",
    "Max processing time (ms)",
    # Processing time, 5-number summary over just the last 10000 events
    "Min processing time last 10000 (ms)",
    "Q1 processing time last 10000 (ms)",
    "Median processing time last 10000 (ms)",
    "Q3 processing time last 10000 (ms)",
    "Max processing time last 10000 (ms)",
    # Local time (process_event's own self-reported timer), overall since run start
    "Local min time (ms)",
    "Local max time (ms)",
    # Local time, 5-number summary over just the last 10000 events
    "Local min time last 10000 (ms)",
    "Local Q1 time last 10000 (ms)",
    "Local median time last 10000 (ms)",
    "Local Q3 time last 10000 (ms)",
    "Local max time last 10000 (ms)",
    # Memory: tracemalloc's own continuous peak since the run started
    "Total space used (MB)",
    # Memory sampled once per event (tracemalloc's *current* usage),
    # overall since run start
    "Min memory (MB)",
    "Max memory (MB)",
    # Memory, 5-number summary over just the last 10000 events
    "Min memory last 10000 (MB)",
    "Q1 memory last 10000 (MB)",
    "Median memory last 10000 (MB)",
    "Q3 memory last 10000 (MB)",
    "Max memory last 10000 (MB)",
]

# XES traces are serialized (one Java toString() dump per record) like:
# <org.deckfour.xes.model.impl.XTraceImpl><log ...><trace>
#   <string key="concept:name" value="case_42"/>
#   <event>...</event>
# </trace></log></org.deckfour.xes.model.impl.XTraceImpl>
XES_NS = "{http://www.xes-standard.org/}"
XTRACE_TAG = "org.deckfour.xes.model.impl.XTraceImpl"
XTRACE_BLOCK_RE = re.compile(
    rf"<{re.escape(XTRACE_TAG)}>.*?</{re.escape(XTRACE_TAG)}>",
    re.DOTALL,
)


@dataclass
class StreamEvent:
    """A single activity occurrence read from the stream."""

    index: int
    case_id: str
    activity: str
    timestamp: Optional[float] = None


def _parse_xes_timestamp(value: str) -> Optional[float]:
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return None


def _find_attr_value(
    parent: ET.Element,
    tag: str,
    key: str,
) -> Optional[str]:
    for element in parent.findall(f"{XES_NS}{tag}"):
        if element.get("key") == key:
            return element.get("value")

    return None


def _iter_xes_trace_events(text: str) -> Iterator[StreamEvent]:
    """
    Parses one or more serialized XTraceImpl records, treating each
    <trace> as its own case and each of its <event> children as a
    separate event belonging to that case.
    """

    index = 0

    for match in XTRACE_BLOCK_RE.finditer(text):
        try:
            root = ET.fromstring(match.group(0))
        except ET.ParseError:
            continue

        for trace in root.iter(f"{XES_NS}trace"):
            case_id = (
                _find_attr_value(trace, "string", "concept:name")
                or UNKNOWN_CASE_ID
            )

            for event_el in trace.findall(f"{XES_NS}event"):
                activity = _find_attr_value(
                    event_el, "string", "concept:name"
                )

                if not activity:
                    continue

                raw_timestamp = _find_attr_value(
                    event_el, "date", "time:timestamp"
                )
                timestamp = (
                    _parse_xes_timestamp(raw_timestamp)
                    if raw_timestamp
                    else None
                )

                yield StreamEvent(
                    index=index,
                    case_id=case_id,
                    activity=activity,
                    timestamp=timestamp,
                )

                index += 1


def _iter_plain_or_csv_events(text: str) -> Iterator[StreamEvent]:
    """
    Supports a CSV/TSV stream with a header and an activity column
    (and optionally a case column; rows fall back to a single shared
    UNKNOWN_CASE_ID when no case column is found).
    """

    lines = text.splitlines()

    if not lines:
        return

    first_line = lines[0]

    try:
        dialect = csv.Sniffer().sniff(first_line, delimiters=",;\t")
        has_header = csv.Sniffer().has_header(first_line)
    except csv.Error:
        dialect = csv.excel
        has_header = False

    handle = io.StringIO(text)

    if not has_header:
        raise ValueError(
            "The stream has no header. Expected CSV/TSV with a header."
        )

    reader = csv.DictReader(handle, dialect=dialect)

    if reader.fieldnames is None:
        raise ValueError("The stream has no header")

    field_names = {
        field.strip(): field
        for field in reader.fieldnames
        if field is not None
    }

    activity_column: Optional[str] = None

    for column_name in EVENT_COLUMNS:
        if column_name in field_names:
            activity_column = field_names[column_name]
            break

    if activity_column is None:
        raise ValueError(
            "No activity column found. Expected one of: "
            f"{EVENT_COLUMNS}"
        )

    case_column: Optional[str] = None

    for column_name in CASE_COLUMNS:
        if column_name in field_names:
            case_column = field_names[column_name]
            break

    for index, row in enumerate(reader):
        activity = row.get(activity_column)

        if not activity or not activity.strip():
            continue

        # Reset for every row so a missing/blank case value never
        # inherits the previous row's case id (or raises NameError).
        case_id = UNKNOWN_CASE_ID

        if case_column:
            raw_case_id = row.get(case_column)

            if raw_case_id and raw_case_id.strip():
                case_id = raw_case_id.strip()

        yield StreamEvent(
            index=index,
            case_id=case_id,
            activity=activity.strip(),
        )


def read_stream(path: Path) -> Iterator[StreamEvent]:
    """
    Reads the stream file and yields one StreamEvent per activity
    occurrence, in order.

    Auto-detects the format:

    - serialized XES traces (each <trace> is treated as its own case,
      so multiple traces in the same file stay separate)
    - CSV/TSV data with a header, an activity column and an optional
      case column
    """

    with path.open(
        "r",
        encoding="utf-8-sig",
        errors="replace",
    ) as handle:
        text = handle.read()

    if not text:
        return

    if XTRACE_TAG in text:
        yield from _iter_xes_trace_events(text)
        return

    yield from _iter_plain_or_csv_events(text)


def stream_event_to_bevent(stream_event: StreamEvent) -> BEvent:
    """Converts an already-parsed StreamEvent into a pybeamline BEvent."""

    event_time = (
        stream_event.timestamp
        if stream_event.timestamp is not None
        else float(stream_event.index)
    )

    return BEvent(
        stream_event.activity,
        stream_event.case_id,
        event_time=event_time,
    )


class ParsedStreamSource(BaseSource):
    """
    A pybeamline BaseSource that replays our already-parsed StreamEvents
    (from read_stream) as BEvents. Parsing itself is done entirely by
    read_stream; this class only feeds the results into pybeamline's
    Stream/Observable machinery.
    """

    def __init__(self, stream_events: Iterator[StreamEvent]):
        self._stream_events = stream_events

    def execute(self) -> None:
        for stream_event in self._stream_events:
            self.produce(stream_event_to_bevent(stream_event))

        self.completed()


def pybeamline_stream_from_path(path: Path) -> Stream:
    return Stream.source(ParsedStreamSource(read_stream(path)))


def print_first_events(path: Path, count: int = 10) -> None:
    print(f"First {count} events (case id: activity):")

    for stream_event in itertools.islice(read_stream(path), count):
        print(f"  {stream_event.case_id}: {stream_event.activity}")


def load_model(path: Path) -> Model:
    net, initial_marking, final_marking = pnml_importer.apply(
        str(path)
    )

    return Model(
        name=path.stem,
        path=path,
        net=net,
        initial_marking=initial_marking,
        final_marking=final_marking,
    )


@dataclass
class _RunStats:
    event_count: int = 0
    total_time_ms: float = 0.0
    local_time_ms: float = 0.0
    window_total_ms: float = 0.0
    window_local_ms: float = 0.0

    # Running (since run start) min/max — O(1) per event, cheap.
    min_total_ms: float = math.inf
    max_total_ms: float = -math.inf
    min_local_ms: float = math.inf
    max_local_ms: float = -math.inf
    min_memory_mb: float = math.inf
    max_memory_mb: float = -math.inf

    # Raw per-event samples within the current 10000-event window, kept
    # only so a five-number summary (box-plot stats) can be computed at
    # each checkpoint, then cleared.
    window_total_samples: List[float] = field(default_factory=list)
    window_local_samples: List[float] = field(default_factory=list)
    window_memory_samples: List[float] = field(default_factory=list)


def _five_number_summary(
    values: List[float],
) -> tuple[float, float, float, float, float]:
    """Returns (min, Q1, median, Q3, max) for a box plot."""

    q1, median, q3 = statistics.quantiles(
        values, n=4, method="inclusive"
    )

    return min(values), q1, median, q3, max(values)


def output_path(configuration_name: str) -> Path:
    safe_name = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        configuration_name,
    )

    return RESULTS_DIR / f"{safe_name}.csv"


def select_configurations(
    requested: Optional[List[str]] = None,
) -> List[tuple[str, str, dict]]:
    """
    Returns the token replay "remaining policy" configurations
    (PM4PY / IGNORE / MFS / BRD) defined by token_replay_new_tests,
    optionally filtered by name.

    build_configurations only yields token replay configurations now
    (behavior and prefix alignment are disabled there), so the
    prefix-revert argument is unused and passed as an empty list.
    """

    configurations = build_configurations(prefix_revert_values=[])

    if requested is None:
        return configurations

    available_names = {name for name, _, _ in configurations}
    unknown = set(requested) - available_names

    if unknown:
        print(f"UNKNOWN CONFIG: {sorted(unknown)}")

    configurations = [
        config for config in configurations if config[0] in requested
    ]

    if not configurations:
        raise ValueError("No matching configurations")

    return configurations


def run_configuration(
    configuration_name: str,
    configuration_type: str,
    checker_kwargs: Dict[str, object],
    model: Model,
) -> float:
    """
    Runs one configuration end-to-end: read_stream parses the stream
    file into StreamEvents, each is converted to a BEvent, and the
    events are pushed through a pybeamline Stream (built from
    ParsedStreamSource) whose subscribe(on_next=...) callback calls
    process_event(...) and writes checkpoint rows to the results CSV.

    Returns the total wall-clock time (seconds) spent driving the
    stream for this configuration.
    """

    checker = create_checker(
        configuration_type=configuration_type,
        model=model,
        **checker_kwargs,
    )

    result_path = output_path(configuration_name)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    stats = _RunStats()

    tracemalloc.reset_peak()

    with result_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=RESULT_COLUMNS,
        )
        writer.writeheader()

        def handle_event(event: BEvent) -> None:
            event_started = perf_counter_ns()

            (
                _conformance,
                _completeness,
                _confidence,
                _cost,
                processing_time_ms,
            ) = process_event(
                checker=checker,
                configuration_type=configuration_type,
                event=event,
            )

            event_time_ms = (
                perf_counter_ns() - event_started
            ) / 1_000_000.0

            # Sampled *after* the timer above stops, so this bookkeeping
            # (including the tracemalloc call) never leaks into the
            # measured event_time_ms / processing_time_ms values — it
            # only adds a little to the overall wall-clock run time.
            current_memory_bytes, _ = tracemalloc.get_traced_memory()
            current_memory_mb = current_memory_bytes / (1024 * 1024)

            stats.event_count += 1
            stats.total_time_ms += event_time_ms
            stats.local_time_ms += processing_time_ms
            stats.window_total_ms += event_time_ms
            stats.window_local_ms += processing_time_ms

            stats.min_total_ms = min(stats.min_total_ms, event_time_ms)
            stats.max_total_ms = max(stats.max_total_ms, event_time_ms)
            stats.min_local_ms = min(
                stats.min_local_ms, processing_time_ms
            )
            stats.max_local_ms = max(
                stats.max_local_ms, processing_time_ms
            )
            stats.min_memory_mb = min(
                stats.min_memory_mb, current_memory_mb
            )
            stats.max_memory_mb = max(
                stats.max_memory_mb, current_memory_mb
            )

            stats.window_total_samples.append(event_time_ms)
            stats.window_local_samples.append(processing_time_ms)
            stats.window_memory_samples.append(current_memory_mb)

            if stats.event_count % CHECKPOINT_SIZE != 0:
                return

            _, peak_memory_bytes = tracemalloc.get_traced_memory()

            (
                total_min,
                total_q1,
                total_median,
                total_q3,
                total_max,
            ) = _five_number_summary(stats.window_total_samples)

            (
                local_min,
                local_q1,
                local_median,
                local_q3,
                local_max,
            ) = _five_number_summary(stats.window_local_samples)

            (
                memory_min,
                memory_q1,
                memory_median,
                memory_q3,
                memory_max,
            ) = _five_number_summary(stats.window_memory_samples)

            writer.writerow(
                {
                    "event": stats.event_count,
                    "total time": stats.total_time_ms,
                    "Average processing time per event": (
                        stats.total_time_ms / stats.event_count
                    ),
                    "local avg time": (
                        stats.local_time_ms / stats.event_count
                    ),
                    "Average time (last 10000 events)": (
                        stats.window_total_ms / CHECKPOINT_SIZE
                    ),
                    "Local average time (last 10000 events)": (
                        stats.window_local_ms / CHECKPOINT_SIZE
                    ),
                    "Min processing time (ms)": stats.min_total_ms,
                    "Max processing time (ms)": stats.max_total_ms,
                    "Min processing time last 10000 (ms)": total_min,
                    "Q1 processing time last 10000 (ms)": total_q1,
                    "Median processing time last 10000 (ms)": (
                        total_median
                    ),
                    "Q3 processing time last 10000 (ms)": total_q3,
                    "Max processing time last 10000 (ms)": total_max,
                    "Local min time (ms)": stats.min_local_ms,
                    "Local max time (ms)": stats.max_local_ms,
                    "Local min time last 10000 (ms)": local_min,
                    "Local Q1 time last 10000 (ms)": local_q1,
                    "Local median time last 10000 (ms)": local_median,
                    "Local Q3 time last 10000 (ms)": local_q3,
                    "Local max time last 10000 (ms)": local_max,
                    "Total space used (MB)": (
                        peak_memory_bytes / (1024 * 1024)
                    ),
                    "Min memory (MB)": stats.min_memory_mb,
                    "Max memory (MB)": stats.max_memory_mb,
                    "Min memory last 10000 (MB)": memory_min,
                    "Q1 memory last 10000 (MB)": memory_q1,
                    "Median memory last 10000 (MB)": memory_median,
                    "Q3 memory last 10000 (MB)": memory_q3,
                    "Max memory last 10000 (MB)": memory_max,
                }
            )

            handle.flush()

            stats.window_total_ms = 0.0
            stats.window_local_ms = 0.0
            stats.window_total_samples = []
            stats.window_local_samples = []
            stats.window_memory_samples = []

            print(
                f"{configuration_name}: "
                f"{stats.event_count} events completed"
            )

        run_started = perf_counter_ns()

        pybeamline_stream_from_path(STREAM_PATH).subscribe(
            on_next=handle_event,
            on_error=lambda error: print(
                f"{configuration_name}: pybeamline stream error: {error}"
            ),
        )

        total_wall_time_s = (
            perf_counter_ns() - run_started
        ) / 1_000_000_000.0

    print(f"Results written to {result_path}")

    return total_wall_time_s


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Stress test the token replay (remaining policy) "
            "configurations on a long stream"
        )
    )

    parser.add_argument(
        "--configurations",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Run only these configurations, e.g. "
            "--configurations TokenReplay_remaining_BRD "
            "TokenReplay_remaining_MFS. Omit to run all."
        ),
    )

    parser.add_argument(
        "--list-configurations",
        action="store_true",
        help="Print all configuration names and exit",
    )

    args = parser.parse_args()

    if args.list_configurations:
        for name, _, _ in select_configurations():
            print(name)
        return

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    if not STREAM_PATH.exists():
        raise FileNotFoundError(
            f"Stream not found: {STREAM_PATH}"
        )

    print_first_events(STREAM_PATH, count=10)
    print()

    model = load_model(MODEL_PATH)

    configurations = select_configurations(args.configurations)

    tracemalloc.start()

    try:
        for (
            configuration_name,
            configuration_type,
            checker_kwargs,
        ) in configurations:
            print()
            print(f"Running {configuration_name}")

            total_seconds = run_configuration(
                configuration_name=configuration_name,
                configuration_type=configuration_type,
                checker_kwargs=checker_kwargs,
                model=model,
            )

            print(
                f"{configuration_name}: completed in "
                f"{total_seconds:.3f}s"
            )

    finally:
        tracemalloc.stop()

    print()
    print("Stress test completed")


if __name__ == "__main__":
    main()