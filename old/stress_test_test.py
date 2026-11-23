from __future__ import annotations

import csv
import gc
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
from typing import Dict, Iterator, List, Optional, TextIO
from xml.etree import ElementTree as ET

from pm4py.objects.petri_net.importer import importer as pnml_importer
from pybeamline.bevent import BEvent
from pybeamline.stream.base_source import BaseSource
from pybeamline.stream.stream import Stream

from token_replay_new_tests_correlation import (
    Model,
    build_configurations,
    create_checker,
    process_event,
)


STRESS_DIR = Path("Tests_online/stress-test")
MODEL_PATH = STRESS_DIR / "model.pnml"
STREAM_PATH = STRESS_DIR / "stream"
RESULTS_DIR = STRESS_DIR / "results_stress_test"

CHECKPOINT_SIZE = 10_000

# Fallback case id used only when a record genuinely carries no case
# information of its own (e.g. no case column, or a blank case value).
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
    # Memory: this configuration's own tracemalloc peak, since this
    # configuration's checker was created (see run_configuration).
    "Total space used (MB)",
    # Memory attributable to process_event() itself: the change in
    # live traced bytes measured tightly around just that call, so
    # driver/harness bookkeeping (stats lists, CSV buffering, stream
    # parsing) cancels out of the delta rather than being counted.
    # Can be negative when a call frees more state than it allocates.
    "Min processing memory delta (MB)",
    "Max processing memory delta (MB)",
    # Processing memory delta, 5-number summary over the last 10000 events
    "Min processing memory delta last 10000 (MB)",
    "Q1 processing memory delta last 10000 (MB)",
    "Median processing memory delta last 10000 (MB)",
    "Q3 processing memory delta last 10000 (MB)",
    "Max processing memory delta last 10000 (MB)",
]

# XES traces are serialized (one Java toString() dump per record) like:
# <org.deckfour.xes.model.impl.XTraceImpl><log ...><trace>
#   <string key="concept:name" value="case_42"/>
#   <event>...</event>
# </trace></log></org.deckfour.xes.model.impl.XTraceImpl>
XES_NS = "{http://www.xes-standard.org/}"
XTRACE_TAG = "org.deckfour.xes.model.impl.XTraceImpl"
XTRACE_OPEN = f"<{XTRACE_TAG}>"
XTRACE_CLOSE = f"</{XTRACE_TAG}>"

# How much of the file we read up front just to *detect* the format
# (XES vs. CSV/TSV). Streams are expected to be homogeneous, so seeing
# an opening XTraceImpl tag anywhere in this window is enough; if the
# file were somehow plain CSV for 64KB+ before switching to XES this
# would misdetect it, but that's not a shape any of our streams take.
FORMAT_SNIFF_BYTES = 1 << 16  # 64 KB

# Chunk size used while scanning for XES trace blocks incrementally.
XES_READ_CHUNK = 1 << 20  # 1 MB


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


class _PrependedReader:
    """
    Wraps an already-open text handle so that a chunk we already read
    (e.g. while sniffing the file format) is replayed first, and
    further .read(size) calls transparently continue from the
    underlying handle once the prefix is exhausted.

    This lets the XES block scanner keep using a plain `.read(n)`
    interface without us having to re-read from the start of the file.
    """

    def __init__(self, prefix: str, handle: TextIO):
        self._prefix = prefix
        self._handle = handle

    def read(self, size: int = -1) -> str:
        if not self._prefix:
            return self._handle.read(size)

        if size is None or size < 0 or size >= len(self._prefix):
            prefix, self._prefix = self._prefix, ""
            if size is None or size < 0:
                return prefix + self._handle.read()
            return prefix + self._handle.read(size - len(prefix))

        chunk, self._prefix = self._prefix[:size], self._prefix[size:]
        return chunk


def _iter_xes_trace_events(source) -> Iterator[StreamEvent]:
    """
    Streams serialized XTraceImpl records out of `source` (anything
    with a `.read(size)` method, e.g. a file handle or a
    _PrependedReader), treating each <trace> as its own case and each
    of its <event> children as a separate event belonging to that
    case.

    Reads the file in bounded chunks and only ever keeps the
    as-yet-unmatched tail of the file in `buffer` in memory — once a
    complete <XTraceImpl>...</XTraceImpl> block is found, it is parsed
    and immediately dropped from the buffer, rather than holding the
    entire file's text in memory for the life of the generator.
    """

    index = 0
    buffer = ""

    while True:
        chunk = source.read(XES_READ_CHUNK)

        if not chunk:
            break

        buffer += chunk

        while True:
            start = buffer.find(XTRACE_OPEN)

            if start == -1:
                # No block opening anywhere in the buffer; keep only a
                # small tail in case the tag itself straddles the next
                # read, and drop the (irrelevant) rest.
                keep = max(len(XTRACE_OPEN), len(XTRACE_CLOSE)) - 1
                buffer = buffer[-keep:] if keep > 0 else ""
                break

            end = buffer.find(XTRACE_CLOSE, start)

            if end == -1:
                # Found the start of a block but it hasn't closed
                # within what we've read so far; wait for more data.
                buffer = buffer[start:]
                break

            end += len(XTRACE_CLOSE)
            block = buffer[start:end]
            buffer = buffer[end:]

            try:
                root = ET.fromstring(block)
            except ET.ParseError:
                continue

            for trace in root.iter(f"{XES_NS}trace"):
                case_id = _find_attr_value(
                    trace, "string", "concept:name"
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


def _sniff_csv(first_line: str) -> tuple:
    """Determines the CSV dialect and header presence from one line."""

    try:
        dialect = csv.Sniffer().sniff(first_line, delimiters=",;\t")
        has_header = csv.Sniffer().has_header(first_line)
    except csv.Error:
        dialect = csv.excel
        has_header = False

    return dialect, has_header


def _iter_plain_or_csv_events(
    first_line: str,
    remaining_lines: Iterator[str],
) -> Iterator[StreamEvent]:
    """
    Supports CSV/TSV data with an activity column (and optionally a
    case column). Rows with no case column, or a blank case value,
    fall back to UNKNOWN_CASE_ID.

    Streams line-by-line: only `first_line` (already read while
    sniffing the format) is held onto, everything else is pulled from
    `remaining_lines` lazily via csv.DictReader, instead of buffering
    the whole file into one string/StringIO up front.
    """

    dialect, has_header = _sniff_csv(first_line)

    if not has_header:
        raise ValueError(
            "The stream has no header. Expected CSV/TSV with a header."
        )

    def _all_lines() -> Iterator[str]:
        yield first_line
        yield from remaining_lines

    reader = csv.DictReader(_all_lines(), dialect=dialect)

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

        # `case_id` is reset every row so a missing case column (or a
        # blank case value on this particular row) always falls back
        # to UNKNOWN_CASE_ID, instead of silently reusing whatever the
        # previous row happened to have (or raising NameError on row 0
        # when there's no case column at all).
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
    occurrence, in order, without ever buffering the whole file into
    memory at once.

    Auto-detects the format from a bounded peek at the start of the
    file:

    - serialized XES traces (each <trace> is treated as its own case,
      so multiple traces in the same file stay separate) — scanned
      incrementally in fixed-size chunks
    - CSV/TSV data with a header (activity column, optional case
      column) — read lazily line by line via csv.DictReader
    """

    handle = path.open(
        "r",
        encoding="utf-8-sig",
        errors="replace",
    )

    try:
        peek = handle.read(FORMAT_SNIFF_BYTES)
    except Exception:
        handle.close()
        raise

    if not peek:
        handle.close()
        return

    try:
        if XTRACE_TAG in peek:
            yield from _iter_xes_trace_events(
                _PrependedReader(peek, handle)
            )
            return

        peek_lines = peek.splitlines(keepends=True)
        first_line = peek_lines[0] if peek_lines else ""

        def _remaining() -> Iterator[str]:
            yield from peek_lines[1:]
            yield from handle

        yield from _iter_plain_or_csv_events(first_line, _remaining())
    finally:
        handle.close()


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

    # Running min/max of the per-event processing memory delta (see
    # handle_event: memory_after - memory_before around process_event).
    min_memory_delta_mb: float = math.inf
    max_memory_delta_mb: float = -math.inf

    # Raw per-event samples within the current 10000-event window, kept
    # only so a five-number summary (box-plot stats) can be computed at
    # each checkpoint, then cleared.
    window_total_samples: List[float] = field(default_factory=list)
    window_local_samples: List[float] = field(default_factory=list)
    window_memory_delta_samples: List[float] = field(
        default_factory=list
    )


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

    Memory independence between configurations: any still-alive
    objects from a previous configuration's checker are forced to be
    collected (Petri-net/marking structures commonly form reference
    cycles that don't die the instant they go out of scope, so simply
    waiting isn't enough), and tracemalloc itself is stopped and
    restarted so its trace table — and thus every current/min/max/peak
    reading below — reflects only allocations made during THIS
    configuration's run. `model`, which is loaded once and shared
    across all configurations, was already allocated before any
    tracemalloc.start() call, so it never appears in these figures for
    any configuration (tracemalloc only tracks blocks allocated while
    tracing is active).

    Returns the total wall-clock time (seconds) spent driving the
    stream for this configuration.
    """

    gc.collect()

    if tracemalloc.is_tracing():
        tracemalloc.stop()

    tracemalloc.start()

    checker = create_checker(
        configuration_type=configuration_type,
        model=model,
        **checker_kwargs,
    )

    result_path = output_path(configuration_name)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    stats = _RunStats()

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

            # Bracket the memory reading tightly around process_event
            # itself. Anything the harness already had resident before
            # this call (stats/window lists, CSV buffering, the stream
            # parser's own state) is present in both `memory_before`
            # and `memory_after`, so it cancels out of the delta — the
            # delta reflects only what THIS call to process_event
            # caused to remain allocated (checker state growing), or
            # freed (checker state being trimmed), independent of the
            # surrounding harness overhead.
            memory_before, _ = tracemalloc.get_traced_memory()

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

            memory_after, _ = tracemalloc.get_traced_memory()

            event_time_ms = (
                perf_counter_ns() - event_started
            ) / 1_000_000.0

            memory_delta_mb = (
                (memory_after - memory_before) / (1024 * 1024)
            )

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
            stats.min_memory_delta_mb = min(
                stats.min_memory_delta_mb, memory_delta_mb
            )
            stats.max_memory_delta_mb = max(
                stats.max_memory_delta_mb, memory_delta_mb
            )

            stats.window_total_samples.append(event_time_ms)
            stats.window_local_samples.append(processing_time_ms)
            stats.window_memory_delta_samples.append(memory_delta_mb)

            if stats.event_count % CHECKPOINT_SIZE != 0:
                return

            # This configuration's own peak, since its checker was
            # created (fresh tracemalloc start above) — dominated by
            # processing state since harness bookkeeping is comparatively
            # tiny (a handful of 10000-element float lists).
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
                memdelta_min,
                memdelta_q1,
                memdelta_median,
                memdelta_q3,
                memdelta_max,
            ) = _five_number_summary(
                stats.window_memory_delta_samples
            )

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
                    "Min processing memory delta (MB)": (
                        stats.min_memory_delta_mb
                    ),
                    "Max processing memory delta (MB)": (
                        stats.max_memory_delta_mb
                    ),
                    "Min processing memory delta last 10000 (MB)": (
                        memdelta_min
                    ),
                    "Q1 processing memory delta last 10000 (MB)": (
                        memdelta_q1
                    ),
                    "Median processing memory delta last 10000 (MB)": (
                        memdelta_median
                    ),
                    "Q3 processing memory delta last 10000 (MB)": (
                        memdelta_q3
                    ),
                    "Max processing memory delta last 10000 (MB)": (
                        memdelta_max
                    ),
                }
            )

            handle.flush()

            stats.window_total_ms = 0.0
            stats.window_local_ms = 0.0
            stats.window_total_samples = []
            stats.window_local_samples = []
            stats.window_memory_delta_samples = []

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

    # Loaded once and shared by every configuration. Because this
    # happens before any tracemalloc.start() call (each one now lives
    # inside run_configuration), `model`'s own memory never shows up
    # in any configuration's tracked figures.
    model = load_model(MODEL_PATH)

    configurations = build_configurations(
        prefix_revert_values=[ 20, 5, 10]
    )

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
        # Safety net: run_configuration always leaves tracing active
        # (ready for the next configuration to gc.collect()+restart),
        # so make sure it's stopped once there are no more configs.
        if tracemalloc.is_tracing():
            tracemalloc.stop()

    print()
    print("Stress test completed")


if __name__ == "__main__":
    main()