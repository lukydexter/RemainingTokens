from __future__ import annotations

import argparse
import csv
import re
import statistics
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from pathlib import Path
from time import perf_counter_ns
from typing import Dict, List, Optional

from pm4py.objects.petri_net.importer import importer as pnml_importer
from pybeamline.bevent import BEvent

from algorithm.conformance.behaviour_conformance import BehavioralConformanceChecker, petri_net_2_behavioral_model
from algorithm.conformance.prefix_alignment import PrefixAlignmentConformanceChecker
from algorithm.conformance.token_replay_FULL import RemainingPolicy, SilentTransitionHandling, TokenReplayConformanceChecker

LOG_PATTERN = re.compile(r"^log_(?P<model>.+?)\.pnml_noise_trace_.*_noise_event_.*(?:\.csv)?$", re.IGNORECASE)

CASE_COLUMN = "T:concept:name"
EVENT_COLUMN = "E:concept:name"
ID_COLUMN = "id"

DEFAULT_LOGS_DIR = Path("Tests_online/correlation-tests/logs")
DEFAULT_MODELS_DIR = Path("Tests_online/correlation-tests/models")

DEFAULT_RESULTS_DIR = Path("Tests_online/full_TR_conformance_results_speed_FIX")

DEFAULT_SLOW_MODEL_LOAD_S = 10.0
DEFAULT_SLOW_DISCOVER_LOGS_S = 10.0
DEFAULT_SLOW_READ_LOGS_S = 30.0
DEFAULT_SLOW_EVENT_MS = 250.0
DEFAULT_SLOW_EVENT_MULTIPLIER = 20.0
DEFAULT_SLOW_CHECKER_CREATE_S = 30.0
DEFAULT_SLOW_EVENT_LOOP_S = 60.0
DEFAULT_SLOW_CHECKPOINT_LOAD_S = 10.0
DEFAULT_SLOW_CHECKPOINT_WRITE_S = 10.0
DEFAULT_SLOW_AGGREGATE_S = 5.0
DEFAULT_SLOW_OUTPUT_WRITE_S = 30.0
DEFAULT_SLOW_CONFIG_S= 1800.0
MIN_EVENTS_FOR_BASELINE = 20

@contextmanager
def time_section(label:str, warn_threshold_s: Optional[float] = None):
    started = perf_counter_ns()
    try:
        yield
    finally:
        elapsed_s = (perf_counter_ns() - started) / 1_000_000_000.0
        if warn_threshold_s is not None and elapsed_s > warn_threshold_s:
            print(f"[SLOW] {label} took {elapsed_s:.2f}")

@dataclass(frozen=True)
class Model:
    name: str
    path: Path
    net: object
    initial_marking: object
    final_marking: object

@dataclass(frozen=True)
class LogEvent:
    case_id: str
    event_name: str
    event_id: str
    event_index: str

@dataclass
class EventResult:
    configuration: str
    log_file: str
    model_name: str
    case_id: str
    event: str
    event_id: str
    event_index: str
    conformance: Optional[float]
    completeness: Optional[float]
    confidence: Optional[float]
    conformance_raw: Optional[float]
    completeness_raw: Optional[float]
    confidence_raw: Optional[float]
    cost : Optional[float]
    cost_raw : Optional[float]
    total_time_ms: float
    average_processing_time_per_event_ms: float
    local_average_time_ms: float
    total_space_used_mb: float

@dataclass
class SummaryResult:
    configuration: str
    scope: str
    log_file: str
    model_name: str
    file_count: int
    event_count: int
    mean_conformance: float
    mean_completeness: float
    mean_confidence: float
    mean_cost : float
    total_time_ms: float
    average_processing_time_per_event_ms: float
    local_average_time_ms: float
    total_space_used_mb: float


def safe_mean(values: List[Optional[float]]) -> float:
    valid = [float(value) for value in values if value is not None]
    return statistics.mean(valid) if valid else 0.0


def model_name_from_log(path: Path) -> Optional[str]:
    match = LOG_PATTERN.match(path.name)
    if match:
        return match.group("model")

    name = path.name
    if name.startswith("log_") and ".pnml_noise_trace_" in name:
        return name[4:].split(".pnml_noise_trace_", 1)[0]

    return None

def discover_models(models_dir: Path, slow_model_s : float = DEFAULT_SLOW_MODEL_LOAD_S) -> Dict[str, Model]:
    models: Dict[str, Model] = {}

    for model_path in sorted(models_dir.rglob("*.pnml")):
        model_name = model_path.stem

        if model_name in models:
            continue

        with time_section(f"import model {model_name}", slow_model_s):
            net, initial_marking, final_marking = pnml_importer.apply(str(model_path))

        models[model_name] =Model(name=model_name, path=model_path, net=net, initial_marking=initial_marking, final_marking=final_marking)

    if not models:
        raise RuntimeError("No PNML models")

    return models

def discover_logs(logs_dir:Path, slow_model_s : float =DEFAULT_SLOW_DISCOVER_LOGS_S) -> List[tuple[Path, str]]:

    logs: List[tuple[Path, str]] = []
    with time_section(f"discover logs {logs_dir}", slow_model_s):
        for log_path in sorted(logs_dir.rglob("*")):
            if not log_path.is_file():
                continue

            model_name = model_name_from_log(log_path)

            if model_name is not None:
                logs.append((log_path, model_name))

    if not logs:
        raise RuntimeError("No log files")
    return logs

def read_logs(path:Path) -> List[LogEvent]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        sample = handle.read(4096)
        handle.seek(0)

        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel

        reader = csv.DictReader(handle, dialect=dialect)

        if reader.fieldnames is None:
            raise ValueError("Missing header")

        fieldnames = {field.strip().lstrip("\ufeff") for field in reader.fieldnames if field is not None}

        required = {CASE_COLUMN, EVENT_COLUMN, ID_COLUMN}
        missing = required - fieldnames

        if missing:
            raise ValueError("missing columns")

        events: List[LogEvent] =[]

        for index, raw_row in enumerate(reader):

            row = {str(key).strip().lstrip("\ufeff"): ("" if value is None else str(value).strip()) for key, value in raw_row.items()}

            case_id = row.get(CASE_COLUMN, "")
            event_name = row.get(EVENT_COLUMN, "")
            event_id = row.get(ID_COLUMN, str(index))

            if not case_id or not event_name:
                continue

            events.append(LogEvent(case_id=case_id, event_name=event_name, event_id=event_id, event_index=index))

    return events

def token_configurations() -> List[tuple[str, object, object]]:
    configurations = []
    for remaining_policy in RemainingPolicy:
        for silent_handling in SilentTransitionHandling:
            name = ("TokenReplay"
                    f"_remaining_{remaining_policy.name}"
                    f"_silent_{silent_handling.name}"
                    )
            configurations.append((name, remaining_policy, silent_handling))
    # configurations.append(("TokenReplay_remaining_IGNORE_silent_LTIt", RemainingPolicy.IGNORE, SilentTransitionHandling.LTIt))
    return configurations

def create_checker(configuration_type: str, model: Model, remaining_policy= None, silent_handling= None, prefix_revert:Optional[int] = None, behavioral_model_cache:Optional[Dict[str, tuple]] =None, slow_checker_create_s: float = DEFAULT_SLOW_CHECKER_CREATE_S):
    with time_section(f"create checker [{configuration_type}] {model.name}", slow_checker_create_s):
        if configuration_type == "token_replay":
            return TokenReplayConformanceChecker(model.net, model.initial_marking, model.final_marking, remaining_policy=remaining_policy, silent_transition_handling=silent_handling)

        if configuration_type == "behavior":
            if behavioral_model_cache is not None and model.name in behavioral_model_cache:
                behavioral_model = behavioral_model_cache[model.name]

            else:
                behavioral_model = petri_net_2_behavioral_model(model.net, model.initial_marking, model.final_marking)
                if behavioral_model_cache is not None:
                    behavioral_model_cache[model.name] = behavioral_model

            return BehavioralConformanceChecker(behavioral_model)

        if configuration_type == "prefix_alignment":
            return PrefixAlignmentConformanceChecker(model.net, model.initial_marking, model.final_marking, revert_num=prefix_revert)

        raise ValueError(f"Unknown config type")

def process_event(checker, configuration_type:str, event: BEvent)-> tuple[Optional[float], Optional[float], Optional[float], float]:

    started = perf_counter_ns()

    conformance = None
    completeness = None
    confidence = None
    cost = None

    if configuration_type == "token_replay":
        checker.handle_event(event)
        conformance = checker.get_score(event.get_trace_name())

    elif configuration_type == "behavior":
        transformed = checker.transform(event)

        if transformed:
            values = transformed[0]
            conformance = values[0]
            confidence = values[1]
            completeness = values[2]

    elif configuration_type== "prefix_alignment":
        checker.handle_event(event)
        conformance = checker.get_fitness(event.get_trace_name())
        case = checker.case_administration.get(event.get_trace_name())
        if case is not None:
            last_step = case.get_last()
            if last_step is not None:
                cost = last_step.cost

    elapsed_ms = (perf_counter_ns() - started) / 1_000_000.0

    return(conformance, completeness, confidence, cost, elapsed_ms)

def evaluate_file(log_path: Path, model: Model, configuration_name:str, configuration_type:str, checker_kwargs:dict, event_results: List[EventResult], behavioral_model_cache:Optional[Dict[str, tuple]] =None,
                  slow_read_s: float = DEFAULT_SLOW_READ_LOGS_S, slow_checker_create_s: float =DEFAULT_SLOW_CHECKER_CREATE_S, slow_event_loop_s : float = DEFAULT_SLOW_EVENT_LOOP_S, slow_event_ms: float = DEFAULT_SLOW_EVENT_MS,
                  slow_event_multiplier_s : float = DEFAULT_SLOW_EVENT_MULTIPLIER)-> SummaryResult:
    with time_section(f"read logs {log_path.name}", slow_read_s):
        events= read_logs(log_path)     

    checker = create_checker(configuration_type=configuration_type, model=model, behavioral_model_cache=behavioral_model_cache, slow_checker_create_s=slow_checker_create_s, **checker_kwargs)

    tracemalloc.reset_peak()
    event_loop_started = perf_counter_ns()

    event_times: List[float]=[]
    local_times: List[float]=[]
    conformances: List[Optional[float]] =[]
    completenesses: List[Optional[float]] =[]
    confidences : List[Optional[float]] =[]
    costs : List[Optional[float]] = []
    # seen_case_ids:set =set()
    event_time_sum = 0.0
    local_time_sum = 0.0
    processed_count = 0
    # recorded_event_count = 0
    for log_event in events: 
        stream_event = BEvent(log_event.event_name, log_event.case_id, event_time=float(log_event.event_index))

        # is_first_event_of_case = log_event.case_id not in seen_case_ids
        # seen_case_ids.add(log_event.case_id)
        event_started = perf_counter_ns()

        (conformance, completeness, confidence, cost, processing_time_ns) = process_event(checker=checker, configuration_type=configuration_type, event=stream_event)

        total_time_ms = (perf_counter_ns() - event_started) / 1_000_000.0

        # if configuration_type == "behavior" and is_first_event_of_case:
        #     continue
        conformance_raw = conformance
        completeness_raw = completeness
        confidence_raw = confidence
        cost_raw = cost


        if conformance is None:
            conformance = 0.0
        if completeness is None:
            completeness = 0.0
        if confidence is None:
            confidence = 0.0
        if cost is None:
            cost = 0.0

        event_times.append(total_time_ms)
        local_times.append(processing_time_ns)
        conformances.append(conformance_raw)
        completenesses.append(completeness_raw)
        confidences.append(confidence_raw)
        costs.append(cost_raw)

        # recorded_event_count +=1
        event_time_sum +=total_time_ms
        local_time_sum += processing_time_ns
        processed_count +=1

        running_avg_ms = event_time_sum/processed_count
        is_anomalous = (processed_count >= MIN_EVENTS_FOR_BASELINE and total_time_ms > running_avg_ms * slow_event_multiplier_s)

        if  total_time_ms > slow_event_ms or is_anomalous:
            print(f"[SLOW] {configuration_name}, event id {log_event.event_id}")

        _, peak_memory_bytes = tracemalloc.get_traced_memory()
        memory_mb = peak_memory_bytes/ (1024*1024)

        event_results.append(EventResult(
            configuration=configuration_name,
            log_file=log_path.name,
            model_name=model.name,
            case_id=log_event.case_id,
            event=log_event.event_name,
            event_id=log_event.event_id,
            event_index=log_event.event_index,
            conformance=conformance,
            completeness=completeness,
            confidence=confidence,
            conformance_raw=conformance_raw,
            completeness_raw=completeness_raw,
            confidence_raw=confidence_raw,
            cost= cost,
            cost_raw=cost_raw,
            total_time_ms=total_time_ms,
            average_processing_time_per_event_ms=event_time_sum/processed_count,
            local_average_time_ms=local_time_sum/processed_count,
            total_space_used_mb=memory_mb,
        ))

    total_time_ms = sum(event_times)


    return SummaryResult(
        configuration=configuration_name,
        scope="file",
        log_file=log_path.name,
        model_name=model.name,
        file_count=1,
        event_count=len(events),
        mean_conformance=safe_mean(conformances),
        mean_completeness=safe_mean(completenesses),
        mean_confidence=safe_mean(confidences),
        mean_cost= safe_mean(costs),
        total_time_ms=total_time_ms,
        average_processing_time_per_event_ms= (total_time_ms/len(events) if events else 0.0),
        local_average_time_ms=(local_time_sum/processed_count if processed_count else 0.0),
        total_space_used_mb=(tracemalloc.get_traced_memory()[1]/(1024*1024)),

        )

def aggregate(rows:List[SummaryResult], configuration:str, scope:str, model_name:str="") ->SummaryResult:
    event_count = sum(row.event_count for row in rows)
    file_count = len(rows)
    total_time_ms = sum(row.total_time_ms for row in rows)

    def weighted_mean(field:str) -> float:
        numerator = sum(getattr(row, field) * row.event_count for row in rows)
        return numerator/event_count if event_count else 0.0

    return SummaryResult(
        configuration=configuration,
        scope=scope,
        log_file="",
        model_name=model_name,
        file_count=file_count,
        event_count=event_count,
        mean_conformance=weighted_mean("mean_conformance"),
        mean_completeness=weighted_mean("mean_completeness"),
        mean_confidence=weighted_mean("mean_confidence"),
        mean_cost=weighted_mean("mean_cost"),
        total_time_ms=total_time_ms,
        average_processing_time_per_event_ms=(total_time_ms/ event_count if event_count else 0.0),
        local_average_time_ms= statistics.mean([row.local_average_time_ms for row in rows]) if rows else 0.0,
        total_space_used_mb=max([row.total_space_used_mb for row in rows], default=0.0),
        )

def write_csv(path:Path, rows:List[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    fieldnames = list(asdict(rows[0]).keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(asdict(row))

def write_grouped_by_configuration(base_dir:Path, rows: List[object]) ->None:
    grouped: Dict[str, List[object]] ={}
    for row in rows:
        grouped.setdefault(row.configuration, []).append(row)
    for configuration_name, group_rows in grouped.items():
        write_csv(base_dir/ f"{configuration_name}.csv", group_rows)

def write_comparison(path:Path, rows:List[SummaryResult]) -> None:
    grouped : Dict[tuple[str, str], List[SummaryResult]] ={}
    for row in rows:
        grouped.setdefault((row.scope, row.model_name), []).append(row)

    comparison_rows =[]
    for (scope, model_name), group in grouped.items():
        for row in group:
            comparison_rows.append(
                {
                    "scope": scope,
                    "model_name": model_name,
                    "configuration":row.configuration,
                    "event_count":row.event_count,
                    "mean_conformance": row.mean_conformance,
                    "mean_completeness":row.mean_completeness,
                    "mean_confidence":row.mean_confidence,
                    "mean_cost": row.mean_cost,
                    "total_time_ms":row.total_time_ms,
                    "average_processing_time_per_event_ms": row.average_processing_time_per_event_ms,
                    "local_average_time_ms":row.local_average_time_ms,
                    "total_space_used_mb": row.total_space_used_mb,
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        if not comparison_rows:
            return

        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0].keys()))
        writer.writeheader()
        writer.writerows(comparison_rows)


def per_file_checkpoint_dir(results_dir:Path, configuration_name:str)-> Path:
    return results_dir / "per_file_checkpoint" / configuration_name

def checkpoint_directory(results_dir:Path, configuration_name:str, checkpoint_type:str)-> Path:
    return (per_file_checkpoint_dir(results_dir, configuration_name) / checkpoint_type)

def find_checkpoint_path(results_dir:Path, configuration_name:str, log_path: Path, checkpoint_type:str)-> Optional[Path]:
    directory = checkpoint_directory(results_dir,configuration_name, checkpoint_type)
    candidates = sorted(directory.glob(f"{log_path.stem}*.csv"))
    if not candidates:
        return None

    suffix_free_path = directory / f"{log_path.stem}.csv"
    if suffix_free_path.exists():
        return suffix_free_path

    return candidates[0]

def checkpoint_path(results_dir:Path, configuration_name:str, log_path: Path, checkpoint_type:str)-> Path:
    return(checkpoint_directory(results_dir, configuration_name, checkpoint_type) / f"{log_path.stem}.csv")

def event_checkpoint_path(results_dir:Path, configuration_name:str, log_path:Path)-> Path:
    return checkpoint_path(results_dir,configuration_name, log_path, "events")

def summary_checkpoint_path(results_dir:Path, configuration_name:str, log_path:Path)-> Path:
    return checkpoint_path(results_dir,configuration_name, log_path, "summaries")

def _optional_float(value:str) -> Optional[float]:
    if value is None or value =="":
        return None
    return float(value)

def load_event_rows(path:Path) -> List[EventResult]:
    rows: List[EventResult] =[]
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(EventResult(
                configuration=row["configuration"],
                log_file=row["log_file"],
                model_name=row["model_name"],
                case_id=row["case_id"],
                event=row["event"],
                event_id=row["event_id"],
                event_index=row["event_index"],
                conformance=_optional_float(row["conformance"]),
                completeness=_optional_float(row["completeness"]),
                confidence=_optional_float(row["confidence"]),
                conformance_raw=_optional_float(row["conformance_raw"]),
                completeness_raw=_optional_float(row["completeness_raw"]),
                confidence_raw=_optional_float(row["confidence_raw"]),
                cost=_optional_float(row.get("cost")),
                cost_raw=_optional_float(row.get("cost_raw")),
                total_time_ms=float(row["total_time_ms"]),
                average_processing_time_per_event_ms=float(row["average_processing_time_per_event_ms"]),
                local_average_time_ms=float(row["local_average_time_ms"]),
                total_space_used_mb=float(row["total_space_used_mb"]),

            ))
    return rows

def load_summary_rows(path:Path) -> SummaryResult:
    
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        row = next(reader)

    return SummaryResult(
        configuration=row["configuration"],
        scope=row["scope"],
        log_file=row["log_file"],
        model_name=row["model_name"],
        file_count=int(row["file_count"]),
        event_count=int(row["event_count"]),
        mean_conformance=float(row["mean_conformance"]),
        mean_completeness=float(row["mean_completeness"]),
        mean_confidence=float(row["mean_confidence"]),
        mean_cost=float(row.get("mean_cost", 0.0)),
        total_time_ms=float(row["total_time_ms"]),
        average_processing_time_per_event_ms=float(row["average_processing_time_per_event_ms"]),
        local_average_time_ms=float(row["local_average_time_ms"]),
        total_space_used_mb=float(row["total_space_used_mb"]),
    )

def is_file_model_done(results_dir:Path, configuration_name:str, log_path:Path)-> bool:
    return (
        find_checkpoint_path(results_dir, configuration_name, log_path, "events") is not None
        and find_checkpoint_path(results_dir, configuration_name, log_path, "summaries") is not None
    )


def build_configurations(prefix_revert_values: List[Optional[int]]) -> List[tuple[str, str, dict]]:
    configurations = []
    for name, remaining, silent in token_configurations():
        configurations.append((name, "token_replay", {"remaining_policy":remaining, "silent_handling" : silent,}))
    configurations.append(("BehaviorConformance", "behavior", {},))
    for revert_value in prefix_revert_values:
        suffix = ("default" if revert_value is None else str(revert_value))
        configurations.append((f"PrefixAlignment_revert_{suffix}", "prefix_alignment", {"prefix_revert": revert_value}))
    return configurations

def run(logs_dir: Path, models_dir: Path, results_dir:Path, prefix_revert_values: List[Optional[int]], configuration_filter:Optional[List[str]]= None, 
        slow_checkpoint_write_s: Optional[float] = DEFAULT_SLOW_CHECKPOINT_WRITE_S, slow_output : Optional[float] = DEFAULT_SLOW_OUTPUT_WRITE_S, slow_aggregate:Optional[float] = DEFAULT_SLOW_AGGREGATE_S)->None:

    results_dir.mkdir(parents=True, exist_ok=True)

    models = discover_models(models_dir)
    logs = discover_logs(logs_dir)

    event_results:List[EventResult]=[]
    file_results: List[SummaryResult]=[]
    model_results: List[SummaryResult]=[]
    overall_results: List[SummaryResult]=[]

    configurations =build_configurations(prefix_revert_values)

    if configuration_filter is not None:
        available_names = {name for name, _, _, in configurations}
        requested = set(configuration_filter)
        unknown = requested - available_names
        if unknown:
            print(f"UNKNOWN CONFIG: {sorted(unknown)}")
        configurations = [config for config in configurations if config[0] in requested]

        if not configurations:
            raise ValueError("NO matching configs")

    behavioral_model_cache: Dict[str, tuple] ={}
    tracemalloc.start()


    for configuration_name, configuration_type, checker_kwargs in configurations:
        print(f"Running {configuration_name}")
        current_file_results: List[SummaryResult]=[]

        for log_path, model_name in logs:
            model = models.get(model_name)

            if model is None:
                print(f"Missing {log_path.name}")
                continue

            loaded_from_checkpoint = False

            if is_file_model_done(results_dir, configuration_name, log_path):
                try: 
                    print(f"{log_path.name} already done, load checkpoint")
                    # result = load_summary_rows(summary_checkpoint_path(results_dir, configuration_name, log_path))
                    # event_results.extend(load_event_rows(event_checkpoint_path(results_dir, configuration_name, log_path)))
                    # loaded_summary = load_summary_rows(summary_checkpoint_path(results_dir, configuration_name, log_path))
                    # loaded_events = load_event_rows(event_checkpoint_path(results_dir, configuration_name, log_path))
                    summary_path = find_checkpoint_path(results_dir, configuration_name, log_path, "summaries")
                    events_path = find_checkpoint_path(results_dir, configuration_name, log_path, "events")
                    if summary_path is None or events_path is None:
                        raise FileNotFoundError("checkpoint file missing")
                    loaded_summary = load_summary_rows(summary_path)
                    loaded_events = load_event_rows(events_path)
                    loaded_summary.mean_cost = safe_mean([event.cost_raw for event in loaded_events])
                except (StopIteration, KeyError, ValueError, FileNotFoundError) as exc:
                    print(f"{log_path.name}-> checkpoint unreadable ({exc})")
                else:
                    if loaded_summary.log_file != log_path.name:
                        print("Name mismatch")
                    else:
                        print(f"{log_path.name} already done, load checkpoint")
                        result = loaded_summary
                        event_results.extend(loaded_events)
                        loaded_from_checkpoint =True

            if not loaded_from_checkpoint:

                print(f"{log_path.name}")
                events_before = len(event_results)
                try:
                    result = evaluate_file(log_path=log_path, model=model, configuration_name=configuration_name, configuration_type=configuration_type, checker_kwargs=checker_kwargs, event_results=event_results, behavioral_model_cache=behavioral_model_cache)
                except Exception as exc:
                    print(f"FAILED {exc}")
                    del event_results[events_before:]
                    continue

                new_events = event_results[events_before:]
                with time_section(f"Write checkpoint",slow_checkpoint_write_s):
                    write_csv(event_checkpoint_path(results_dir, configuration_name, log_path), new_events)
                    # write_comparison(summary_checkpoint_path(results_dir, configuration_name, log_path), [result])
                    write_csv(summary_checkpoint_path(results_dir, configuration_name, log_path), [result])
                print(f"conformance {result.mean_conformance}")

            file_results.append(result)
            current_file_results.append(result)

        with time_section(f"Aggregate", slow_aggregate):
            by_model:Dict[str, List[SummaryResult]] = {}

            for result in current_file_results:
                by_model.setdefault(result.model_name, []).append(result)

            for model_name, rows in by_model.items():
                model_result = aggregate(rows=rows, configuration=configuration_name, scope="model", model_name=model_name)

                model_results.append(model_result)

            if current_file_results:
                overall_result = aggregate(rows=current_file_results, configuration=configuration_name, scope="overall")
                overall_results.append(overall_result)

    with time_section(f"Write final outpusts", slow_output):
        write_grouped_by_configuration(results_dir / "events", event_results)
        write_grouped_by_configuration(results_dir / "summaries"/ "file", file_results)
        write_grouped_by_configuration(results_dir / "summaries"/ "model", model_results)
        write_grouped_by_configuration(results_dir / "summaries"/ "overall", overall_results)

        write_comparison(results_dir/ "configuration_comparison.csv", model_results+overall_results)

    tracemalloc.stop()

    print()
    print("Completed")


def main() -> None:

    parser = argparse.ArgumentParser(description=("Evaluate indipendently logs with models with different algorithms"))

    parser.add_argument("--logs-dir", type=Path, default=DEFAULT_LOGS_DIR)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--prefix-revert", type=int, nargs="*", default=[5,10, 20], help=("Prefix-alignment revert values"))

    parser.add_argument(
        "--configurations",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Run configs eg -- configurations BehaviorConformance PrefixAlignment_revert_10"
            "Omit to run all. use --list-configurations"
        )
    )

    parser.add_argument(
        "--list-configurations",
        action="store_true",
        help=("Print all configs and exit without run")
    )

    args = parser.parse_args()
    prefix_values: List[Optional[int]] = list(args.prefix_revert)

    if args.list_configurations:
        for name, _, _ in build_configurations(prefix_values):
            print(name)
        return

    run(logs_dir=args.logs_dir, models_dir=args.models_dir, results_dir=args.results_dir, prefix_revert_values=prefix_values, configuration_filter=args.configurations,)

if __name__ == "__main__":
    main()

