"""Streaming (per-event) token replay built on pm4py's own token-based-replay code.

Four variants that are identical in everything except how *remaining* tokens
are estimated:

    1. PM4PY   - pm4py's native behaviour: every token left in the marking
                 (after the final marking is consumed) is remaining.
    2. IGNORE  - remaining is always 0.
    3. MFS     - remaining = tokens in marking - size of the largest legal
                 (reachable) marking that is contained in the marking.
    4. BRD     - remaining = tokens in marking - size of the legal marking
                 closest to the final marking (reverse BFS from the final
                 marking) that is contained in the marking.

Reused from pm4py.algo.conformance.tokenreplay.variants.token_replay:
    apply_hidden_trans, add_missing_tokens, get_consumed_tokens,
    get_produced_tokens, get_req_transitions_for_final_marking,
    break_condition_final_marking, TechnicalParameters
and pm4py.objects.petri_net.utils.petri_utils.get_places_shortest_path_by_hidden.

All direct pm4py calls are isolated in the adapters `_walk_hidden`,
`_add_missing_tokens` and `_hidden_transitions_to_final`.
"""

from __future__ import annotations

import inspect
from collections import deque
from copy import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from pm4py.algo.conformance.tokenreplay.variants import token_replay as pm4py_tbr
from pm4py.objects.petri_net import semantics
from pm4py.objects.petri_net.obj import Marking, PetriNet
from pm4py.objects.petri_net.utils.petri_utils import get_places_shortest_path_by_hidden
from pybeamline.bevent import BEvent

from src.compass.models.petri.petri_net_utils import PetriNetUtils

# Older pm4py releases do not have the exhaustive_invisible_exploration argument.
_HIDDEN_HAS_EXHAUSTIVE = (
    "exhaustive_invisible_exploration" in inspect.signature(pm4py_tbr.apply_hidden_trans).parameters
)


class RemainingPolicy(Enum):
    PM4PY = 0    # all tokens left in the marking are remaining (pm4py native)
    IGNORE = 1   # remaining = 0
    MFS = 2
    BRD = 3


@dataclass
class _Case:
    marking: Marking
    missing: int = 0
    consumed: int = 0
    produced: int = 0


def _is_sub_marking(small: Marking, big: Marking) -> bool:
    return PetriNetUtils.is_sub_marking(small, big)


def _signature(marking: Marking) -> Tuple[Tuple[str, int], ...]:
    return tuple(sorted((p.name, n) for p, n in marking.items() if n > 0))


def _total(marking: Marking) -> int:
    return PetriNetUtils.get_total_token_count(marking)


class _LegalMarkingIndex:
    """Reachable markings, used by the MFS and BRD estimators (bounded nets only)."""

    def __init__(self, net: PetriNet, im: Marking, fm: Marking, need_brd: bool) -> None:
        from pm4py.objects.petri_net.utils.reachability_graph import marking_flow_petri

        incoming, outgoing, _ = marking_flow_petri(net, im)
        legal = set(incoming) | set(outgoing)
        self.legal_by_size: List[Tuple[Marking, int]] = sorted(
            ((m, _total(m)) for m in legal), key=lambda item: item[1], reverse=True
        )
        self.mfs_cache: Dict[tuple, int] = {}
        self.brd_cache: Dict[tuple, int] = {}
        self.brd_order: List[Marking] = []
        if need_brd and fm in incoming:
            reverse: Dict[Marking, List[Marking]] = {}
            for source, transitions in outgoing.items():
                for target in transitions.values():
                    reverse.setdefault(target, []).append(source)
            seen = {fm}
            queue = deque([fm])
            while queue:
                current = queue.popleft()
                self.brd_order.append(current)
                for nxt in reverse.get(current, []):
                    if nxt not in seen:
                        seen.add(nxt)
                        queue.append(nxt)

    def mfs_tokens(self, marking: Marking) -> int:
        sig = _signature(marking)
        if sig not in self.mfs_cache:
            self.mfs_cache[sig] = next(
                (size for legal, size in self.legal_by_size if _is_sub_marking(legal, marking)), 0
            )
        return self.mfs_cache[sig]

    def brd_tokens(self, marking: Marking) -> int:
        sig = _signature(marking)
        if sig not in self.brd_cache:
            self.brd_cache[sig] = next(
                (_total(c) for c in self.brd_order if _is_sub_marking(c, marking)), 0
            )
        return self.brd_cache[sig]


class PM4PyStreamingTokenReplay:
    """Per-event token replay reusing pm4py's replay primitives.

    Parameters
    ----------
    strict_end_of_trace
        False (default): the final marking is consumed only once it is reached
        (prefix-friendly). True: pm4py's behaviour - at a *virtual* end of the
        trace the final marking is always consumed, adding missing tokens for
        final places not yet marked (and first trying to reach it through hidden
        transitions if `reach_final_through_hidden`). This is evaluated on a copy
        when scoring, so the case state is not modified and replay can continue.
    walk_through_hidden_trans
        Use pm4py's hidden-transition walking to enable a transition.
    check_soundness
        Raise ValueError if the net fails PetriNetUtils.is_sound (as the old checker did).
    count_unknown_as_missing
        pm4py ignores activities that are not in the model (default here too).
        True counts each as 1 missing + 1 consumed (as the old checker did).
    """

    def __init__(
        self,
        net: PetriNet,
        initial_marking: Marking,
        final_marking: Marking,
        remaining_policy: RemainingPolicy = RemainingPolicy.PM4PY,
        strict_end_of_trace: bool = False,
        walk_through_hidden_trans: bool = True,
        reach_final_through_hidden: bool = True,
        exhaustive_invisible_exploration: bool = False,
        count_unknown_as_missing: bool = False,
        check_soundness: bool = True,
    ) -> None:
        # Same check as the old checker (easy soundness, via PetriNetUtils.is_sound).
        if check_soundness and not PetriNetUtils.is_sound(net, initial_marking, final_marking):
            raise ValueError("The reference net is not sound")
        self.net = net
        self.initial_marking = initial_marking
        self.final_marking = final_marking
        self.remaining_policy = remaining_policy
        self.strict_end_of_trace = strict_end_of_trace
        self.walk_through_hidden_trans = walk_through_hidden_trans
        self.reach_final_through_hidden = reach_final_through_hidden
        self.exhaustive_invisible_exploration = exhaustive_invisible_exploration
        self.count_unknown_as_missing = count_unknown_as_missing
        self.case_states: Dict[str, _Case] = {}

        # pm4py's trans_map: per label, transitions sorted by name; pm4py picks the last one.
        self.transitions_by_label: Dict[str, List[PetriNet.Transition]] = {}
        for t in sorted(net.transitions, key=lambda x: x.name):
            if t.label is not None:
                self.transitions_by_label.setdefault(t.label, []).append(t)

        self.places_shortest_path_by_hidden = get_places_shortest_path_by_hidden(
            net, pm4py_tbr.TechnicalParameters.MAX_REC_DEPTH.value
        )

        self.legal_index: Optional[_LegalMarkingIndex] = None
        if remaining_policy in (RemainingPolicy.MFS, RemainingPolicy.BRD):
            self.legal_index = _LegalMarkingIndex(
                net, initial_marking, final_marking, need_brd=remaining_policy == RemainingPolicy.BRD
            )

    # ------------------------------------------------------------------ API
    def handle_event(self, event: BEvent) -> None:
        case_id = event.get_trace_name()
        case = self.case_states.get(case_id)
        if case is None:
            case = _Case(marking=copy(self.initial_marking), produced=sum(self.initial_marking.values()))
            self.case_states[case_id] = case

        candidates = self.transitions_by_label.get(event.get_event_name())
        if candidates:
            self._replay_visible(case, candidates)
        elif self.count_unknown_as_missing:
            case.missing += 1
            case.consumed += 1

        if not self.strict_end_of_trace and _is_sub_marking(self.final_marking, case.marking):
            self._consume_final(case)

    def get_score(self, case_id: str) -> float:
        stats = self.get_stats(case_id)
        if stats["consumed"] > 0 and stats["produced"] > 0:
            return 0.5 * (1.0 - stats["missing"] / stats["consumed"]) + 0.5 * (
                1.0 - stats["remaining"] / stats["produced"]
            )
        return 1.0

    def get_stats(self, case_id: str) -> Dict[str, int]:
        case = self.case_states[case_id]
        marking = copy(case.marking)
        missing, consumed, produced = case.missing, case.consumed, case.produced
        if self.strict_end_of_trace:
            if self.reach_final_through_hidden:
                marking, c, p = self._hidden_transitions_to_final(marking)
                consumed += c
                produced += p
            for place, n in self.final_marking.items():
                if n - marking[place] > 0:
                    missing += n - marking[place]
            consumed += sum(self.final_marking.values())
            for place, n in self.final_marking.items():
                marking[place] = max(0, marking[place] - n)
        return {
            "missing": missing,
            "consumed": consumed,
            "produced": produced,
            "remaining": self._remaining(marking),
        }

    def forget_case(self, case_id: str) -> None:
        self.case_states.pop(case_id, None)

    # ------------------------------------------------------------- internals
    def _replay_visible(self, case: _Case, candidates: List[PetriNet.Transition]) -> None:
        marking = case.marking
        # pm4py: prefer an already-enabled transition with this label, else the trans_map one
        t = next((c for c in candidates if semantics.is_enabled(c, self.net, marking)), candidates[-1])

        if not semantics.is_enabled(t, self.net, marking) and self.walk_through_hidden_trans:
            marking, fired = self._walk_hidden(t, marking)
            for h in fired:
                case.consumed += pm4py_tbr.get_consumed_tokens(h)[0]
                case.produced += pm4py_tbr.get_produced_tokens(h)[0]
            case.marking = marking

        if not semantics.is_enabled(t, self.net, marking):
            case.missing += self._add_missing_tokens(t, marking)

        case.consumed += pm4py_tbr.get_consumed_tokens(t)[0]
        case.produced += pm4py_tbr.get_produced_tokens(t)[0]
        if semantics.is_enabled(t, self.net, marking):
            case.marking = semantics.execute(t, self.net, marking)

    def _consume_final(self, case: _Case) -> None:
        case.consumed += sum(self.final_marking.values())
        for place, n in self.final_marking.items():
            left = case.marking[place] - n
            if left > 0:
                case.marking[place] = left
            else:
                case.marking.pop(place, None)

    def _remaining(self, marking: Marking) -> int:
        total = _total(marking)
        if self.remaining_policy == RemainingPolicy.PM4PY:
            return total
        if self.remaining_policy == RemainingPolicy.IGNORE:
            return 0
        if self.remaining_policy == RemainingPolicy.MFS:
            return total - self.legal_index.mfs_tokens(marking)
        return total - self.legal_index.brd_tokens(marking)

    # ----------------------------------------------------- pm4py adapters
    def _walk_hidden(self, t, marking) -> Tuple[Marking, list]:
        """pm4py.apply_hidden_trans -> [net, marking, act_tr, vis_mark]. Returns (marking, fired transitions)."""
        args = [t, self.net, copy(marking), self.places_shortest_path_by_hidden, [], 0, set(), []]
        if _HIDDEN_HAS_EXHAUSTIVE:
            args.append(self.exhaustive_invisible_exploration)
        _, new_marking, fired, _ = pm4py_tbr.apply_hidden_trans(*args)
        return new_marking, fired

    def _add_missing_tokens(self, t, marking) -> int:
        """pm4py.add_missing_tokens mutates `marking` in place; returns number of missing tokens."""
        missing, _ = pm4py_tbr.add_missing_tokens(t, marking)
        return missing

    def _hidden_transitions_to_final(self, marking: Marking) -> Tuple[Marking, int, int]:
        """Mirrors the first final-marking loop of pm4py.apply_trace. Returns (marking, consumed, produced)."""
        consumed = produced = 0
        for _ in range(pm4py_tbr.TechnicalParameters.MAX_IT_FINAL1.value):
            if pm4py_tbr.break_condition_final_marking(marking, self.final_marking):
                break
            groups = pm4py_tbr.get_req_transitions_for_final_marking(
                marking, self.final_marking, self.places_shortest_path_by_hidden
            )
            for group in groups:
                for h in group:
                    if semantics.is_enabled(h, self.net, marking):
                        marking = semantics.execute(h, self.net, marking)
                        consumed += pm4py_tbr.get_consumed_tokens(h)[0]
                        produced += pm4py_tbr.get_produced_tokens(h)[0]
                if pm4py_tbr.break_condition_final_marking(marking, self.final_marking):
                    break
        return marking, consumed, produced


# --- the four variants, constructor-compatible with PM4PyTokenReplayConformanceChecker(net, im, fm)
class TokenReplayRemainingPM4Py(PM4PyStreamingTokenReplay):
    def __init__(self, net, im, fm, **kwargs):
        super().__init__(net, im, fm, remaining_policy=RemainingPolicy.PM4PY, **kwargs)


class TokenReplayRemainingIgnore(PM4PyStreamingTokenReplay):
    def __init__(self, net, im, fm, **kwargs):
        super().__init__(net, im, fm, remaining_policy=RemainingPolicy.IGNORE, **kwargs)


class TokenReplayRemainingMFS(PM4PyStreamingTokenReplay):
    def __init__(self, net, im, fm, **kwargs):
        super().__init__(net, im, fm, remaining_policy=RemainingPolicy.MFS, **kwargs)


class TokenReplayRemainingBRD(PM4PyStreamingTokenReplay):
    def __init__(self, net, im, fm, **kwargs):
        super().__init__(net, im, fm, remaining_policy=RemainingPolicy.BRD, **kwargs)