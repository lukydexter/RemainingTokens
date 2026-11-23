from pandas import DataFrame
from pm4py.objects.petri_net.obj import Marking, PetriNet
from pm4py.objects.petri_net.utils import petri_utils
from pybeamline.bevent import BEvent

import src.algorithm.conformance.token_replay_FULL as token_replay_module
from algorithm.conformance.behaviour_conformance import BehavioralModelBuilder
from algorithm.conformance import (
    convert_b_events_to_log,
)


def test_convert_b_events_to_log_returns_dataframe():
    events = [BEvent("A", "case-1"), BEvent("B", "case-1")]

    log = convert_b_events_to_log(events)

    assert isinstance(log, DataFrame)
    assert len(log) >= 1



def test_token_replay_consume_final_marking_accounts_for_missing_tokens():
    checker = token_replay_module.TokenReplayConformanceChecker.__new__(
        token_replay_module.TokenReplayConformanceChecker
    )
    final_place = PetriNet.Place("p_final")
    other_place = PetriNet.Place("p_other")
    checker.final_marking = Marking({final_place: 2})
    checker.case_states = {
        "case-1": (
            Marking({final_place: 1, other_place: 3}),
            token_replay_module.TokenReplayStats(consumed=4, missing=1),
        )
    }

    checker._consume_final_marking("case-1")

    current_marking, stats = checker.case_states["case-1"]

    assert current_marking == Marking({other_place: 3})
    assert stats.consumed == 6
    assert stats.missing == 2


def test_token_replay_handle_event_updates_case_marking():
    checker = token_replay_module.TokenReplayConformanceChecker.__new__(
        token_replay_module.TokenReplayConformanceChecker
    )
    net = PetriNet("simple")
    p1 = PetriNet.Place("p1")
    p2 = PetriNet.Place("p2")
    net.places.update({p1, p2})
    transition = PetriNet.Transition("A", "A")
    net.transitions.add(transition)
    petri_utils.add_arc_from_to(p1, transition, net)
    petri_utils.add_arc_from_to(transition, p2, net)

    checker.net = net
    checker.initial_marking = Marking({p1: 1})
    checker.final_marking = Marking()
    checker.remaining_policy = token_replay_module.RemainingPolicy.MFS
    checker.case_states = {}
    checker.legal_markings = []

    checker.handle_event(BEvent("A", "case-1"))

    current_marking, stats = checker.case_states["case-1"]

    assert current_marking == Marking({p2: 1})
    assert stats.consumed == 1
    assert stats.produced == 2


def test_behavioral_model_builder_from_petri_net_skips_silent_dfrs():
    net = PetriNet("silent-bridge")
    p0 = PetriNet.Place("p0")
    p1 = PetriNet.Place("p1")
    p2 = PetriNet.Place("p2")
    p3 = PetriNet.Place("p3")
    net.places.update({p0, p1, p2, p3})

    a = PetriNet.Transition("a", "a")
    tau = PetriNet.Transition("tau", None)
    b = PetriNet.Transition("b", "b")
    net.transitions.update({a, tau, b})

    petri_utils.add_arc_from_to(p0, a, net)
    petri_utils.add_arc_from_to(a, p1, net)
    petri_utils.add_arc_from_to(p1, tau, net)
    petri_utils.add_arc_from_to(tau, p2, net)
    petri_utils.add_arc_from_to(p2, b, net)
    petri_utils.add_arc_from_to(b, p3, net)

    initial_marking = Marking({p0: 1})
    final_marking = Marking({p3: 1})

    builder = BehavioralModelBuilder()
    builder.from_petri_net(net, initial_marking, final_marking)
    B, P, F = builder.get_model()

    assert ("a", "b") in B
    assert all(source is not None and target is not None for source, target in B)
    assert set(P.keys()) == set(B)
    assert set(F.keys()) == set(B)
