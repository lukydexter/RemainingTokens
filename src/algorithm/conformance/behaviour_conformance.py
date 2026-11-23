import sys
from collections import defaultdict
from typing import Optional, List

from typing_extensions import override

from pybeamline.bevent import BEvent
from pybeamline.stream.base_map import BaseMap
from pm4py.objects.petri_net.obj import PetriNet, Marking



class BehavioralConformanceChecker(BaseMap[BEvent, tuple]):

    def __init__(self, model):
        self.model = model
        self.observed_events = 0
        self.bc = BehavioralConformance(M=model)

    @override
    def transform(self, event: BEvent) -> Optional[List[tuple]]:
        self.observed_events += 1
        self.bc.ingest_event(event)
        case_id = event.get_trace_name()
        if case_id in self.bc.get_conformance():
            return [(self.bc.get_conformance(case_id),
                     self.bc.get_confidence(case_id),
                     self.bc.get_completeness(case_id),
                     self.observed_events)]
        return None


def behavioral_conformance(model) -> BehavioralConformanceChecker:
    return BehavioralConformanceChecker(model)


def mine_behavioral_model_from_stream(source) -> tuple:
    bmb = BehavioralModelBuilder()
    source.subscribe(
        on_next=lambda i:
        {
            bmb.ingest_event(i),
        },
        on_error=lambda e, l: print("Error Occurred: {0} to {1}".format(e, l)),
    )
    bmb.end_xes_to_model()

    return bmb.get_model()

def petri_net_2_behavioral_model(petri_net: PetriNet, init_marking: Marking, final_marking: Marking):
   bmb = BehavioralModelBuilder()
   bmb.from_petri_net(petri_net, init_marking, final_marking)
   return bmb.get_model()


# Class originally developed by Magnus Frederiksen as part of his BSc project at DTU entitled
# "Development of Process Mining and Complex Event Processing using Python"
class BehavioralConformance:
    def __init__(self, M=([], defaultdict(list), defaultdict(list))):
        # the input is expecting to be the model
        self.set_model(M)  # calls method to set, B, P, and F
        self.__trace_last_event = dict()  # recalls last event for trace to find relation
        self.__conformance = dict()  # saves a traces' conformance
        self.__completeness = dict()  # saves a traces' completeness
        self.__confidence = dict()  # saves a traces' confidence
        self.__obs = defaultdict(list)  # saves all distinct relations in a trace that has occurred
        self.__inc = dict()  # saves amount of incorrect relations according to the reference model of a trace

    def ingest_event(self, event):
        case_id = event.get_trace_name()  # easier reference to caseID
        event_name = event.get_event_name()  # easier trefermce to eventNa,e

        if case_id not in self.__trace_last_event:  # if this is first time caseID appears
            self.__trace_last_event[case_id] = event_name  # save current event

            # self.__obs[caseID] = []
            self.__inc[case_id] = 0  # set amount of incorrect relations for that CaseId to 0

        else:  # if the caseID has been seen before

            new_pattern = (self.__trace_last_event[case_id], event_name)  # locally save the relation
            # Step 1: update internal data structures
            if new_pattern in self.__B:  # if the relation is in the approved relation list
                if new_pattern not in self.__obs[case_id]:  # and that relation has not occurred for that caseID before
                    self.__obs[case_id].append(new_pattern)  # save the relation to that CaseId
            else:  # if the relation is "illegal" according to B
                self.__inc[case_id] += 1  # increment incorrect for that caseID

            # Step 2: compute online conformance values
            self.__conformance[case_id] = len(self.__obs[case_id]) / (
                    len(self.__obs[case_id]) + self.__inc[case_id])  # calculated conformance

            if new_pattern in self.__B:  # if the relation is legal in B
                # if the relation occurrence is within P_min and P_max
                if self.__P[new_pattern][0] <= len(self.__obs[case_id]) <= self.__P[new_pattern][1]:
                    self.__completeness[case_id] = 1  # set completeness for that caseID
                else:  # if not within P_min and P_max
                    self.__completeness[case_id] = min(1, len(self.__obs[case_id]) / (
                            self.__P[new_pattern][0] + 1))  # calculate completeness

                self.__confidence[case_id] = 1 - (self.__F[new_pattern] / self.__maxOfMinRelationsAfter)
            else:
                self.__confidence[case_id] = -1
                self.__completeness[case_id] = -1

            # Step 3: cleanup
            self.__trace_last_event[case_id] = event_name

    def get_conformance(self, case_id=None):
        if case_id is None:
            return self.__conformance
        return self.__conformance[case_id]

    def get_completeness(self, case_id=None):
        if case_id is None:
            return self.__completeness
        return self.__completeness[case_id]

    def get_confidence(self, case_id=None):
        if case_id is None:
            return self.__confidence
        return self.__confidence[case_id]

    def get_model(self):
        return (self.__B, self.__P, self.__F)

    def set_model(self, M):
        self.__B = M[0]
        self.__P = M[1]
        self.__F = M[2]
        self.__maxOfMinRelationsAfter = 0
        for relations in self.__F:
            self.__maxOfMinRelationsAfter = max(self.__F[relations], self.__maxOfMinRelationsAfter)

# Class originally developed by Magnus Frederiksen as part of his BSc project at DTU entitled
# "Development of Process Mining and Complex Event Processing using Python"
class BehavioralModelBuilder:

    def __init__(self):
        self.__B = []
        self.__P = dict()
        self.__F = dict()
        self.__trace_logs = defaultdict(list)
        self.__mark = []  # used to find P_max for all relation
        self.__mark2 = []

    def get_model(self):
        return self.__B, self.__P, self.__F

    def ingest_event(self, new_event):
        case_id = new_event.get_trace_name()  # locally save caseID
        event_name = new_event.get_event_name()  # locally save event
        if case_id in self.__trace_logs:  # if this caseID has occurred before
            relation = (self.__trace_logs[case_id][-1], event_name)  # find relation based on former event
            trace_length = len(self.__trace_logs[case_id]) - 1  # find the length of the trace so far
            if relation not in self.__B:  # if this relation already exist
                self.__B.append(relation)  # add relation

        else:  # if the trace hasn't been seen yet
            self.__trace_logs[case_id] = []

        self.__trace_logs[case_id].append(event_name)  # add the event to the trace log

    def __set_f(self, relation):
        queue = []  # breath first search from an accepting state
        bfs_mark = []
        bfs_mark.append(relation)
        queue.append((relation, 0))
        while len(queue) > 0:
            ((A, B), depth) = queue.pop(0)
            self.__F[(A, B)] = min(self.__F[(A, B)], depth)
            for (C, D) in self.__B:
                if D == A and (C, D) not in bfs_mark:
                    bfs_mark.append((C, D))
                    queue.append(((C, D), depth + 1))

    def __find_p_max(self, relation, depth):
        if relation not in self.__mark:  # brute force to find the longest path
            self.__mark.append(relation)

            (p_min, p_max) = self.__P[relation]
            self.__P[relation] = (p_min, max(depth, p_max))

            (A, B) = relation
            for (C, D) in self.__B:
                if C == B and depth <= 5:
                    self.__find_p_max((C, D), depth + 1)

            self.__mark.remove(relation)
        else:
            if relation in self.__mark2:
                return
            self.__mark2.append(relation)

            (p_min, p_max) = self.__P[relation]
            self.__P[relation] = (p_min, max(depth, p_max))
            (A, B) = relation
            for (C, D) in self.__B:
                if C == B and depth <= 5:
                    self.__find_p_max((C, D), depth)

            self.__mark2.remove(relation)

    def __set_p(self, relation):
        queue = []  # breath first state from beggining state
        bfs_mark = []
        bfs_mark.append(relation)
        queue.append((relation, 0))
        while len(queue) > 0:
            ((a, b), depth) = queue.pop(0)
            (p_min, p_max) = self.__P[(a, b)]
            self.__P[(a, b)] = (min(depth, p_min), max(depth, p_max))
            for (C, D) in self.__B:
                if C == b and (C, D) not in bfs_mark:
                    bfs_mark.append((C, D))
                    queue.append(((C, D), depth + 1))

        self.__find_p_max(relation, 0)

    def end_xes_to_model(self):

        end_state = []
        start_state = []

        for caseID, trace in self.__trace_logs.items():
            if trace[-1] not in end_state:
                end_state.append(trace[-1])
            if trace[0] not in start_state:
                start_state.append(trace[0])

        self.__set_p_and_a(start_state, end_state)



    def __set_p_and_a(self, start_states: list, end_states: list):
        for relation in self.__B:
            self.__F[relation] = sys.maxsize
            self.__P[relation] = (sys.maxsize, -1)

        for (A, B) in self.__B:
            if A in start_states:
                self.__set_p((A, B))

        for (A, B) in self.__B:
            if B in end_states:
                self.__set_f((A, B))


    def from_petri_net(self, petri_net: PetriNet, init_marking: Marking, final_marking: Marking):
        def _is_visible(transition) -> bool:
            return bool(getattr(transition, "label", None))

        def _forward_visible_from_places(start_places):
            visible_labels = set()
            place_queue = list(start_places)
            visited_places = set()
            visited_silent = set()

            while place_queue:
                place = place_queue.pop(0)
                if place in visited_places:
                    continue
                visited_places.add(place)

                for arc in place.out_arcs:
                    transition = arc.target
                    if _is_visible(transition):
                        visible_labels.add(transition.label)
                        continue

                    if transition in visited_silent:
                        continue
                    visited_silent.add(transition)

                    for out_arc in transition.out_arcs:
                        place_queue.append(out_arc.target)

            return visible_labels

        def _backward_visible_from_places(start_places):
            visible_labels = set()
            place_queue = list(start_places)
            visited_places = set()
            visited_silent = set()

            while place_queue:
                place = place_queue.pop(0)
                if place in visited_places:
                    continue
                visited_places.add(place)

                for arc in place.in_arcs:
                    transition = arc.source
                    if _is_visible(transition):
                        visible_labels.add(transition.label)
                        continue

                    if transition in visited_silent:
                        continue
                    visited_silent.add(transition)

                    for in_arc in transition.in_arcs:
                        place_queue.append(in_arc.source)

            return visible_labels

        dfr_pairs = set()
        for transition in petri_net.transitions:
            if not _is_visible(transition):
                continue

            successor_places = [arc.target for arc in transition.out_arcs]
            for successor_label in _forward_visible_from_places(successor_places):
                dfr_pairs.add((transition.label, successor_label))

        self.__B = list(dfr_pairs)

        start_activities = list(_forward_visible_from_places(list(init_marking.keys())))
        end_activities = list(_backward_visible_from_places(list(final_marking.keys())))
        self.__set_p_and_a(start_activities, end_activities)



