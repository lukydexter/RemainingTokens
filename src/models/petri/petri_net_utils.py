import itertools
from typing import Optional

import networkx as nx
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter
from pm4py.objects.petri_net.semantics import enabled_transitions, PetriNetSemantics, execute
from pm4py.objects.petri_net.utils import check_soundness
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils import petri_utils
from pm4py.objects.petri_net.utils.petri_utils import is_sub_marking
from pm4py.objects.petri_net.utils.reachability_graph import marking_flow_petri



class PetriNetUtils:

    @staticmethod
    def load_petri_net_from_file(file_path: str):
        """Load a Petri net from a PNML file. Returns (net, initial_marking, final_marking)."""
        try:
            net, im, fm = pnml_importer.apply(file_path)
            return net, im, fm
        except Exception as e:
            raise RuntimeError(f"Failed to load Petri net from {file_path}: {e}")

    @staticmethod
    def export_petri_net_to_file(net: PetriNet, im: Marking, fm: Marking, file_path: str):
        """Export a Petri net to a PNML file."""
        try:
            pnml_exporter.apply(net, im, file_path, final_marking=fm)
        except Exception as e:
            raise RuntimeError(f"Failed to export Petri net to {file_path}: {e}")

    @staticmethod
    def is_sound(net: PetriNet, im: Marking, fm: Marking) -> bool:
        """Check if the Petri net is sound (easy soundness)."""
        try:
            return check_soundness.check_easy_soundness_net_in_fin_marking(net, im, fm)
        except Exception as e:
            raise RuntimeError(f"Failed to check soundness: {e}")

    @staticmethod
    def get_transition_labels(net: PetriNet):
        """Return a list of transition names in the Petri net."""
        return [t.label or t.name for t in net.transitions]

    @staticmethod
    def get_place_names(net: PetriNet):
        """Return a list of place names in the Petri net."""
        return [p.name for p in net.places]

    @staticmethod
    def get_arcs(net: PetriNet):
        """Return a list of arcs as (source, target, weight)."""
        return [(a.source.name, a.target.name, getattr(a, 'weight', 1)) for a in net.arcs]

    @staticmethod
    def get_total_token_count(marking: Marking) -> int:
        """Return the total token count of the given Marking."""
        count: int = 0
        for place, token_count in marking.items():
            count += token_count
        return count

    @staticmethod
    def get_transition_by_name(name: str, net: PetriNet) -> Optional[PetriNet.Transition]:
        """ Return the transition corresponding to the given name if exists, else return None."""
        return petri_utils.get_transition_by_name(net, name)

    @staticmethod
    def get_transition_by_event_name(event_name: str, net: PetriNet) -> Optional[PetriNet.Transition]:
        """Resolve an event name to a transition by label first, then by transition name."""
        for transition in net.transitions:
            if transition.label == event_name:
                return transition
        return PetriNetUtils.get_transition_by_name(event_name, net)

    @staticmethod
    def get_transition_by_event_name_enabled(event_name: str, net: PetriNet, marking: Marking) -> Optional[PetriNet.Transition]:
        """Resolve an event name to a transition by label first and enabled, then by transition name if enabled."""
        label_match_disabled = None
        name_match_enabled = None
        for transition in net.transitions:
            if PetriNetSemantics.is_enabled(net, transition, marking):
                if transition.label == event_name:
                    return transition
                if transition.name == event_name:
                    name_match_enabled = transition
            elif (transition.label == event_name) or (transition.name == event_name and not label_match_disabled):
                label_match_disabled = transition
        return name_match_enabled if name_match_enabled else label_match_disabled

    @staticmethod
    def is_transition_enabled(net: PetriNet, marking: Marking, transition: PetriNet.Transition) -> bool:
        """Check if the given transition is enabled."""
        return PetriNetSemantics.is_enabled(net, transition, marking)

    @staticmethod
    def enabled_transitions(net: PetriNet, marking: Marking):
        """ Return the enabled transitions in the net given the marking"""
        return enabled_transitions(net, marking)

    @staticmethod
    def fire_transition(net: PetriNet, marking: Marking, transition: PetriNet.Transition):
        """ Fire a transition in the net given and marking"""
        return execute(transition, net, marking)

    @staticmethod
    def is_sub_marking(marking_a: Marking, marking_b: Marking) -> bool:
        """ Return if the given marking is a sub marking of the other"""
        return is_sub_marking(marking_a, marking_b)

    @staticmethod
    def get_all_markings(net: PetriNet, initial_marking: Marking):
        """ Return a list of all markings in the net given the initial marking"""
        incoming_transitions, _, _ = marking_flow_petri(net, initial_marking)
        return list(incoming_transitions.keys())

    @staticmethod
    def get_all_sub_markings(marking: Marking):
        """ Return all possible sub marking for a given marking"""
        active_places = {place: tokens for place, tokens in marking.items() if tokens > 0}
        if not active_places:
            return [Marking()]

        places = list(active_places.keys())
        token_ranges = [range(active_places[place] + 1) for place in places]

        sub_markings = []
        for combination in itertools.product(*token_ranges):
            sub_m = Marking()
            for place, token_count in zip(places, combination):
                if token_count > 0:
                    sub_m[place] = token_count
            sub_markings.append(sub_m)

        return sub_markings

    @staticmethod
    def get_total_tokens(marking: Marking) -> int:
        """
        Returns the total number of tokens present across all places in a given marking.
        """
        return sum(marking.values())

    @staticmethod
    def find_closest_marking_to_end(net, initial_marking: Marking, final_marking: Marking, candidate_markings: list[Marking]) -> Marking:
        """ Returns the closest marking to the final marking from the candidates"""
        incoming_transitions, outgoing_transitions, _ = marking_flow_petri(net, initial_marking)
        g = nx.DiGraph()
        for source_marking, transitions in outgoing_transitions.items():
            for target_marking in transitions.values():
                g.add_edge(source_marking, target_marking)

        if final_marking not in incoming_transitions:
            return Marking()

        reversed_g = g.reverse(copy=False)
        closest_marking = Marking()
        for _, child in nx.bfs_edges(reversed_g, source=final_marking):
            if child in candidate_markings:
                closest_marking = child
                break
        return closest_marking

    @staticmethod
    def find_silent_route_names_via_graph(net: PetriNet, initial_marking: Marking, current_marking: Marking,
                                          target_transition: PetriNet.Transition) -> Optional[list[str]]:
        """
        Finds a route of silent transitions to enable a target transition
        and returns a list of their transition names.
        """
        if PetriNetUtils.is_transition_enabled(net, current_marking, target_transition):
            return []

        _, outgoing_transitions, _ = marking_flow_petri(net, initial_marking)

        g = nx.DiGraph()
        for source_marking, transitions_dict in outgoing_transitions.items():
            for transition, target_marking in transitions_dict.items():
                if transition.label is None:
                    t_name = str(transition.name) if transition.name else str(transition)
                    g.add_edge(source_marking, target_marking, name=t_name)

        if current_marking not in g:
            return None

        destination_markings = {
            m for m in outgoing_transitions
            if PetriNetUtils.is_transition_enabled(net, m, target_transition)
        }

        if not destination_markings:
            return None

        found_destination = None
        for source, child in nx.bfs_edges(g, source=current_marking):
            if child in destination_markings:
                found_destination = child
                break

        if found_destination:
            shortest_path_nodes = nx.shortest_path(g, source=current_marking, target=found_destination)

            silent_route_names: list[str] = []
            for i in range(len(shortest_path_nodes) - 1):
                u = shortest_path_nodes[i]
                v = shortest_path_nodes[i + 1]
                edge_name = str(g[u][v]['name'])
                silent_route_names.append(edge_name)

            return silent_route_names

        return None


