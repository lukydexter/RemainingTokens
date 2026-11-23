import re

from pm4py.objects.petri_net.obj import PetriNet, Marking

from pm4py.visualization.petri_net import visualizer as pn_vis


class ModelVisualizer:

    @staticmethod
    def _set_or_insert_attr(line: str, key: str, value: str) -> str:
        pattern = rf"\b{re.escape(key)}=([^\s\]]+|\"[^\"]*\")"
        if re.search(pattern, line):
            return re.sub(pattern, f"{key}={value}", line)
        closing = line.rfind("]")
        if closing == -1:
            return line
        return f"{line[:closing]} {key}={value}{line[closing:]}"

    @staticmethod
    def _apply_transition_node_overrides(gviz, node_attrs: dict[str, str]) -> None:
        updated_body: list[str] = []
        for line in gviz.body:
            # Node declarations have '[' and ']' but no edges ('->').
            is_node_decl = "[" in line and "]" in line and "->" not in line
            if not is_node_decl:
                updated_body.append(line)
                continue

            # Places are explicit circles/double-circles in pm4py DOT output.
            is_place = "shape=circle" in line or "shape=doublecircle" in line
            if is_place:
                updated_body.append(line)
                continue

            rewritten = line
            for key, value in node_attrs.items():
                rewritten = ModelVisualizer._set_or_insert_attr(rewritten, str(key), str(value))
            updated_body.append(rewritten)

        gviz.body[:] = updated_body

    @staticmethod
    def visualize_petri_net(
        net: PetriNet,
        initial_marking: Marking,
        final_marking: Marking,
        title: str = "Petri Net",
        save_path: str | None = None,
        output_format: str | None = None,
        graph_attrs: dict[str, str] | None = None,
        node_attrs: dict[str, str] | None = None,
        edge_attrs: dict[str, str] | None = None,
        view: bool = True,
    ):
        parameters = {
            "title": title
        }
        if output_format:
            parameters["format"] = output_format
        gviz = pn_vis.apply(net, initial_marking, final_marking, parameters=parameters)
        if graph_attrs:
            for key, value in graph_attrs.items():
                gviz.graph_attr[str(key)] = str(value)
        if node_attrs:
            for key, value in node_attrs.items():
                gviz.node_attr[str(key)] = str(value)
            # pm4py emits explicit per-node attrs (e.g., fontsize=12), so
            # override transition node lines directly to make size changes stick.
            ModelVisualizer._apply_transition_node_overrides(gviz, node_attrs)
        if edge_attrs:
            for key, value in edge_attrs.items():
                gviz.edge_attr[str(key)] = str(value)
        if save_path:
            pn_vis.save(gviz, save_path)
        if view:
            pn_vis.view(gviz)
        return gviz