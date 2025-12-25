from collections import defaultdict

from lark import Lark, Transformer, v_args

the_grammar = r"""
    start: solution | children_group_rule | tour_rule | group_to_tour_rule | bus_to_tours_rule
    solution: "Solution" "[" children_list "|" tour_list "|" group_to_tour_list "|" bus_tours_list "]"
    children_list: "ChildrenList" "[" children_group ("," children_group)* "]"
    children_group_rule: children_group
    children_group: "ChildrenGroup" "<" group_id ">" "[" origin "-" destination "-" size "]"
    group_id: ID
    origin: ID
    destination: ID
    size: /\d+(\.\d+)?/
    
    
    tour_list: "TourList" "[" tour ("," tour)* "]"
    tour_rule: tour
    tour: "Tour" "<" tour_id ">" "[" node_list? "]"
    node_list: node ("-" node)*
    node: ID

    group_to_tour_list: "GroupToTourAssignment" "<GroupToTour>" "[" group_to_tour ("," group_to_tour)* "]"
    group_to_tour: "(" children_id MINUS tour_id ")"
    group_to_tour_rule: group_to_tour_list
    children_id: ID
    tour_id: ID
    
    bus_tours_list: "BusToToursAssignment" "<BusToTours>" "[" bus_to_tours  ("," bus_to_tours)* "]"
    bus_to_tours: "(" tour_id MINUS bus_id ")"
    bus_to_tours_rule: bus_tours_list
    bus_id: ID
    MINUS: "-"
    ID: /[^,\-\[\]<>:\(\)\|]+/


    %import common.WS
    %ignore WS
"""


# --- Transformer ---


@v_args(inline=True)
class SolutionTransformer(Transformer):
    def start(self, item):
        return item

    def solution(self, children_list, tour_list, group_tour_list, bus_tours_list):
        from .decision_variables import (
            Solution,
        )

        return Solution(
            set_of_groups=set(children_list),
            set_of_tours=set(tour_list),
            group_to_tour=group_tour_list,
            bus_to_tours=bus_tours_list,
        )

    def children_list(self, *groups):
        return list(groups)

    def tour_list(self, *tours):
        return list(tours)

    def group_to_tour(self, group_id, _, tour_id):
        return (str(group_id), str(tour_id))

    def bus_to_tours(self, bus_id, _, tour_id):
        return (str(bus_id), str(tour_id))

    def group_to_tour_list(self, *group_to_tours):
        from .decision_variables import GroupToTourAssignment

        the_assignment = GroupToTourAssignment(
            {group_tour[0]: group_tour[1] for group_tour in group_to_tours}
        )
        return the_assignment

    def bus_tours_list(self, *tour_buses):
        from .decision_variables import BusToToursAssignment

        bus_to_tours = defaultdict(list)

        for bus_id, tour_id in tour_buses:
            bus_to_tours[bus_id].append(tour_id)

        bus_to_tours = dict(bus_to_tours)
        the_assignment = BusToToursAssignment(bus_to_tours)
        return the_assignment

    def children_id(self, item):
        return str(item)

    def tour_id(self, item):
        return str(item)

    def bus_id(self, item):
        return str(item)

    def group_to_tour_rule(self, item):
        return item

    def bus_to_tours_rule(self, item):
        return item

    def bus_list(self, *buses):
        return list(buses)

    def combined_decision_rule(self, item):
        return item

    def children_group_rule(self, item):
        return item

    def tour_rule(self, item):
        return item

    def bus_rule(self, item):
        return item

    def decision_list(self, first, *rest):
        decisions = []
        if hasattr(first, 'data') and first.data == 'decision':
            decisions.append(first.children[0])
        for item in rest:
            if isinstance(item, list):
                decisions.extend(item)
            elif hasattr(item, 'data') and item.data == 'decision':
                decisions.append(item.children[0])
        return decisions

    def children_group(self, group_id, origin, destination, size):
        from .decision_variables import ChildrenGroup

        return ChildrenGroup(
            group_id=str(group_id.children[0]),
            origin_name=str(origin.children[0]),
            destination_name=str(destination.children[0]),
            size=float(size.children[0]),
        )

    def tour(self, tour_id, node_list=None):
        from .decision_variables import Tour

        if node_list is None:
            nodes = []
        else:
            nodes = [
                str(n.children[0]) for n in node_list.children
            ]  # every other item (skip separators)
        return Tour(tour_id=tour_id, list_of_nodes=nodes)


# --- Parser setup ---

parser = Lark(
    the_grammar,
    parser="lalr",
    transformer=SolutionTransformer(),
    start='start',
)


def parse_code(code: str):
    return parser.parse(code)
