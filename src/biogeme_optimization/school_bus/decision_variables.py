"""Implementation of the school bus routing and scheduling problem inspired by Spada et al. (2005)

Classes for the decisions

Spada M., Bierlaire M., Liebling T. (2005). Decision-aid methodology
for the school bus routing and scheduling problem. Transportation Science 39 (4):477-490
https://dx.doi.org/10.1287/trsc.1040.0096

Michel Bierlaire
Mon May 26 2025, 10:38:44
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Generator, Iterator

import networkx as nx
from icecream import ic

from .decision_parser import parse_code

logger = logging.getLogger(__name__)

LIST_SEPARATOR = ','
ITEMS_SEPARATOR = '-'


def tour_id_generator() -> Iterator[str]:
    counter = 1
    while True:
        yield f'Merged tour {counter}'
        counter += 1


class Decision(ABC):
    open_bracket = '['
    close_bracket = ']'
    id_open_bracket = '<'
    id_close_bracket = '>'
    id_separator = ':'

    def __init__(self, the_id: str):
        self.the_id = the_id

    @abstractmethod
    def simple_code(self) -> str: ...

    def generate_code(self) -> str:
        """Generate a string code characterizing the decision"""
        return (
            f'{self.__class__.__name__}{self.id_open_bracket}{self.the_id}{self.id_close_bracket}'
            f'{self.open_bracket}{self.simple_code()}{self.close_bracket}'
        )


class ChildrenGroup(Decision):
    def __init__(
        self,
        group_id: str,
        origin_name: str,
        destination_name: str,
        size: float,
    ):
        """A group of children is traveling together, at the same time in the same bus.

        :param group_id: identifier of the group
        :param origin_name: origin of the trip,
        :param destination_name: school where they are going,
        :param size: size of the group.
        """
        super().__init__(the_id=group_id)
        for value_name, value in [
            ('group_id', group_id),
            ('origin_name', origin_name),
            ('destination_name', destination_name),
        ]:
            if ITEMS_SEPARATOR in value:
                raise ValueError(
                    f"{value_name} '{value}' contains forbidden character '{ITEMS_SEPARATOR}'"
                )

        self.group_size: float = size
        self.origin_name: str = origin_name
        self.destination_name: str = destination_name

    def __hash__(self) -> int:
        return hash(
            (self.the_id, self.origin_name, self.destination_name, self.group_size)
        )

    def __eq__(self, other: ChildrenGroup) -> bool:
        if self.the_id != other.the_id:
            return False

        if self.origin_name != other.origin_name:
            raise ValueError(
                f'Group {self.the_id} has two inconsistent origins: {self.origin_name} and {other.origin_name}'
            )
        if self.destination_name != other.destination_name:
            raise ValueError(
                f'Group {self.the_id} has two inconsistent destinations: {self.destination_name} and {other.destination_name}'
            )
        if self.group_size != other.group_size:
            raise ValueError(
                f'Group {self.the_id} has two inconsistent sizes: {self.group_size} and {other.group_size}'
            )
        return True

    @classmethod
    def from_code(cls, the_code: str) -> ChildrenGroup:
        return parse_code(the_code)

    def simple_code(self) -> str:
        """Generates a code characterizing the decision"""
        return (
            f'{self.origin_name}{ITEMS_SEPARATOR}{self.destination_name}{ITEMS_SEPARATOR}'
            f'{self.group_size:.2g}'
        )

    def __str__(self) -> str:
        return (
            f'Group[{self.the_id}] {self.group_size} children from {self.origin_name} '
            f'to {self.destination_name}'
        )

    def __repr__(self) -> str:
        return (
            f'Group[{self.the_id}] {self.group_size} children from {self.origin_name} '
            f'to {self.destination_name}'
        )


def feasible_tours(groups: set[ChildrenGroup]) -> Generator[list[str], None, None]:
    """
    :param groups: groups of children involved
    :return: generator of feasible list of nodes

    Each passenger defines a precedence constraint:
    origin must appear before destination in the tour.

    This can be modeled as a partial order, and then enumerate all linear extensions (i.e., total orders) consistent with this partial order.

    1.	Build the precedence constraints: for each passenger, create a directed edge: origin → destination.
    2.	Add constraints into a graph: represent these constraints as a DAG (directed acyclic graph).
    3.	Enumerate topological sorts of the DAG: Each topological sort corresponds to a feasible tour.


    :return:
    """
    # Build set of all nodes
    nodes = set()
    for group in groups:
        nodes.add(group.origin_name)
        nodes.add(group.destination_name)

    # Build precedence graph
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    for group in groups:
        G.add_edge(group.origin_name, group.destination_name)

    # Ensure there are no cycles (i.e., check for feasibility of the precedence constraints)
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError('Conflicting precedence constraints: no feasible tour exists.')

    # Enumerate all topological sorts (i.e., feasible tours)
    yield from nx.all_topological_sorts(G)


class Tour(Decision):

    def __init__(self, tour_id: str, list_of_nodes: list[str] | None = None):
        if ITEMS_SEPARATOR in tour_id:
            raise ValueError(
                f"Tour ID '{tour_id}' contains forbidden character '{ITEMS_SEPARATOR}'"
            )

        if len(set(list_of_nodes)) != len(list_of_nodes):
            error_msg = (
                f'Tour {tour_id}: a node cannot appear more than once in a tour.'
            )
            raise ValueError(error_msg)

        for node in list_of_nodes:
            if ITEMS_SEPARATOR in node:
                raise ValueError(
                    f"Node name '{node}' contains forbidden character '{ITEMS_SEPARATOR}'"
                )
        super().__init__(the_id=tour_id)
        self.list_of_nodes: list[str] = list_of_nodes if list_of_nodes else []

    def __eq__(self, other: Tour) -> bool:
        if self.the_id != other.the_id:
            return False
        if self.list_of_nodes != other.list_of_nodes:
            raise ValueError(
                f'Tour {self.the_id} has two inconsistent list of nodes: {self.list_of_nodes}'
                f' and {other.list_of_nodes}'
            )
        return True

    def __hash__(self):
        return hash(repr(self))

    @classmethod
    def from_code(cls, the_code: str) -> Tour:
        """Generates an instance of the class from a string code"""
        return parse_code(the_code)

    def simple_code(self) -> str:
        """Generates a code characterizing the decision"""
        return ITEMS_SEPARATOR.join(self.list_of_nodes)

    def __str__(self) -> str:
        return f'Tour {self.the_id}: ' + '->'.join(self.list_of_nodes)

    def __repr__(self) -> str:
        return f'Tour {self.the_id}: ' + '->'.join(self.list_of_nodes)


class GroupToTourAssignment(Decision):
    def __init__(self, assignment: dict[str, str]):
        super().__init__(the_id='GroupToTour')
        if not isinstance(assignment, dict):
            raise TypeError(f'Must be a dict, not {type(assignment)}')
        self.assignment: dict[str, str] = assignment

    def assign(self, group: ChildrenGroup, tour: Tour):
        self.assignment[group.the_id] = tour.the_id

    def get_tour_id(self, group_id: str) -> str:
        return self.assignment[group_id]

    def get_groups_for_tour(self, tour_id: str) -> set[str]:
        return {gid for gid, tid in self.assignment.items() if tid == tour_id}

    def simple_code(self) -> str:
        all_items = [
            f'({group}{ITEMS_SEPARATOR}{tour})'
            for group, tour in self.assignment.items()
        ]
        return LIST_SEPARATOR.join(all_items)

    @classmethod
    def from_code(cls, the_code: str) -> GroupToTourAssignment:
        """Generates an instance of the class from a string code"""
        return parse_code(the_code)


class BusToToursAssignment(Decision):
    def __init__(self, assignment: dict[str, list[str]]):
        super().__init__('BusToTours')
        if not isinstance(assignment, dict):
            raise TypeError(f'Must be a dict, not {type(assignment)}')
        for bus, tours in assignment.items():
            if not isinstance(tours, list):
                raise TypeError(
                    f'Tours for bus {bus} must be a list, not {type(tours)}'
                )
        self.assignment = assignment
        self.assert_unique_tour_assignment()

    def update_buses(self, set_of_buses: set[str]) -> None:
        for bus in set_of_buses:
            if bus not in self.assignment:
                self.assignment[bus] = []

    def assert_unique_tour_assignment(self) -> None:
        tour_to_buses = defaultdict(list)

        for bus, tours in self.assignment.items():
            for tour in tours:
                tour_to_buses[tour].append(bus)

        duplicates = {
            tour: buses for tour, buses in tour_to_buses.items() if len(buses) > 1
        }

        if duplicates:
            messages = [
                f"Tour '{tour}' is assigned to multiple buses: {', '.join(buses)}"
                for tour, buses in duplicates.items()
            ]
            raise ValueError(
                "Duplicate tour assignments detected:\n" + "\n".join(messages)
            )

    def get_bus_id(self, tour_id: str) -> str:
        for bus_id, tour_list in self.assignment.items():
            if tour_id in tour_list:
                return bus_id
        raise KeyError(f'Tour {tour_id} is not assigned to any bus.')

    def get_tours_for_bus(self, bus_id: str) -> list[str]:
        ic(self.assignment)
        return self.assignment[bus_id]

    def simple_code(self) -> str:
        all_items = []
        for bus, tours in self.assignment.items():
            for tour in tours:
                all_items.append(f'({bus}{ITEMS_SEPARATOR}{tour})')
        return LIST_SEPARATOR.join(all_items)

    @classmethod
    def from_code(cls, the_code: str) -> BusToToursAssignment:
        """Generates an instance of the class from a string code"""
        return parse_code(the_code)


class Solution:
    list_separator = ','
    the_generator = tour_id_generator()

    def __init__(
        self,
        set_of_groups: set[ChildrenGroup],
        set_of_tours: set[Tour],
        group_to_tour: GroupToTourAssignment,
        bus_to_tours: BusToToursAssignment,
    ):
        self.set_of_groups = set_of_groups
        self.set_of_tours = set_of_tours
        self.group_to_tour = group_to_tour
        self.bus_to_tours = bus_to_tours

        self.check_names()
        is_correct, message = self.check_tours_validity()
        if not is_correct:
            raise ValueError(message)

    @property
    def group_from_id(self) -> dict[str, ChildrenGroup]:
        return {group.the_id: group for group in self.set_of_groups}

    @property
    def tour_from_id(self) -> dict[str, Tour]:
        return {tour.the_id: tour for tour in self.set_of_tours}

    def generate_unique_tour_id(self):
        while True:
            new_id = next(self.the_generator)
            if new_id not in self.tour_from_id.keys():
                return new_id

    def __str__(self) -> str:
        result = 'Groups\n'
        for group in self.set_of_groups:
            result += f'{group.the_id}: {group.group_size} children from {group.origin_name} to {group.destination_name}\n'
        result += 'Tours\n'
        for tour in self.set_of_tours:
            result += f'{tour.the_id}: {tour.list_of_nodes}\n'
        result += 'Tour assigned to groups\n'
        for group, tour in self.group_to_tour.assignment.items():
            result += f'Group {group} uses tour {tour}\n'
        result += 'Tours assigned to buses\n'
        for bus, list_of_tours in self.bus_to_tours.assignment.items():
            result += f'Bus {bus} serves tours {list_of_tours}\n'
        return result

    def check_names(self) -> None:
        group_names = {group.the_id for group in self.set_of_groups}
        tour_names = {tour.the_id for tour in self.set_of_tours}
        for group, tour in self.group_to_tour.assignment.items():
            if group not in group_names:
                error = f'Unknown group {group}. Known groups: {group_names}'
                raise ValueError(error)
            if tour not in tour_names:
                error = f'Unknown tour {tour}. Known tours: {tour_names}'
            the_tour = self.tour_from_id[tour]
            the_group = self.group_from_id[group]
            if not the_group.origin_name in the_tour.list_of_nodes:
                error = f'Tour {tour} {the_tour.list_of_nodes} is assigned to group {group} but dos not involves its origin {the_group.origin_name}'
                raise ValueError(error)
            if not the_group.destination_name in the_tour.list_of_nodes:
                error = f'Tour {tour} {the_tour.list_of_nodes} is assigned to group {group} but dos not involves its destination {the_group.destination_name}'
                raise ValueError(error)

        for list_of_tours in self.bus_to_tours.assignment.values():
            for tour in list_of_tours:
                if tour not in tour_names:
                    error = f'Unknown tour {tour}. Known tours: {tour_names}'

    @classmethod
    def from_code(cls, the_code: str) -> Solution:
        return parse_code(the_code)

    def update_buses(self, set_of_buses: set[str]) -> None:
        """Complete the bus assignment map with  the buses defined in the problem and currently not used."""
        self.bus_to_tours.update_buses(set_of_buses)

    def generate_code(self):
        groups_codes = sorted(group.generate_code() for group in self.set_of_groups)
        tours_codes = sorted(tour.generate_code() for tour in self.set_of_tours)
        group_to_tour_code = self.group_to_tour.generate_code()
        bus_to_tours_code = self.bus_to_tours.generate_code()
        return (
            f'Solution[ChildrenList[{self.list_separator.join(groups_codes)}]|'
            f'TourList[{self.list_separator.join(tours_codes)}]|'
            f'{group_to_tour_code}|'
            f'{bus_to_tours_code}]'
        )

    def get_groups_for_tour(self, tour_name: str) -> set[ChildrenGroup]:
        group_names = self.group_to_tour.get_groups_for_tour(tour_id=tour_name)
        return {self.group_from_id[group_id] for group_id in group_names}

    def get_tour(self, tour_name: str) -> Tour:
        the_tour = self.tour_from_id.get(tour_name)
        if the_tour is None:
            error = f"Unknown tour: '{tour_name}'. Existing tour: {list(self.tour_from_id.keys())}"
            raise ValueError(error)
        return the_tour

    def get_bus_tours(self, bus_name: str) -> list[Tour]:
        tour_ids = self.bus_to_tours.get_tours_for_bus(bus_name)
        return [self.get_tour(tid) for tid in tour_ids]

    def check_tours_validity(self) -> tuple[bool, str | None]:
        for tour in self.set_of_tours:
            if len(tour.list_of_nodes) != len(set(tour.list_of_nodes)):
                return (
                    False,
                    f'Incorrect list of nodes in tour {tour.the_id}: {tour.list_of_nodes}',
                )

            groups_for_tour: set[ChildrenGroup] = self.get_groups_for_tour(
                tour_name=tour.the_id
            )
            for group in groups_for_tour:
                if tour.list_of_nodes.index(
                    group.origin_name
                ) > tour.list_of_nodes.index(group.destination_name):
                    error_msg = (
                        f'Tour {tour.the_id} cannot be used for group {group.the_id} as '
                        f'origin {group.origin_name} appears after destination {group.destination_name} [{tour}]'
                    )
                    return False, error_msg
        return True, None

    def replace_list_of_nodes(
        self, tour_name: str, new_list_of_nodes: list[str]
    ) -> Solution:
        new_set_of_tours = {
            tour for tour in self.set_of_tours if tour.the_id != tour_name
        }
        new_tour = Tour(tour_id=tour_name, list_of_nodes=new_list_of_nodes)
        new_set_of_tours.add(new_tour)
        new_solution = Solution(
            set_of_groups=self.set_of_groups,
            set_of_tours=new_set_of_tours,
            group_to_tour=self.group_to_tour,
            bus_to_tours=self.bus_to_tours,
        )
        return new_solution
