"""Implementation of the school bus routing and scheduling problem inspired by Spada et al. (2005)

Implementation of the operators

Spada M., Bierlaire M., Liebling T. (2005). Decision-aid methodology
for the school bus routing and scheduling problem. Transportation Science 39 (4):477-490
https://dx.doi.org/10.1287/trsc.1040.0096

Michel Bierlaire
Mon May 26 2025, 10:38:44
"""

import functools
import random
from collections.abc import Callable
from itertools import islice

from icecream import ic

from biogeme_optimization.pareto import SetElement

from .decision_variables import (
    BusToToursAssignment,
    ChildrenGroup,
    GroupToTourAssignment,
    Solution,
    Tour,
    feasible_tours,
)
from .element_pareto import ElementSolution
from .starting_points import generate_tour_for_groups

MAXIMUM_COMBINATIONS_FOR_TOURS = 10000


def log_function_call(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Calling: {func.__name__}")
        return func(*args, **kwargs)

    return wrapper


def split_set_randomly(s: set) -> tuple[set, set]:
    """Splits a set into two randomly assigned subsets of approximately equal size."""
    elements = list(s)
    random.shuffle(elements)
    midpoint = len(elements) // 2
    return set(elements[:midpoint]), set(elements[midpoint:])


def improve_tour(
    element: SetElement,
    improvement_criterion: Callable[[ElementSolution, ElementSolution], bool],
    size: int = 1,
) -> tuple[SetElement | None, int]:
    """Generic tour improvement function based on a given criterion"""
    current_solution = ElementSolution.from_code(element.element_id)
    list_of_tour_names = [
        tour.the_id for tour in current_solution.the_solution.set_of_tours
    ]
    actual_number_of_changes = 0
    for tour_name in islice(list_of_tour_names, size):
        children = current_solution.the_solution.get_groups_for_tour(
            tour_name=tour_name
        )
        the_generator = feasible_tours(groups=children)
        for list_of_nodes in islice(the_generator, MAXIMUM_COMBINATIONS_FOR_TOURS):
            new_solution = current_solution.the_solution.replace_list_of_nodes(
                tour_name=tour_name, new_list_of_nodes=list_of_nodes
            )
            new_element_solution = ElementSolution(the_solution=new_solution)
            if improvement_criterion(current_solution, new_element_solution):
                current_solution = new_element_solution
                actual_number_of_changes += 1
                break
    return current_solution.get_element(), actual_number_of_changes


@log_function_call
def improve_tour_travel_time(
    element: SetElement, size: int = 1
) -> tuple[SetElement | None, int]:
    """Generate a tour that improves the total travel time"""

    def is_better_by_travel_time(
        current: ElementSolution, new: ElementSolution
    ) -> bool:
        return new.timetable.total_travel_time < current.timetable.total_travel_time

    return improve_tour(
        element, improvement_criterion=is_better_by_travel_time, size=size
    )


@log_function_call
def improve_tour_maximum_travel_time(
    element: SetElement, size: int = 1
) -> tuple[SetElement | None, int]:
    """Generate a tour that improves the total travel time"""

    def is_better_by_maximum_travel_time(
        current: ElementSolution, new: ElementSolution
    ) -> bool:
        return new.timetable.maximum_travel_time < current.timetable.maximum_travel_time

    return improve_tour(
        element, improvement_criterion=is_better_by_maximum_travel_time, size=size
    )


@log_function_call
def improve_tour_early_arrival(
    element: SetElement, size: int = 1
) -> tuple[SetElement | None, int]:
    """Generate a tour that improves the total travel time"""

    def is_better_by_early_arrivals(
        current: ElementSolution, new: ElementSolution
    ) -> bool:
        return (
            new.timetable.total_early_arrivals < current.timetable.total_early_arrivals
        )

    return improve_tour(
        element, improvement_criterion=is_better_by_early_arrivals, size=size
    )


@log_function_call
def improve_tour_maximum_early_arrivals(
    element: SetElement, size: int = 1
) -> tuple[SetElement | None, int]:
    """Generate a tour that improves the total travel time"""

    def is_better_by_maximum_early_arrivals(
        current: ElementSolution, new: ElementSolution
    ) -> bool:
        return (
            new.timetable.maximum_early_arrivals
            < current.timetable.maximum_early_arrivals
        )

    return improve_tour(
        element, improvement_criterion=is_better_by_maximum_early_arrivals, size=size
    )


@log_function_call
def improve_tour_late_arrivals(
    element: SetElement, size: int = 1
) -> tuple[SetElement | None, int]:
    """Generate a tour that improves the total travel time"""

    def is_better_by_late_arrivals(
        current: ElementSolution, new: ElementSolution
    ) -> bool:
        return new.timetable.total_late_arrivals < current.timetable.total_late_arrivals

    return improve_tour(
        element, improvement_criterion=is_better_by_late_arrivals, size=size
    )


@log_function_call
def improve_tour_maximum_late_arrivals(
    element: SetElement, size: int = 1
) -> tuple[SetElement | None, int]:
    """Generate a tour that improves the total travel time"""

    def is_better_by_maximum_late_arrivals(
        current: ElementSolution, new: ElementSolution
    ) -> bool:
        return (
            new.timetable.maximum_late_arrivals
            < current.timetable.maximum_late_arrivals
        )

    return improve_tour(
        element, improvement_criterion=is_better_by_maximum_late_arrivals, size=size
    )


@log_function_call
def split_tour(element: SetElement, size: int = 1) -> tuple[SetElement | None, int]:
    """Split up to ``size`` tours that contain multiple groups.

    The order in which tours are considered is intentionally stochastic. Tests
    and callers that need a particular tour should control the random source.
    """
    current_element_solution = ElementSolution.from_code(element.element_id)
    current_solution = current_element_solution.the_solution
    sorted_list_of_tours: list[Tour] = list(current_solution.set_of_tours)

    if random.random() < 0.5:
        # Sort: short tours first
        sorted_list_of_tours.sort(
            key=lambda tour: len(tour.list_of_nodes), reverse=True
        )
    else:
        # Shuffle the list randomly
        random.shuffle(sorted_list_of_tours)
    actual_number_of_changes = 0
    for a_tour in islice(sorted_list_of_tours, size):
        groups: set[ChildrenGroup] = current_solution.get_groups_for_tour(
            tour_name=a_tour.the_id
        )
        if len(groups) <= 1:
            # Tour cannot be split. Move to the next group
            continue

        first_groups, second_groups = split_set_randomly(groups)
        first_groups_names = {group.the_id for group in first_groups}
        second_groups_names = {group.the_id for group in second_groups}
        first_tour_id = f'{a_tour.the_id}_a'
        second_tour_id = f'{a_tour.the_id}_b'
        group_assignment = {}
        for group_name, tour_id in current_solution.group_to_tour.assignment.items():
            if group_name in first_groups_names:
                group_assignment[group_name] = first_tour_id
            elif group_name in second_groups_names:
                group_assignment[group_name] = second_tour_id
            else:
                group_assignment[group_name] = tour_id
        bus_assignment = {}
        ic('Before ', current_solution.bus_to_tours.assignment)
        for bus_name, list_of_tours in current_solution.bus_to_tours.assignment.items():
            if a_tour.the_id in list_of_tours:
                new_list_of_tours = [
                    tour for tour in list_of_tours if tour != a_tour.the_id
                ]
                new_list_of_tours.append(first_tour_id)
                new_list_of_tours.append(second_tour_id)
                bus_assignment[bus_name] = new_list_of_tours
            else:
                bus_assignment[bus_name] = list_of_tours
        ic('After ', bus_assignment)
        first_tour = Tour(
            tour_id=first_tour_id,
            list_of_nodes=generate_tour_for_groups(groups=first_groups),
        )
        second_tour = Tour(
            tour_id=second_tour_id,
            list_of_nodes=generate_tour_for_groups(groups=second_groups),
        )
        new_set_of_tours = {
            tour
            for tour in current_solution.set_of_tours
            if tour.the_id != a_tour.the_id
        } | {first_tour, second_tour}
        new_tour_names = {tour.the_id for tour in new_set_of_tours}
        for list_of_tours in bus_assignment.values():
            for tour in list_of_tours:
                if tour not in new_tour_names:
                    raise ValueError(
                        f'Problem with bus list of tours. Tour {tour} not known. Known tours: {new_tour_names}'
                    )
        current_solution = Solution(
            set_of_groups=current_solution.set_of_groups,
            set_of_tours=new_set_of_tours,
            group_to_tour=GroupToTourAssignment(assignment=group_assignment),
            bus_to_tours=BusToToursAssignment(assignment=bus_assignment),
        )
        actual_number_of_changes += 1

    new_element_solution = ElementSolution(the_solution=current_solution)
    return new_element_solution.get_element(), actual_number_of_changes


@log_function_call
def merge_groups(element: SetElement, size: int = 1) -> tuple[SetElement | None, int]:
    current_element_solution = ElementSolution.from_code(element.element_id)
    current_solution = current_element_solution.the_solution
    sorted_list_of_tours: list[Tour] = list(current_solution.set_of_tours)

    if random.random() < 0.5:
        # Sort: short tours first
        sorted_list_of_tours.sort(key=lambda tour: len(tour.list_of_nodes))
    else:
        # Shuffle the list randomly
        random.shuffle(sorted_list_of_tours)

    tours_to_merge = sorted_list_of_tours[: size + 1]
    tour_names_to_merge = [tour.the_id for tour in tours_to_merge]
    actual_number_of_changes = len(tours_to_merge) - 1
    tours_to_keep = sorted_list_of_tours[size + 1 :]
    groups = {
        group
        for tour in tours_to_merge
        for group in current_solution.get_groups_for_tour(tour.the_id)
    }
    groups_name = {group.the_id for group in groups}
    merged_list_of_nodes = generate_tour_for_groups(groups=groups)
    merge_tour_id = current_solution.generate_unique_tour_id()
    if merge_tour_id in current_solution.tour_from_id:
        error = f'Id {merge_tour_id} is already used.'
        raise ValueError(error)

    merged_tour = Tour(tour_id=merge_tour_id, list_of_nodes=merged_list_of_nodes)
    new_set_of_tours = set(tours_to_keep) | {merged_tour}
    bus_assignment = {}
    merged_inserted = False
    for bus_name, list_of_tours in current_solution.bus_to_tours.assignment.items():
        new_tour_list = []
        for tour in list_of_tours:
            if tour in tour_names_to_merge:
                if not merged_inserted:
                    new_tour_list.append(merged_tour.the_id)
                    merged_inserted = True
                # skip additional merged tours
            else:
                new_tour_list.append(tour)
        bus_assignment[bus_name] = new_tour_list
    group_assignment = {
        group: merged_tour.the_id if group in groups_name else tour
        for group, tour in current_solution.group_to_tour.assignment.items()
    }
    ic(bus_assignment)
    new_solution = Solution(
        set_of_groups=current_solution.set_of_groups,
        set_of_tours=new_set_of_tours,
        group_to_tour=GroupToTourAssignment(assignment=group_assignment),
        bus_to_tours=BusToToursAssignment(assignment=bus_assignment),
    )
    new_element_solution = ElementSolution(the_solution=new_solution)
    return new_element_solution.get_element(), actual_number_of_changes


@log_function_call
def move_tour_to_another_bus(
    element: SetElement, size: int = 1
) -> tuple[SetElement | None, int]:
    """Move up to ``size`` tours to another bus when possible."""
    current_element_solution = ElementSolution.from_code(element.element_id)
    current_solution = current_element_solution.the_solution
    number_of_tours = len(current_solution.set_of_tours)
    if (
        len(current_solution.bus_to_tours.assignment) == 1
        or size <= 0
        or size > number_of_tours
    ):
        """There is only one bus, or the requested move is not possible."""
        return element, 0
    tours_to_change = random.sample(list(current_solution.set_of_tours), size)
    for tour in tours_to_change:
        items = list(current_solution.bus_to_tours.assignment.items())
        random.shuffle(items)
        original_bus = None
        for bus_name, the_list in items:
            if tour.the_id in the_list:
                the_list.remove(tour.the_id)
                original_bus = bus_name
                break

        for bus_name, the_list in items:
            if bus_name != original_bus and tour.the_id not in the_list:
                the_list.append(tour.the_id)
                break

        current_solution = Solution(
            set_of_tours=current_solution.set_of_tours,
            set_of_groups=current_solution.set_of_groups,
            group_to_tour=current_solution.group_to_tour,
            bus_to_tours=BusToToursAssignment(assignment=dict(items)),
        )

    new_element_solution = ElementSolution(the_solution=current_solution)
    return new_element_solution.get_element(), len(tours_to_change)
