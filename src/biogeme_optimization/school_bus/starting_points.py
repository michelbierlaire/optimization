import random

from .data_classes import SchoolBusProblem
from .decision_variables import (
    BusToToursAssignment,
    ChildrenGroup,
    GroupToTourAssignment,
    Solution,
    Tour,
    feasible_tours,
)


def generate_tour_for_groups(groups: set[ChildrenGroup]) -> list[str]:
    tour_generator = feasible_tours(groups=groups)
    try:
        the_tour_nodes: list[str] = next(tour_generator)
    except StopIteration:
        error_msg = f'No feasible tour could be found for groups {[group.the_id for group in groups]}'
    return the_tour_nodes


def single_bus(the_problem: SchoolBusProblem) -> Solution:
    """Generate a feasible solution by organizing all transfers with a single bus"""

    tour_id = 'single_tour'

    set_of_groups = {
        ChildrenGroup(
            group_id=f'{origin}_to_{destination}',
            origin_name=origin,
            destination_name=destination,
            size=flow,
        )
        for (origin, destination), flow in the_problem.od_table.items()
    }

    the_tour_nodes = generate_tour_for_groups(groups=set_of_groups)
    the_tour = Tour(tour_id=tour_id, list_of_nodes=the_tour_nodes)
    # We take the bus with the largest capacity
    the_bus = max(the_problem.buses, key=lambda bus: bus.capacity)
    print(f'Bus with the largest capacity: {the_bus.name}')
    bus_assignment = {bus.name: [] for bus in the_problem.buses}
    bus_assignment[the_bus.name] = [tour_id]
    tours_per_bus = BusToToursAssignment(assignment=bus_assignment)
    group_to_tours = GroupToTourAssignment(
        assignment={
            f'{origin}_to_{destination}': tour_id
            for (origin, destination) in the_problem.od_table.keys()
        }
    )

    the_solution = Solution(
        set_of_groups=set_of_groups,
        set_of_tours={the_tour},
        bus_to_tours=tours_per_bus,
        group_to_tour=group_to_tours,
    )
    return the_solution


def taxi_solution(the_problem: SchoolBusProblem):
    set_of_groups = {
        ChildrenGroup(
            group_id=f'{origin}_to_{destination}',
            origin_name=origin,
            destination_name=destination,
            size=flow,
        )
        for (origin, destination), flow in the_problem.od_table.items()
    }
    set_of_tours = {
        Tour(tour_id=f'{origin}_to_{destination}', list_of_nodes=[origin, destination])
        for (origin, destination) in the_problem.od_table.keys()
    }
    group_assignment = {
        f'{origin}_to_{destination}': f'{origin}_to_{destination}'
        for (origin, destination) in the_problem.od_table.keys()
    }
    # Step 1: Randomly assign one unique tour to each bus to ensure every bus gets one
    list_of_buses = list(the_problem.buses)
    list_of_tours = [tour.the_id for tour in set_of_tours]
    random.shuffle(list_of_tours)

    bus_assignment = {
        bus.name: [] for bus in list_of_buses
    }  # Ensure all buses are included

    # Step 1: Assign one tour to each bus (up to the number of tours)
    for bus, tour in zip(list_of_buses, list_of_tours):
        bus_assignment[bus.name].append(tour)

    # Step 2: Assign remaining tours randomly to buses
    remaining_tours = list_of_tours[len(list_of_buses) :]
    for tour in remaining_tours:
        chosen_bus = random.choice(list_of_buses)
        bus_assignment[chosen_bus.name].append(tour)

    the_solution = Solution(
        set_of_groups=set_of_groups,
        set_of_tours=set_of_tours,
        group_to_tour=GroupToTourAssignment(assignment=group_assignment),
        bus_to_tours=BusToToursAssignment(assignment=bus_assignment),
    )

    return the_solution
