"""Implementation of the school bus routing and scheduling problem inspired by Spada et al. (2005)

Functions verifying the feasibility of a solution

Spada M., Bierlaire M., Liebling T. (2005). Decision-aid methodology
for the school bus routing and scheduling problem. Transportation Science 39 (4):477-490
https://dx.doi.org/10.1287/trsc.1040.0096

Michel Bierlaire
Mon May 26 2025, 10:38:44
"""

from decision_variables import ChildrenGroup, Tour


def is_tour_feasible(tour: Tour, groups: set[ChildrenGroup]) -> tuple[bool, str | None]:
    node_position = {node: idx for idx, node in enumerate(tour.list_of_nodes)}
    for group in groups:
        if group.origin_name not in node_position:
            raise ValueError(
                f'Tour {tour.the_id} does not involve origin {group.origin_name}'
            )
        if group.destination_name not in node_position:
            raise ValueError(
                f'Tour {tour.the_id} does not involve destination {group.destination_name}'
            )
        if node_position[group.origin_name] >= node_position[group.destination_name]:
            msg = f'Tour {tour.the_id} is infeasible for group {group.the_id} as origin {group.origin_name} is reached after destination {group.destination_name}'
            return False, msg
    return True, None
