from biogeme_optimization.neighborhood import Neighborhood
from biogeme_optimization.pareto import SetElement
from .data_classes import SchoolBusProblem
from .decision_variables import Solution
from .operators import (
    improve_tour_early_arrival,
    improve_tour_late_arrivals,
    improve_tour_maximum_early_arrivals,
    improve_tour_maximum_late_arrivals,
    improve_tour_maximum_travel_time,
    improve_tour_travel_time,
    merge_groups,
    move_tour_to_another_bus,
    split_tour,
)
from .sanity_check import SanityCheck


class SchoolBus(Neighborhood):
    """Class characterizing the school bus optimization problem. Note the
    inheritance from the abstract class Neighborhood. It guarantees
    the compliance with the requirements of the algorithm.

    """

    def __init__(self, the_problem: SchoolBusProblem):
        """Ctor"""
        self.the_problem = the_problem
        self.operators = {
            'Move tour': move_tour_to_another_bus,
            'Improve travel time': improve_tour_travel_time,
            'Improve maximum travel time': improve_tour_maximum_travel_time,
            'Improve early arrivals': improve_tour_early_arrival,
            'Improve maximum early arrivals': improve_tour_maximum_early_arrivals,
            'Improve late arrivals': improve_tour_late_arrivals,
            'Improve maximum late arrivals': improve_tour_maximum_late_arrivals,
            'Split tour': split_tour,
            'Merge groups': merge_groups,
        }
        self.last_operator = None
        super().__init__(self.operators)

    def is_valid(self, element: SetElement) -> tuple[bool, str | None]:
        """Check if the sack verifies the capacity constraint

        Implementation of the abstract method

        :param element: representation of the sack to check
        :return: True if the capacity constraint is verified. The string provides a message if the solution is not valid
        """
        solution = Solution.from_code(element.element_id)
        sanity_check = SanityCheck(the_problem=self.the_problem, the_solution=solution)
        return sanity_check.all_checks()
