"""Implementation of the school bus routing and scheduling problem inspired by Spada et al. (2005)

Classes interfacing the Pareto set

Spada M., Bierlaire M., Liebling T. (2005). Decision-aid methodology
for the school bus routing and scheduling problem. Transportation Science 39 (4):477-490
https://dx.doi.org/10.1287/trsc.1040.0096

Michel Bierlaire
Wed Jun 04 2025, 17:56:16
"""

from __future__ import annotations

from biogeme_optimization.pareto import SetElement
from .data_classes import SchoolBusProblem
from .decision_variables import Solution
from .indicators import TimeTable
from .sanity_check import SanityCheck


class ElementSolution:
    the_problem: SchoolBusProblem = None

    def __init__(self, the_solution: Solution):
        if self.the_problem is None:
            raise ValueError(f'The problem has not been defined')
        self.the_solution = the_solution
        self.the_solution.update_buses({bus.name for bus in self.the_problem.buses})
        self.timetable = TimeTable(
            the_problem=self.the_problem, the_solution=the_solution
        )
        _ = SanityCheck(the_problem=self.the_problem, the_solution=the_solution)

    @classmethod
    def from_code(cls, the_code: str) -> ElementSolution:
        the_solution = Solution.from_code(the_code=the_code)
        return ElementSolution(the_solution=the_solution)

    @classmethod
    def from_element(cls, the_element: SetElement) -> ElementSolution:
        the_solution = Solution.from_code(the_code=the_element.element_id)
        return ElementSolution(the_solution=the_solution)

    def get_element(self) -> SetElement:
        """Implementation of abstract method"""
        return SetElement(
            self.the_solution.generate_code(),
            [
                self.timetable.total_travel_time,
                self.timetable.total_late_arrivals,
                self.timetable.total_early_arrivals,
                self.timetable.maximum_travel_time,
                self.timetable.maximum_late_arrivals,
                self.timetable.maximum_early_arrivals,
            ],
        )
