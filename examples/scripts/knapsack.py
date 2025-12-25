"""File knapsack.py

:author: Michel Bierlaire
:date: Fri Jul  7 15:15:24 2023

This illustrates how to use the multi-objective VNS algorithm using a
simple knapsack problem.  There are two objectives: maximize utility
and minimize weight. The weight cannot go beyond capacity.

"""

from __future__ import annotations

import random

import numpy as np

from biogeme_optimization.neighborhood import Neighborhood
from biogeme_optimization.pareto import SetElement


class Sack:
    """Implements a solution. Here, a sack configuration."""

    SEPARATOR = '-'
    utility_data = None
    weight_data = None
    cost_data = None

    def __init__(self, decisions: list[int]) -> None:
        """Creates a sack from a list of decisions.

        :param decisions: decisions coded as a lisr of 0/1

        :Example:
            >>> Sack.utility_data = [10, 20, 30, 40, 50, 60]
            >>> Sack.weight_data = [1, 2, 3, 4, 5, 6]
            >>> a_sack = Sack([0, 1, 1, 0, 0, 1])
            >>> a_sack.utility
            110
            >>> a_sack.weight
            11"""
        self.decisions = decisions
        self.utility = sum(x * u for x, u in zip(self.decisions, self.utility_data))
        self.weight = sum(x * w for x, w in zip(self.decisions, self.weight_data))
        self.cost = sum(x * c for x, c in zip(self.decisions, self.cost_data))

    @classmethod
    def from_string_representation(cls, string_representation: str) -> Sack:
        """
        Creates a sack from a string representation.

        :param string_representation: the string representation of a sack using 0/1 values separated by hyphens.

        :Example:
            >>> Sack.utility_data = [10, 20, 30, 40, 50, 60]
            >>> Sack.weight_data = [1, 2, 3, 4, 5, 6]
            >>> a_sack = Sack.from_string_representation('0-1-1-0-0-1')
            >>> a_sack.utility
            110
            >>> a_sack.weight
            11
        """
        decisions = [int(i) for i in string_representation.split(cls.SEPARATOR)]
        return Sack(decisions=decisions)

    def generate_string_representation(self) -> str:
        """Provide a string ID for the sack

        :return: identifier of the solution. Used to organize the Pareto set.
        :rtype: str
        """
        return self.SEPARATOR.join([str(x) for x in self.decisions])

    def get_element(self) -> SetElement:
        """Implementation of abstract method"""
        return SetElement(
            self.generate_string_representation(), [-self.utility, self.cost]
        )

    def describe(self):
        """Short description of the solution. Used for reporting.

        :return: short description of the solution.
        :rtype: str
        """
        return f'{self.generate_string_representation()}: U={self.utility} C={self.cost} W={self.weight}'


# Operators
def add_items(element: SetElement, size: int = 1) -> tuple[SetElement | None, int]:
    """Add ``size`` items in the sack

    :param element: representation of the current sack
    :param size: number of items to add into the sack
    :return: representation of the new sack, and number of changes actually made
    """
    solution = Sack.from_string_representation(element.element_id)
    absent = [i for i, x in enumerate(solution.decisions) if x == 0]
    if not absent:
        return None, 0
    random.shuffle(absent)
    n = min(len(absent), size)
    x_plus = solution.decisions.copy()
    for i in absent[:n]:
        x_plus[i] = 1
    neighbor = Sack(x_plus)
    return neighbor.get_element(), n


def remove_items(element: SetElement, size: int = 1) -> tuple[SetElement | None, int]:
    """Remove ``size`` items from the sack

    :param element: representation of the current sack
    :param size: number of items to remove from the sack

    :return: representation of the new sack, and number of changes actually made
    """
    solution = Sack.from_string_representation(element.element_id)
    present = [i for i, x in enumerate(solution.decisions) if x == 1]
    if not present:
        return None, 0
    random.shuffle(present)
    n = min(len(present), size)
    x_plus = solution.decisions.copy()
    for i in present[:n]:
        x_plus[i] = 0
    neighbor = Sack(x_plus)
    return neighbor.get_element(), n


def change_decisions(
    element: SetElement, size: int = 1
) -> tuple[SetElement | None, int]:
    """Change the status of ``size`` items in the sack.

    :param element: representation of the current sack
    :param size: number of items to modify
    :return: representation of the new sack, and number of changes actually made

    """
    solution = Sack.from_string_representation(element.element_id)
    length = len(solution.decisions)
    n = min(length, size)
    order = np.random.permutation(length)
    x_plus = solution.decisions.copy()
    for i in range(n):
        x_plus[order[i]] = 1 - x_plus[order[i]]
    neighbor = Sack(x_plus)
    return neighbor.get_element(), n


class Knapsack(Neighborhood):
    """Class characterizing the knapsack problem. Note the
    inheritance from the abstract class Neighborhood. It guarantees
    the compliance with the requirements of the algorithm.

    """

    def __init__(self, utility: list[float], weight: list[float], capacity: float):
        """Ctor"""
        self.utility = utility
        self.weight = weight
        self.capacity = capacity
        self.operators = {
            'Add items': add_items,
            'Remove items': remove_items,
            'Change decision for items': change_decisions,
        }
        self.last_operator = None
        super().__init__(self.operators)

    def is_valid(self, element: SetElement) -> tuple[bool, str | None]:
        """Check if the sack verifies the capacity constraint

        Implementation of the abstract method

        :param element: representation of the sack to check
        :return: True if the capacity constraint is verified. The string provides a message if the solution is not valid
        """
        solution = Sack.from_string_representation(element.element_id)
        if solution.weight <= self.capacity:
            return True, None
        return False, 'Infeasible sack'
