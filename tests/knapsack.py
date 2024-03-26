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
from biogeme_optimization.neighborhood import Neighborhood, Operator, OperatorOutput
from biogeme_optimization.pareto import SetElement


class Sack:
    """Implements a solution. Here, a sack configuration."""

    SEPARATOR: str = '-'
    utility_data: list[float | None] = None
    weight_data: list[float | None] = None

    def __init__(self, string_representation: str):
        """Creates a sack from a string representation"""
        self.decisions: list[int] = [
            int(i) for i in string_representation.split(self.SEPARATOR)
        ]
        self.utility: float = sum(
            x * u for x, u in zip(self.decisions, self.utility_data)
        )
        self.weight: float = sum(
            x * u for x, u in zip(self.decisions, self.weight_data)
        )

    @classmethod
    def from_decisions(cls, decisions: list[int]):
        """Creates a sack from the actual decisions"""
        the_string = cls.code_decisions(decisions)
        return cls(the_string)

    def __eq__(self, other: Sack) -> bool:
        return self.decisions == other.decisions

    def __str__(self) -> str:
        return self.describe()

    def __repr__(self) -> str:
        return self.code_id()

    @classmethod
    def empty_sack(cls, size: int):
        """
        :return: empty sack of the same dimension.
        :rtype: class Sack
        """
        element_id = cls.SEPARATOR.join(['0'] * size)
        return cls(element_id)

    def get_element(self) -> SetElement:
        """Implementation of abstract method"""
        return SetElement(self.code_id(), [-self.utility, self.weight])

    def describe(self) -> str:
        """Short description of the solution. Used for reporting.

        :return: short description of the solution.
        :rtype: str
        """
        return f'{self.code_id()}: U={self.utility} W={self.weight}'

    def code_id(self) -> str:
        """Provide a string ID for the sack

        :return: identifier of the solution. Used to organize the Pareto set.
        :rtype: str
        """
        return self.code_decisions(self.decisions)

    @classmethod
    def code_decisions(cls, decisions: list[int]) -> str:
        """Provide a string ID for the sack given the decisions

        :return: identifier of the solution. Used to organize the Pareto set.
        :rtype: str
        """
        return cls.SEPARATOR.join([str(x) for x in decisions])


# Operators
def add_items(element: SetElement, size: int = 1) -> OperatorOutput:
    """Add ``size`` items in the sack

    :param element: ID of the current sack
    :type element: class SetElement

    :param size: number of items to add into the sack
    :type size: int

    :return: Id of the new sack, and number of changes actually made
    :rtype: tuple(class SetElement, int)
    """
    solution = Sack(element.element_id)
    absent = [i for i, x in enumerate(solution.decisions) if x == 0]
    if not absent:
        return None, 0
    random.shuffle(absent)
    n = min(len(absent), size)
    xplus = solution.decisions.copy()
    for i in absent[:n]:
        xplus[i] = 1
    neighbor = Sack.from_decisions(xplus)
    return neighbor.get_element(), n


def remove_items(element: SetElement, size: int = 1) -> OperatorOutput:
    """Remove ``size`` items from the sack

    :param element: current sack
    :type element: class Sack

    :param size: number of items to remove from the sack
    :type size: int

    :return: Id of the new sack, and number of changes actually made
    :rtype: tuple(class SetElement, int)
    """
    solution = Sack(element.element_id)
    present = [i for i, x in enumerate(solution.decisions) if x == 1]
    if not present:
        return None, 0
    random.shuffle(present)
    n = min(len(present), size)
    xplus = solution.decisions.copy()
    for i in present[:n]:
        xplus[i] = 0
    neighbor = Sack.from_decisions(xplus)
    return neighbor.get_element(), n


def change_decisions(element: SetElement, size: int = 1) -> OperatorOutput:
    """Change the status of ``size`` items in the sack.

    :param element: representation of the current sack
    :type element: class SetElement

    :param size: number of items to modify
    :type size: int

    :return: representation of the new sack, and number of changes actually made
    :rtype: tuple(class SetElement, int)

    """
    solution = Sack(element.element_id)
    length = len(solution.decisions)
    n = min(length, size)
    order = np.random.permutation(length)
    xplus = solution.decisions.copy()
    for i in range(n):
        xplus[order[i]] = 1 - xplus[order[i]]
    neighbor = Sack.from_decisions(xplus)
    return neighbor.get_element(), n


class Knapsack(Neighborhood):
    """Class characterizing the knapsack problem. Note the
    inheritance from the abstract class Neighborhood. It guarantees
    the compliance with the requirements of the algorithm.

    """

    def __init__(self, utility: list[float], weight: list[float], capacity: float):
        """Ctor"""
        self.utility: list[float] = utility
        self.weight: list[float] = weight
        self.capacity: float = capacity
        self.operators: dict[str, Operator] = {
            'Add items': add_items,
            'Remove items': remove_items,
            'Change decision for items': change_decisions,
        }
        self.last_operator: Operator | None = None
        super().__init__(self.operators)

    def is_valid(self, element: SetElement) -> tuple[bool, str]:
        """Check if the sack verifies the capacity constraint

        Implementation of the abstract method

        :param element: representation of the sack to check
        :type element: class SetElement

        :return: True if the capacity constraint is verified
        :rtype: bool
        """
        solution = Sack(element.element_id)
        if solution.weight <= self.capacity:
            return True, ''
        return False, 'Infeasible sack'
