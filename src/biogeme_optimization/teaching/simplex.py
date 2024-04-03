"""Simplex algorithm: algebraic form

Michel Bierlaire
Mon Apr 1 14:34:53 2024
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from .linear_optimization import LinearOptimization


class CauseInterruptionIterations(Enum):
    OPTIMAL = auto()
    UNBOUNDED = auto()
    INFEASIBLE = auto()

    def __str__(self) -> str:
        messages = {
            self.OPTIMAL: 'Optimal basis found.',
            self.UNBOUNDED: 'Optimization problem is unbounded.',
            self.INFEASIBLE: 'Optimization problem is infeasible.',
        }
        return messages[self]


@dataclass
class IterationReport:
    iteration_number: int
    iterate: np.ndarray
    entering_variable: int | None
    exiting_variable: int | None
    direction: np.ndarray | None
    step: float | None

    def __str__(self) -> str:
        iterate = [f'{value: 8.3g}' for value in self.iterate]
        return f'{self.iteration_number:4} ({", ".join(iterate)})'


def longest_step(
    current_point: np.ndarray, direction: np.ndarray
) -> tuple[float, int | None]:
    """
    Function that calculates the longest step along a direction so that all entries stay non-negative.
    :param current_point: current point.
    :param direction: direction.
    :return: the step, as well as the index of one variable that becomes zero, or None of the direction is unbounded.
    """
    alpha = np.array(
        [-x / d if d < 0 else np.infty for x, d in zip(current_point, direction)]
    )
    min_value = np.min(alpha)
    if min_value == np.infty:
        # The problem is unbounded.
        return np.infty, None
    min_index = np.argmin(alpha)
    return min_value, min_index


class SimplexAlgorithm:
    def __init__(
        self, optimization_problem: LinearOptimization, initial_basis: list[int]
    ) -> None:
        self.current_iteration = 0
        self.problem: LinearOptimization = optimization_problem
        self.problem.basic_indices = initial_basis
        self.stopping_cause: CauseInterruptionIterations | None = None
        self.direction: np.ndarray | None = None
        self.step: float | None = None
        self.entering_index: int | None = None
        self.exiting_index: int | None = None
        self.finished = False

        # Verify that the initial basis exists and is feasible.
        if self.problem.basic_solution is None:
            raise ValueError('Invalid starting basis')
        if not self.problem.is_basic_solution_feasible:
            raise ValueError(
                f'Infeasible starting basic solution {self.problem.basic_solution}'
            )

    @property
    def solution(self) -> np.ndarray:
        return self.problem.basic_solution

    def __iter__(self) -> SimplexAlgorithm:
        return self

    def index_entering_basis(self) -> int | None:
        """
        Function that identifies a non-basic index to enter the basis, or detect optimality
        :return: a non-basic index corresponding to a negative reduced cost, or None if optimal.
        """
        for non_basic_index in self.problem.non_basic_indices:
            reduced_cost = self.problem.reduced_cost(variable_index=non_basic_index)
            if reduced_cost < 0:
                return non_basic_index
        # No negative cost has been found. The solution is optimal.
        return None

    def index_leaving_basis(
        self,
        index_entering: int,
    ) -> int | None:
        """function that identifies a basic index to leave the basis, or identify an unbounded problem.

        :param index_entering: non-basic index entering the basis
        :return: index of the variable leaving the basis, or None if unbounded
        """
        basic_part_basic_solution = self.problem.basic_part_basic_solution
        if basic_part_basic_solution is None:
            raise ValueError(
                f'Invalid list of basic indices {self.problem.basic_indices}'
            )
        basic_part_basic_direction = self.problem.basic_part_basic_direction(
            non_basic_index=index_entering
        )
        # Calculate the steps alpha
        alpha, index = longest_step(
            current_point=basic_part_basic_solution,
            direction=basic_part_basic_direction,
        )
        # For reporting only
        self.direction = self.problem.basic_direction(non_basic_index=index_entering)
        self.step = alpha

        if index is None:
            # The problem is unbounded along the direction.
            return None
        return self.problem.basic_indices[index]

    def __next__(self) -> IterationReport:
        if self.finished:
            raise StopIteration
        self.entering_index = self.index_entering_basis()
        if self.entering_index is None:
            # Optimal solution found.
            self.stopping_cause = CauseInterruptionIterations.OPTIMAL
            self.finished = True
            report = IterationReport(
                iteration_number=self.current_iteration,
                iterate=self.problem.basic_solution,
                entering_variable=None,
                exiting_variable=None,
                direction=None,
                step=None,
            )
            return report

        self.exiting_index = self.index_leaving_basis(
            index_entering=self.entering_index,
        )
        if self.exiting_index is None:
            # Problem is unbounded.
            self.stopping_cause = CauseInterruptionIterations.UNBOUNDED
            self.finished = True
            report = IterationReport(
                iteration_number=self.current_iteration,
                iterate=self.problem.basic_solution,
                entering_variable=self.entering_index,
                exiting_variable=None,
                direction=None,
                step=None,
            )
            return report

        new_basis = [
            self.entering_index if x == self.exiting_index else x
            for x in self.problem.basic_indices
        ]
        report = IterationReport(
            iteration_number=self.current_iteration,
            iterate=self.problem.basic_solution,
            entering_variable=self.entering_index,
            exiting_variable=self.exiting_index,
            direction=self.direction,
            step=self.step,
        )
        self.problem.basic_indices = new_basis
        self.current_iteration += 1
        return report
