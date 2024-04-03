"""Linear optimization problem in standard form.

Michel Bierlaire
Mon Apr 1 14:34:53 2024
"""

import numpy as np

from .linear_constraints.standard_form import StandardForm


class LinearOptimization:
    """Class defining a linear optimization problem in standard form."""

    def __init__(
        self,
        objective: np.ndarray,
        constraint_matrix: np.ndarray,
        right_hand_side: np.ndarray,
    ) -> None:
        """
        Constructor. The problem is assumed to be written in standard form.

        :param objective: vector of coefficients of the objective function.
        :param constraint_matrix: matrix of coefficients of the constraints.
        :param right_hand_side: right hand side of the constraints.
        """
        if (
            not isinstance(objective, np.ndarray)
            or not isinstance(constraint_matrix, np.ndarray)
            or not isinstance(right_hand_side, np.ndarray)
        ):
            error_msg = (
                'Numpy arrays are requested here. An array of float an be transformed '
                'into a numpy array as follows: np.array([1.0, 2.0, 3.0])'
            )
            raise AttributeError(error_msg)
        if len(objective) != constraint_matrix.shape[1]:
            error_msg = (
                f'Incompatible sizes: {len(objective)} and {constraint_matrix.shape[1]}'
            )
            raise ValueError(error_msg)
        self.objective = objective
        self.constraints = StandardForm(
            matrix=constraint_matrix, vector=right_hand_side
        )

    @property
    def n_variables(self) -> int:
        return self.constraints.n_variables

    @property
    def n_constraints(self) -> int:
        return self.constraints.n_constraints

    @property
    def non_basic_indices(self) -> list[int]:
        return self.constraints.non_basic_indices

    @property
    def basic_indices(self) -> list[int]:
        """

        :return: list of basic indices.
        """
        return self.constraints.basic_indices

    @basic_indices.setter
    def basic_indices(self, indices: list[int]) -> None:
        """
        Sets the list after verification.
        :param indices: list of basic indices
        """
        self.constraints.basic_indices = indices

    @property
    def is_basic_solution_feasible(self) -> bool:
        """Checks feasibility"""
        return self.constraints.is_basic_solution_feasible

    @property
    def basic_solution(self) -> np.ndarray | None:
        """Basic solution in the full dimension"""
        return self.constraints.basic_solution

    @property
    def basic_part_basic_solution(self) -> np.ndarray | None:
        """
        Calculates the basic part of the solution.
        """
        return self.constraints.basic_part_basic_solution

    def basic_part_basic_direction(self, non_basic_index: int) -> np.ndarray | None:
        """
        Calculates the basic part of a basic direction.

        :param non_basic_index: index of the non-basic variable.
        :return: basic part of the basic direction, if it exists.
        """
        return self.constraints.basic_part_basic_direction(
            non_basic_index=non_basic_index
        )

    def basic_direction(self, non_basic_index: int) -> np.ndarray | None:
        """
        Calculates a basic direction.

        :param non_basic_index: index of the non-basic variable.
        :return: basic direction, if it exists.
        """
        return self.constraints.basic_direction(non_basic_index=non_basic_index)

    @property
    def value_objective_at_vertex(self) -> float | None:
        """Value of the objective function at the vertex identified by the basic indices, if it exists."""
        if self.basic_solution is None:
            return None
        return self.objective_function(solution=self.basic_solution)

    def objective_function(self, solution: np.ndarray) -> float:
        """Calculate the value of the objective function

        :param solution: solution where to evaluate
        :return: value of the objective function
        """
        if solution is None:
            error_msg = 'A solution must be provided, not None.'
            raise ValueError(error_msg)

        if len(solution) != self.n_variables:
            error_msg = (
                f'Wrong dimension: {len(solution)} instead of {self.n_variables}'
            )
            raise ValueError(error_msg)

        return np.inner(self.objective, solution)

    def reduced_cost(self, variable_index: int) -> float:
        """Calculate the reduced cost for a non basic variable"""
        if self.basic_indices is None:
            error_msg = 'Basic indices have not been selected'
            raise EnvironmentError(error_msg)

        # The reduced cost of basic variables is zero.
        if variable_index in self.basic_indices:
            return 0

        direction = self.constraints.basic_part_basic_direction(
            non_basic_index=variable_index
        )
        basic_costs = self.objective[self.basic_indices]

        return self.objective[variable_index] + np.inner(direction, basic_costs)
