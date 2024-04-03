"""Simplex tableau

Michel Bierlaire
Wed Apr 3 09:21:36 2024
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RowColumn:
    """Data structure that stores the row and column of an element of the tableau"""

    row: int
    column: int


class SimplexTableau:
    """Class managing the simplex tableau"""

    def __init__(self, tableau: np.ndarray) -> None:
        self.tableau: np.ndarray = tableau
        self.n_rows, self.n_columns = tableau.shape
        self.n_variables = self.n_columns - 1
        self.n_constraints = self.n_rows - 1
        self.identity_ones = self.identify_basic_variables()
        if len(self.identity_ones) != self.n_constraints:
            error_msg = (
                f'A valid tableau must contain {self.n_constraints} columns of the identity matrix, '
                f'not {len(self.identity_ones)}'
            )
            raise ValueError(error_msg)
        self.basic_indices = [element.column for element in self.identity_ones]
        self.non_basic_indices = list(
            set(range(self.n_variables)) - set(self.basic_indices)
        )

    @classmethod
    def from_basis(
        cls,
        matrix: np.ndarray,
        right_hand_side: np.ndarray,
        objective: np.ndarray,
        basic_indices: list[int],
    ) -> SimplexTableau:
        """Function that builds the tableau using its definition
        :param matrix: constraint matrix (standard form)
        :param right_hand_side: right hand side
        :param objective: coefficients of the objective function
        :param basic_indices: indices of the basic variables
        :return: simplex tableau
        """
        n_constraints, n_variables = matrix.shape

        if len(right_hand_side) != n_constraints:
            error_msg = (
                f'Inconsistent dimensions {len(right_hand_side)} and {n_constraints}'
            )
            raise ValueError(error_msg)

        if len(objective) != n_variables:
            error_msg = f'Inconsistent dimensions {len(objective)} and {n_variables}'
            raise ValueError(error_msg)

        if len(basic_indices) != n_constraints:
            error_msg = (
                f'Inconsistent dimensions {len(basic_indices)} and {n_constraints}'
            )
            raise ValueError(error_msg)

        wrong_indices = [
            index for index in basic_indices if index < 0 or index >= n_variables
        ]
        if wrong_indices:
            error_msg = f'Wrong basic indices: {wrong_indices}'
            raise ValueError(error_msg)

        basic_matrix = matrix[:, basic_indices]
        basic_costs = objective[basic_indices]
        upper_left = np.linalg.solve(basic_matrix, matrix)
        upper_right = np.linalg.solve(basic_matrix, right_hand_side)
        lower_left = objective - basic_costs @ upper_left
        lower_right = -np.inner(basic_costs, upper_right)
        tableau = np.empty((n_constraints + 1, n_variables + 1))
        tableau[:n_constraints, :n_variables] = upper_left
        tableau[:n_constraints, n_variables] = upper_right
        tableau[n_constraints, :n_variables] = lower_left
        tableau[n_constraints, n_variables] = lower_right
        return SimplexTableau(tableau=tableau)

    def __str__(self) -> str:
        return str(self.tableau)

    def report(self) -> str:
        """Provides a report about the solution associated with the tableau"""

        the_report = []
        the_report.append(
            f'{self.n_variables} variables, {self.n_constraints} constraints.'
        )
        variables_in_the_basis: list[RowColumn] = self.identify_basic_variables()
        for element in variables_in_the_basis:
            the_report.append(
                f'Basic variable {element.column} corresponds to row {element.row}.'
            )

        # The corresponding feasible solution.
        the_report.append(f'Feasible basic solution: {self.feasible_basic_solution}.')

        # Value of the objective function
        the_report.append(f'Objective function: {self.value_objective_function}.')

        # Reduced costs
        reduced_costs: dict[int, float] = self.reduced_costs
        the_report.append(f'Reduced costs of non basic variables: {reduced_costs}.')

        # Basic directions
        basic_directions: dict[int, np.ndarray] = self.basic_directions
        the_report.append(
            f'Basic directions of non basic variables: {basic_directions}.'
        )
        return '\n'.join(the_report)

    def identify_basic_variables(self) -> list[RowColumn]:
        """Function that identifies the column associated with the basic variables.

        :return: a list reporting the position of the 1's in the tableau corresponding to the basic variable.
        """

        # We need to identify where the columns of the identify matrix are, and where the ones in those columns are
        # located.
        ones = []
        for col_index in range(self.n_columns):
            column = self.tableau[:, col_index]
            # Check if there's exactly one entry with 1 and the rest are 0
            if np.sum(column == 1) == 1 and np.sum(column == 0) == self.n_rows - 1:
                # Find the index of the 1
                row_index = int(np.where(column == 1)[0][0])
                ones.append(RowColumn(row_index, col_index))
        return ones

    @property
    def feasible_basic_solution(self) -> np.ndarray:
        """Function that constructs the feasible basic solution corresponding to the tableau.

        :return: feasible basic solution
        :rtype: np.ndarray
        """
        result = np.zeros(self.n_variables)
        for element in self.identity_ones:
            result[element.column] = self.tableau[element.row, self.n_variables]
        return result

    @property
    def value_objective_function(self) -> float:
        """
        It is simply the opposite of the lower right cell of the tableau.

        :return: value of the objective function.
        :rtype: float
        """
        return -self.tableau[-1, -1]

    @property
    def reduced_costs(self) -> dict[int, float]:
        """
        Function that calculates the reduced costs of the non-basic variables.
        :return: a dict mapping the indices of the non-basic variables and the reduced costs.
        :rtype: dict[int, float]
        """
        result = {
            index: float(self.tableau[-1, index]) for index in self.non_basic_indices
        }
        return result

    @property
    def basic_directions(self) -> dict[int, np.ndarray]:
        """
        Each column of non-basic variable corresponds to the opposite of the basic part of the basic directions.
        :return: dict mapping the non-basic variables with the basic directions.
        :rtype: dict[int, np.ndarray]
        """
        result = dict()
        for index in self.non_basic_indices:
            basic_part_direction = -self.tableau[: self.n_rows, index]
            basic_direction = np.zeros(self.n_variables)
            basic_direction[index] = 1.0
            for element in self.identity_ones:
                basic_direction[element.column] = basic_part_direction[element.row]
            result[index] = basic_direction
        return result

    def pivoting(self, pivot: RowColumn):
        """Perform the pivoting of the tableau.

        :param pivot: position of the pivot
        """
        if pivot.row < 0 or pivot.row >= self.n_rows:
            error_msg = f'Row index {pivot.row} out of range [0, {self.n_rows-1}]'
            raise ValueError(error_msg)
        if pivot.column < 0 or pivot.column >= self.n_columns:
            error_msg = (
                f'Column index {pivot.column} out of range [0, {self.n_columns-1}]'
            )
            raise ValueError(error_msg)
        pivot_value = self.tableau[pivot.row, pivot.column]
        if np.isclose(pivot_value, 0.0):
            error_msg = f'Pivot is numerically too close to zero: {pivot_value}'
            raise ValueError(error_msg)
        row_of_the_pivot = self.tableau[pivot.row, :]
        for row in range(self.n_rows):
            if row != pivot.row:
                multiplier = -self.tableau[row, pivot.column] / pivot_value
                self.tableau[row, :] += multiplier * row_of_the_pivot
        self.tableau[pivot.row, :] /= pivot_value
