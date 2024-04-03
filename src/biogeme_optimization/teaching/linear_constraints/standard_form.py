"""Polyhedron in standard form.

Michel Bierlaire
Sat Mar 30 17:20:42 2024
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


class StandardForm:
    """Constraints of a linear optimization problem in standard form: $Ax = b$, $x  \geq 0$."""

    def __init__(self, matrix: np.ndarray, vector: np.ndarray) -> None:
        """
        Constructor
        :param matrix: matrix $A$.
        :param vector: vector $b$.
        """
        self.matrix: np.ndarray = matrix
        self.vector: np.ndarray = vector
        self.n_constraints, self.n_variables = matrix.shape
        self._basic_indices: list[int] | None = None

        if self.n_constraints != len(self.vector):
            raise ValueError(
                f'The length of the right-hand side [{len(self.vector)}] must match '
                f'the row dimension of the matrix [{self.n_constraints}].'
            )

    @property
    def basic_indices(self) -> list[int] | None:
        """

        :return: list of basic indices.
        """
        return self._basic_indices

    @basic_indices.setter
    def basic_indices(self, indices: list[int]) -> None:
        """
        Sets the list after verification.
        :param indices: list of basic indices
        """
        if len(indices) != self.n_constraints:
            error_msg = (
                f'The number of basic indices [{len(indices)}] must be equal '
                f'to the number of constraints [{self.n_constraints}].'
            )
            raise ValueError(error_msg)

        # Verify that each index is a valid column index of the matrix.
        max_index = self.n_variables - 1
        if not all(0 <= index <= max_index for index in indices):
            raise ValueError(
                f'One or more indices are out of the valid column index '
                f'range of the matrix: {indices}.'
            )
        self._basic_indices = indices

    @property
    def non_basic_indices(self) -> list[int]:
        return list(set(range(self.n_variables)) - set(self.basic_indices))

    @property
    def basic_matrix(self) -> np.ndarray | None:
        """Extract basic matrix."""
        if self.basic_indices is None:
            return None
        return self.matrix[:, self.basic_indices]

    @property
    def basic_part_basic_solution(self) -> np.ndarray | None:
        """
        Calculates the basic part of the solution.
        """
        if self.basic_matrix is None:
            return None
        try:
            basic_variables = np.linalg.solve(self.basic_matrix, self.vector)
            return basic_variables
        except np.linalg.LinAlgError:
            # The system does not have a solution ####
            return None

    @property
    def is_basic_solution_feasible(self) -> bool:
        """Checks feasibility"""
        if self.basic_part_basic_solution is None:
            return False
        return np.all(self.basic_part_basic_solution >= 0)

    @property
    def basic_solution(self) -> np.ndarray | None:
        """Basic solution in the full dimension"""
        if self.basic_part_basic_solution is None:
            return None
        complete_solution = np.zeros(self.n_variables)

        # Assign the values of basic variables to their corresponding positions
        for variable, index in zip(self.basic_part_basic_solution, self.basic_indices):
            complete_solution[index] = variable
        return complete_solution

    def basic_part_basic_direction(self, non_basic_index: int) -> np.ndarray | None:
        """
        Calculates the basic part of a basic direction.

        :param non_basic_index: index of the non-basic variable.
        :return: basic part of the basic direction, if it exists.
        """
        if self.basic_indices is None:
            logger.warning('No basis has been selected.')
            return None

        if non_basic_index < 0 or non_basic_index >= self.n_variables:
            error_msg = f'Invalid index {non_basic_index}. Must be between 0 and {self.n_variables-1}'
            raise ValueError(error_msg)
        if non_basic_index in self.basic_indices:
            error_msg = f'Index {non_basic_index} is a basic index.'
            raise ValueError(error_msg)

        non_basic_column = self.matrix[:, non_basic_index]

        # Solve the linear system and return the solution, if it exists.
        try:
            the_basic_direction = np.linalg.solve(self.basic_matrix, -non_basic_column)
        except np.linalg.LinAlgError:
            # The system does not have a solution ####
            return None
        return the_basic_direction

    def basic_direction(self, non_basic_index: int) -> np.ndarray | None:
        """
        Calculates a basic direction.

        :param non_basic_index: index of the non-basic variable.
        :return: basic direction, if it exists.
        """
        the_basic_direction = self.basic_part_basic_direction(
            non_basic_index=non_basic_index
        )
        if the_basic_direction is None:
            return None

        complete_solution = np.zeros(self.n_variables)

        # Assign the values of basic variables to their corresponding positions.
        for variable, index in zip(the_basic_direction, self.basic_indices):
            complete_solution[index] = variable

        # Set the entry corresponding to the non-basic variable to 1.
        complete_solution[non_basic_index] = 1.0
        return complete_solution
