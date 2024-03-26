"""Definition of linear constraints

Michel Bierlaire
Sat Mar 23 17:24:45 2024
"""

from __future__ import annotations

import logging
from collections import Counter
from functools import reduce
from typing import Iterator

import numpy as np

from biogeme_optimization.teaching.matrices import find_columns_multiple_identity
from .constraint import Widths, SignConstraint, Term, Constraint
from .variable import SignVariable, PlusMinusVariables, Variable, sort_variables

logger = logging.getLogger(__name__)

PLUS_SUFFIX = '_+'
MINUS_SUFFIX = '_-'


class AllConstraints:
    def __init__(self, constraints: list[Constraint]) -> None:
        self._constraints: list[Constraint] = constraints
        # Verify that all names are unique
        counts = Counter(constraint.name for constraint in constraints)
        if duplicates := [name for name, count in counts.items() if count > 1]:
            error_msg = (
                f'Constraint name must be unique. It is not the case for {duplicates}'
            )
            raise ValueError(error_msg)
        self.unsorted_set_of_variables: set[Variable] = reduce(
            lambda x, y: x | y,
            (a_constraint.set_of_variables() for a_constraint in self.constraints),
        )
        self._variable_to_index: dict[Variable, int] | None = None
        self._index_to_variable: list[Variable] | None = None
        self._constraint_indices: dict[Variable, int] | None = None
        self.plus_minus_replacements: list[PlusMinusVariables] | None = None
        self._detect_unsigned_variables()

    def __iter__(self) -> Iterator:
        return iter(self._constraints)

    @classmethod
    def from_canonical_form(
        cls, matrix: np.ndarray, vector: np.array
    ) -> AllConstraints:
        """
        Alternative constructor

        :param matrix: matrix of the standard form
        :param vector: vector of the standard form
        :return: object


        """

        # We first identify the sign constraints in the original formulation. They correspond to a row of A
        # containing only one -1, or one 1, with the corresponding entry of b being 0. Those rows can be removed.

        # Identification of the rows corresponding to non negativity constraints.

        n_constraints, n_variables = matrix.shape
        mask_non_negativity_rows = (
            ((matrix == -1).sum(axis=1) == 1)
            & ((matrix == 0).sum(axis=1) == (n_variables - 1))
            & (vector == 0)
        )
        non_negativity_constraints_indices = np.where(mask_non_negativity_rows)[0]

        mask_non_positivity_rows = (
            ((matrix == 1).sum(axis=1) == 1)
            & ((matrix == 0).sum(axis=1) == (n_variables - 1))
            & (vector == 0)
        )

        non_positivity_constraints_indices = np.where(mask_non_positivity_rows)[0]

        mask_sign_constraints = mask_non_negativity_rows | mask_non_positivity_rows

        general_constraints_indices = np.where(~mask_sign_constraints)[0]

        non_negative_variables = []
        for row_index in non_negativity_constraints_indices:
            row = matrix[row_index, :]
            # np.where() will return a tuple where the first element is an array of indices
            col_index = np.where(row == -1)[
                0
            ]  # Get the first (and should be only) index
            if col_index.size > 0:  # Check if there is indeed a -1 entry
                the_index = int(col_index[0])
                non_negative_variables.append(the_index)
        non_positive_variables = []
        for row_index in non_positivity_constraints_indices:
            row = matrix[row_index, :]
            # np.where() will return a tuple where the first element is an array of indices
            col_index = np.where(row == 1)[
                0
            ]  # Get the first (and should be only) index
            if col_index.size > 0:  # Check if there is indeed a -1 entry
                the_index = int(col_index[0])
                non_positive_variables.append(the_index)

        def identify_sign(index: int) -> SignVariable:
            if index in non_negative_variables:
                return SignVariable.NON_NEGATIVE
            if index in non_positive_variables:
                return SignVariable.NON_POSITIVE
            return SignVariable.UNSIGNED

        all_variables = [
            Variable(
                name=f'x_{col_index}', sign=identify_sign(index=col_index), slack=False
            )
            for col_index in range(n_variables)
        ]
        constraints = []
        for row_index in general_constraints_indices:
            row = matrix[row_index, :]
            lhs = []
            for col_index, coefficient in enumerate(row):
                term = Term(
                    coefficient=float(coefficient), variable=all_variables[col_index]
                )
                lhs.append(term)
            sign = SignConstraint.LESSER_OR_EQUAL
            rhs = vector[row_index]
            constraint = Constraint(
                name=f'c_{row_index}',
                left_hand_side=lhs,
                sign=sign,
                right_hand_side=rhs,
            )
            constraints.append(constraint)
        return AllConstraints(constraints)

    @classmethod
    def from_standard_form(cls, matrix: np.ndarray, vector: np.array) -> AllConstraints:
        """
        Alternative constructor

        :param matrix: matrix of the standard form
        :param vector: vector of the standard form
        :return: object
        """
        n_constraints, n_variables = matrix.shape

        # Slack unsorted_set_of_variables: find columns that are multiple of the identity matrix
        the_list = find_columns_multiple_identity(matrix=matrix)
        slack_variables_indices = set([col_index for col_index, _ in the_list])
        slacked_constraints_indices = set([row_index for _, row_index in the_list])
        # Make sure the only entry in those columns is positive
        for col_index, row_index in the_list:
            if matrix[row_index, col_index] < 0:
                matrix[row_index, :] = -matrix[row_index, :]

        regular_variables_indices = set(range(n_variables)) - slack_variables_indices
        regular_variables = {
            index: Variable(
                name=f'x_{index}', sign=SignVariable.NON_NEGATIVE, slack=False
            )
            for index in regular_variables_indices
        }
        constraints = []
        for row_index, row in enumerate(matrix):
            lhs = []
            for col_index, coefficient in enumerate(row):
                if col_index not in slack_variables_indices:
                    term = Term(
                        coefficient=coefficient, variable=regular_variables[col_index]
                    )
                    lhs.append(term)
            sign = (
                SignConstraint.LESSER_OR_EQUAL
                if row_index in slacked_constraints_indices
                else SignConstraint.EQUAL
            )
            rhs = vector[row_index]
            constraint = Constraint(
                name=f'c_{row_index}',
                left_hand_side=lhs,
                sign=sign,
                right_hand_side=rhs,
            )
            constraints.append(constraint)
        return AllConstraints(constraints)

    @property
    def n_constraints(self) -> int:
        return len(self.constraints)

    @property
    def n_variables(self) -> int:
        return len(self.unsorted_set_of_variables)

    @property
    def constraints(self) -> list[Constraint]:
        return self._constraints

    @property
    def index_to_variable(self) -> list[Variable]:
        if self._index_to_variable is None:
            self._index_to_variable = sort_variables(
                list(self.unsorted_set_of_variables)
            )
        return self._index_to_variable

    @property
    def variable_to_index(self) -> dict[Variable, int]:
        """
        :return: dictionary mapping the unsorted_set_of_variables with indices
        """
        if self._variable_to_index is None:
            self._variable_to_index = {
                variable: index for index, variable in enumerate(self.index_to_variable)
            }
        return self._variable_to_index

    def get_constraint_by_name(self, constraint_name: str) -> Constraint | None:
        """"""
        return next(
            (
                constraint
                for constraint in self.constraints
                if constraint.name == constraint_name
            ),
            None,
        )

    def variable_value(self, variable: Variable, vector: np.ndarray) -> float | None:
        """
        Extract the value of a variable from the vector knowing its name.
        :param variable: variable of interest
        :param vector: vector of values
        :return: value of the variable if found. None otherwise.
        """
        assert (
            len(vector) == self.n_variables
        ), f'Incompatible vector size: {len(vector)} instead of {self.n_variables}.'
        index = self.variable_to_index.get(variable)
        if index is None:
            return None
        return float(vector[index])

    @property
    def constraint_indices(self) -> dict[Constraint, int]:
        if self._constraint_indices is None:
            self._constraint_indices = {
                constraint: index for index, constraint in enumerate(self.constraints)
            }
        return self._constraint_indices

    def __str__(self) -> str:
        column_width = self.calculate_column_width()

        if self.constraints is None:
            return ''
        constraints = [
            constraint.set_column_width(column_width=column_width).print_constraint(
                self.unsorted_set_of_variables
            )
            for constraint in self.constraints
        ]
        dict_of_sign = {
            SignVariable.UNSIGNED: 'in R',
            SignVariable.NON_NEGATIVE: '>= 0',
            SignVariable.NON_POSITIVE: '<= 0',
        }
        sorted_variables = sort_variables(list(self.unsorted_set_of_variables))
        constraints += [
            f'{variable.name} {dict_of_sign[variable.sign]}'
            for variable in sorted_variables
        ]
        return '\n'.join(constraints)

    def _detect_unsigned_variables(self) -> None:
        unsigned_variables = [
            the_variable
            for the_variable in self.unsorted_set_of_variables
            if the_variable.sign == SignVariable.UNSIGNED
        ]
        if not unsigned_variables:
            return
        self.plus_minus_replacements = [
            PlusMinusVariables(
                existing=the_variable,
                plus=Variable(
                    name=f'{the_variable.name}{PLUS_SUFFIX}',
                    sign=SignVariable.NON_NEGATIVE,
                ),
                minus=Variable(
                    name=f'{the_variable.name}{MINUS_SUFFIX}',
                    sign=SignVariable.NON_NEGATIVE,
                ),
            )
            for the_variable in unsigned_variables
        ]

    def remove_unsigned_variables(self) -> AllConstraints:
        """Replace each unsigned variable by the difference of non-negative unsorted_set_of_variables."""
        return AllConstraints(
            constraints=[
                the_constraint.replace_variable_by_difference(
                    variables=self.plus_minus_replacements
                )
                for the_constraint in self.constraints
            ]
        )

    def make_variables_non_negative(self) -> AllConstraints:
        """
        Modifies the specification of the constraints to make all unsorted_set_of_variables non-negative.
        :return: equivalent set of constraints.
        """
        only_signed_constraints = self.remove_unsigned_variables()
        for variable in only_signed_constraints.unsorted_set_of_variables:
            if variable.sign == SignVariable.UNSIGNED:
                raise ValueError(
                    'At this point, all unsorted_set_of_variables must be signed'
                )
            if variable.sign == SignVariable.NON_POSITIVE:
                only_signed_constraints = AllConstraints(
                    constraints=[
                        constraint.make_non_positive_variable_non_negative(
                            variable=variable
                        )
                        for constraint in only_signed_constraints.constraints
                    ]
                )
        return only_signed_constraints

    def make_variables_unsigned(self) -> AllConstraints:
        """
        Modifies the specification of the constraints to make all unsorted_set_of_variables unsigned.
        :return: equivalent set of constraints.
        """
        non_negative_variables = [
            variable
            for variable in self.unsorted_set_of_variables
            if variable.sign == SignVariable.NON_NEGATIVE
        ]
        non_negative_constraints = [
            Constraint(
                name=non_negative_variable.name,
                left_hand_side=[Term(coefficient=1.0, variable=non_negative_variable)],
                sign=SignConstraint.GREATER_OR_EQUAL,
                right_hand_side=0,
            )
            for non_negative_variable in non_negative_variables
        ]
        non_positive_variables = [
            variable
            for variable in self.unsorted_set_of_variables
            if variable.sign == SignVariable.NON_POSITIVE
        ]
        non_positive_constraints = [
            Constraint(
                name=non_positive_variable.name,
                left_hand_side=[Term(coefficient=1.0, variable=non_positive_variable)],
                sign=SignConstraint.LESSER_OR_EQUAL,
                right_hand_side=0,
            )
            for non_positive_variable in non_positive_variables
        ]
        signed_variables = non_positive_variables + non_negative_variables
        unsigned_variables = [
            Variable(name=variable.name, sign=SignVariable.UNSIGNED, slack=False)
            for variable in signed_variables
        ]
        constraints = [
            the_constraint.replace_variables(signed_variables, unsigned_variables)
            for the_constraint in (
                self.constraints + non_negative_constraints + non_positive_constraints
            )
        ]

        return AllConstraints(constraints=constraints)

    def slack_value(
        self, values: np.ndarray, constraint_name: str
    ) -> tuple[float, SignConstraint]:
        """
        Calculate the difference between the left and the right hand side of a constraint at a point.
        :param values: coordinates of the point.
        :param constraint_name: name of the constraint.
        :return: the slack, as well as the sign of the constraint.
        """

    def transform_standard_form(self) -> AllConstraints:
        """Transform the set of constraints into standard form"""
        with_slack = AllConstraints(
            constraints=[
                the_constraint.add_slack_variable()
                for the_constraint in self.constraints
            ]
        )
        return with_slack.make_variables_non_negative()

    def transform_canonical_form(self) -> AllConstraints:
        """Transform the set of constraints into canonical form"""
        without_slack = AllConstraints(
            constraints=[
                the_constraint.remove_slack_variable()
                for the_constraint in self.constraints
            ]
        )
        unsigned_variables = without_slack.make_variables_unsigned()
        canonical_form = AllConstraints(
            constraints=[
                result
                for constraint in unsigned_variables.constraints
                for result in constraint.make_lesser_or_equal()
            ]
        )
        return canonical_form

    def calculate_column_width(self) -> Widths:
        """Scan the constraints to calculate the column widths"""
        longest_variable_name = max(
            len(var.name) for var in self.unsorted_set_of_variables
        )
        all_coefficients = [
            f'{term.coefficient:.3g}'
            for constraint in self.constraints
            for term in constraint.left_hand_side
        ] + [f'{constraint.right_hand_side:.3g}' for constraint in self.constraints]

        # Find the longest string length among all formatted elements
        longest_length = max([len(item) for item in all_coefficients])
        return Widths(
            coefficient=longest_length,
            variable=longest_variable_name,
        )

    def is_feasible(self, vector_x: np.ndarray) -> tuple[bool, dict[str, bool]]:
        """
        Check feasibility of a vector.
        :param vector_x: vector of interest.
        :return:  feasibility status for the polyhedron, and for each constraint.
        """

        feasible_dict = {
            constraint.name: constraint.is_feasible(
                vector_x=vector_x, indices=self.variable_to_index
            )
            for constraint in self.constraints
        }
        return all(feasible_dict.values()), feasible_dict
