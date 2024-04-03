"""Linear constraints in standard and canonical forms

Michel Bierlaire
Sat Mar 23 17:25:20 2024
"""

from __future__ import annotations

import numpy as np

from biogeme_optimization.teaching.linear_constraints.constraint import (
    SignConstraint,
    ConstraintNamer,
)
from biogeme_optimization.teaching.linear_constraints.linear_constraints import (
    AllConstraints,
)
from biogeme_optimization.teaching.linear_constraints.variable import (
    SignVariable,
)


class StandardCanonicalForms:
    def __init__(self, constraints: AllConstraints):
        self.constraints = constraints
        self.standard_form: AllConstraints = constraints.transform_standard_form()
        self.canonical_form: AllConstraints = constraints.transform_canonical_form()
        self._standard_matrix: np.ndarray | None = None
        self._standard_vector: np.ndarray | None = None
        self._canonical_matrix: np.ndarray | None = None
        self._canonical_vector: np.ndarray | None = None

    @property
    def standard_matrix(self) -> np.ndarray:
        if self._standard_matrix is None:
            self._standard_matrix = np.zeros(
                shape=(self.standard_form.n_constraints, self.standard_form.n_variables)
            )
            self._standard_vector = np.zeros(self.standard_form.n_constraints)
            for constraint in self.standard_form.constraints:
                constraint_index = self.standard_form.constraint_indices[constraint]
                for term in constraint.left_hand_side:
                    variable_index = self.standard_form.variable_to_index[term.variable]
                    self._standard_matrix[constraint_index, variable_index] = (
                        term.coefficient
                    )
                self._standard_vector[constraint_index] = constraint.right_hand_side

        return self._standard_matrix

    @property
    def standard_vector(self) -> np.ndarray:
        if self._standard_vector is None:
            _ = (
                self.standard_matrix
            )  # triggers the calculation of the matrix and the vector
        return self._standard_vector

    @property
    def canonical_matrix(self) -> np.ndarray:
        if self._canonical_matrix is None:
            self._canonical_matrix = np.zeros(
                shape=(
                    self.canonical_form.n_constraints,
                    self.canonical_form.n_variables,
                )
            )
            self._canonical_vector = np.zeros(self.canonical_form.n_constraints)
            for constraint in self.canonical_form.constraints:
                constraint_index = self.canonical_form.constraint_indices[constraint]
                for term in constraint.left_hand_side:
                    variable_index = self.canonical_form.variable_to_index[
                        term.variable
                    ]
                    self._canonical_matrix[constraint_index, variable_index] = (
                        term.coefficient
                    )
                self._canonical_vector[constraint_index] = constraint.right_hand_side

        return self._canonical_matrix

    @property
    def canonical_vector(self) -> np.ndarray:
        if self._canonical_vector is None:
            _ = (
                self.canonical_matrix
            )  # triggers the calculation of the matrix and the vector
        return self._canonical_vector

    def from_canonical_to_standard(self, canonical_x: np.ndarray) -> np.ndarray:
        """Transforms a vector from the polyhedron in canonical form to the equivalent vector in standard form.
        :param canonical_x: vector in canonical form
        :return: vector in standard form
        """
        assert (
            len(canonical_x) == self.canonical_form.n_variables
        ), f'Wrong dimension: {len(canonical_x)} instead of {self.canonical_form.n_variables}.'

        standard_x = np.full(self.standard_form.n_variables, np.nan)

        for index, variable in enumerate(self.standard_form.index_to_variable):
            # Check first if the same variable exists in the canonical form,
            canonical_index = self.canonical_form.variable_to_index.get(variable)
            if canonical_index is not None:
                standard_x[index] = canonical_x[canonical_index]
            # Check if it is a slack variable.
            elif variable.slack:
                assert (
                    variable.sign == SignVariable.NON_NEGATIVE
                ), f'Slack variable {variable.name} must be non negative.'
                constraint_name = ConstraintNamer.get_base_name(name=variable.name)
                the_constraint = self.canonical_form.get_constraint_by_name(
                    constraint_name=constraint_name
                )
                slack_value, sign = the_constraint.calculate_slack(
                    vector_x=canonical_x, indices=self.canonical_form.variable_to_index
                )
                assert (
                    sign == SignConstraint.LESSER_OR_EQUAL
                ), f'Canonical constraint {constraint_name} is not lesser or equal: {sign}.'
                standard_x[index] = slack_value
            # Last possibility is a plus or minus variable.
            elif self.canonical_form.plus_minus_replacements is not None:
                for plus_minus in self.canonical_form.plus_minus_replacements:
                    if variable == plus_minus.plus:
                        canonical_index = self.canonical_form.variable_to_index.get(
                            plus_minus.existing
                        )
                        assert (
                            canonical_index is not None
                        ), f'Unknown variable {plus_minus.existing}'
                        standard_x[index] = (
                            canonical_x[canonical_index]
                            if canonical_x[canonical_index] > 0
                            else 0.0
                        )
                    elif variable == plus_minus.minus:
                        canonical_index = self.canonical_form.variable_to_index.get(
                            plus_minus.existing
                        )
                        assert (
                            canonical_index is not None
                        ), f'Unknown variable {plus_minus.existing}'
                        standard_x[index] = (
                            -canonical_x[canonical_index]
                            if canonical_x[canonical_index] < 0
                            else 0.0
                        )
            else:
                error_msg = f'Cannot identify standard variable {variable.name}.'
                raise ValueError(error_msg)
        assert not np.any(np.isnan(standard_x)), 'The vector contains NaN entries.'
        return standard_x

    def from_standard_to_canonical(self, standard_x: np.ndarray) -> np.ndarray:
        """Transforms a vector from the polyhedron in standard form to the equivalent vector in canonical form.
        :param standard_x: vector in standard form
        :return: vector in canonical form
        """
        assert (
            len(standard_x) == self.standard_form.n_variables
        ), f'Wrong dimension: {len(standard_x)} instead of {self.standard_form.n_variables}.'

        canonical_x = np.full(self.canonical_form.n_variables, np.nan)
        for index, variable in enumerate(self.canonical_form.index_to_variable):
            # Check first if the same variable exists in the standard form,
            standard_index = self.standard_form.variable_to_index.get(variable)
            if standard_index is not None:
                canonical_x[index] = standard_x[standard_index]
            # Check for plus or minus variable.
            elif self.canonical_form.plus_minus_replacements is not None:
                the_plus_minus = next(
                    (
                        plus_minus
                        for plus_minus in self.canonical_form.plus_minus_replacements
                        if plus_minus.existing == variable
                    ),
                    None,
                )
                if the_plus_minus is not None:
                    standard_plus_index = self.standard_form.variable_to_index.get(
                        the_plus_minus.plus
                    )
                    assert (
                        standard_plus_index is not None
                    ), f'Unknown variable {the_plus_minus.plus}'
                    standard_minus_index = self.standard_form.variable_to_index.get(
                        the_plus_minus.minus
                    )
                    assert (
                        standard_minus_index is not None
                    ), f'Unknown variable {the_plus_minus.minus}'
                    canonical_x[index] = (
                        standard_x[standard_plus_index]
                        - standard_x[standard_minus_index]
                    )
            else:
                error_msg = f'Cannot identify standard variable {variable.name}.'
                raise ValueError(error_msg)
        assert not np.any(np.isnan(canonical_x)), 'The vector contains NaN entries.'
        return canonical_x


def canonical_to_standard(
    canonical_a: np.ndarray, canonical_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Transforms a polyhedron in canonical form into one in standard form.

    :param canonical_a: constraints matrix $A$ for the canonical form.
    :param canonical_b: right hand side for the canonical form.
    :return: three items:

        - constraints matrix $A$ for the standard form,
        - right hand side for the standard form,
        - description of the constraints in standard form.
    """
    the_constraints = AllConstraints.from_canonical_form(
        matrix=canonical_a, vector=canonical_b
    )
    the_forms = StandardCanonicalForms(constraints=the_constraints)
    return (
        the_forms.standard_matrix,
        the_forms.standard_vector,
        str(the_forms.standard_form),
    )
