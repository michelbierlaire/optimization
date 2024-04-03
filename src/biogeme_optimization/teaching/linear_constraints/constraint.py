"""Definition of linear constraints

Michel Bierlaire
Sat Mar 23 19:06:42 2024
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, auto
from typing import Iterable, Callable

import numpy as np

from biogeme_optimization.teaching.linear_constraints.variable import (
    Variable,
    sort_variables,
    SignVariable,
    PlusMinusVariables,
)


@dataclass
class Widths:
    """Stores the widths of each term"""

    coefficient: int
    variable: int

    @property
    def total(self) -> int:
        """Calculate the total width:

        - sign: 2 chars.
        - coefficient: longest_length
        - blank space: 1
        - variable: longest variable name
        """
        return 2 + self.coefficient + 1 + self.variable


class ConstraintNamer:
    """
    Class in charge of using coherent names for constraints and slack variables.
    """

    SLACKED_SUFFIX = '_SLACKED'
    SLACK_PREFIX = 's_'

    def __init__(self, base_name: str) -> None:
        self.base_name = base_name

    @property
    def slacked_constraint_name(self) -> str:
        return f'{self.base_name}{self.SLACKED_SUFFIX}'

    @property
    def slack_variable_name(self) -> str:
        return f"{self.SLACK_PREFIX}{self.base_name}"

    @classmethod
    def get_base_name(cls, name: str) -> str:
        if name.endswith(cls.SLACKED_SUFFIX):
            return name.replace(cls.SLACKED_SUFFIX, '')
        if name.startswith(cls.SLACK_PREFIX):
            return name.replace(cls.SLACK_PREFIX, '')
        return name


class SignConstraint(Enum):
    """Possible constraint types"""

    LESSER_OR_EQUAL = auto()
    GREATER_OR_EQUAL = auto()
    EQUAL = auto()


@dataclass
class Term:
    coefficient: float
    variable: Variable
    column_width: Widths | None = None

    def __str__(self) -> str:
        if self.column_width is None:
            if self.coefficient == 1:
                return f'+ {self.variable}'
            if self.coefficient == -1:
                return f'- {self.variable}'
            if self.coefficient == 0:
                return f''

            return f'{self.coefficient:+} {self.variable}'
        formatted_coefficient = (
            f'{int(self.coefficient):>+{self.column_width.coefficient}d}'
            if self.coefficient == int(self.coefficient)
            else f'{self.coefficient:>+{self.column_width.coefficient}.3g}'
        )

        if np.abs(self.coefficient) < 1.0e-8:
            result = ' ' * self.column_width.total
            return result
        if np.abs(self.coefficient - 1) < 1.0e-8:
            term = f'+ {self.variable.name:<{self.column_width.variable}}'
            result = f'{term:>{self.column_width.total}}'
            return result
        if np.abs(self.coefficient + 1) < 1.0e-8:
            term = f'- {self.variable.name:<{self.column_width.variable}}'
            result = f'{term:>{self.column_width.total}}'
            return result
        term = f'{formatted_coefficient} {self.variable.name:<{self.column_width.variable}}'
        result = f'{term:>{self.column_width.total}}'
        return result


@dataclass
class Constraint:
    name: str
    left_hand_side: list[Term]
    sign: SignConstraint
    right_hand_side: float
    column_width: Widths | None = None

    def __str__(self) -> str:
        return self.print_constraint(all_variables=self.set_of_variables())

    def print_constraint(self, all_variables: Iterable[Variable]) -> str:
        """
        Print a constraint, taking into account all unsorted_set_of_variables, including those not involved in the constraint,
        to align all constraints properly.
        :param all_variables: all unsorted_set_of_variables involved.
        :return: formatted string
        """
        sorted_variables = sort_variables(all_variables)
        all_terms = [
            self.term_of_variable(variable_name=variable.name)
            for variable in sorted_variables
        ]
        lhs = ' '.join(all_terms)
        dict_of_sign = {
            SignConstraint.GREATER_OR_EQUAL: '>=',
            SignConstraint.LESSER_OR_EQUAL: '<=',
            SignConstraint.EQUAL: '==',
        }
        rhs = f'{self.right_hand_side}'
        return f'{lhs} {dict_of_sign[self.sign]} {rhs}     [{self.name}]'

    def __hash__(self) -> int:
        return hash(self.name)

    def term_of_variable(self, variable_name: str) -> str:
        """
        Given the name of a variable, return the corresponding term.

        :param variable_name:
        :return: formatted string for the term.
        """
        self.simplify()
        terms = [
            str(term)
            for term in self.left_hand_side
            if term.variable.name == variable_name
        ]
        assert (
            len(terms) <= 1
        ), f'Constraint {self.name} has {len(terms)} for variable {variable_name}'
        if terms:
            return terms[0]
        return ' ' * self.column_width.total

    def set_column_width(self, column_width: Widths) -> Constraint:
        """
        Set the width of the column, for aligned printing of the constraints.
        :param column_width: widths for the column
        :return: the constraint itself
        """
        self.column_width = column_width
        for term in self.left_hand_side:
            term.column_width = column_width
        return self

    def set_of_variables(self) -> set[Variable]:
        """
        :return: the set of unsorted_set_of_variables involved in the constraint
        :return: the set of unsorted_set_of_variables involved in the constraint
        """
        return set([the_term.variable for the_term in self.left_hand_side])

    def swap_sign(self) -> Constraint:
        """Change LESSER_OR_EQUAL into GREATER_OR_EQUAL, or the other way around."""
        if self.sign == SignConstraint.EQUAL:
            return self
        return Constraint(
            name=f'{self.name}',
            left_hand_side=[
                Term(coefficient=-the_term.coefficient, variable=the_term.variable)
                for the_term in self.left_hand_side
            ],
            sign=(
                SignConstraint.LESSER_OR_EQUAL
                if self.sign == SignConstraint.GREATER_OR_EQUAL
                else SignConstraint.GREATER_OR_EQUAL
            ),
            right_hand_side=-self.right_hand_side,
        )

    def replace_variables(
        self, old_variables: list[Variable], new_variables: list[Variable]
    ) -> Constraint:
        """Replace multiple unsorted_set_of_variables with new ones

        :param old_variables: list of old unsorted_set_of_variables to remove
        :param new_variables: list of new unsorted_set_of_variables
        :return: updated constraint
        """
        # Check if lists are of equal length
        if len(old_variables) != len(new_variables):
            raise ValueError(
                f'The lists of old and new unsorted_set_of_variables must have the same length: '
                f'{len(old_variables)} != {len(new_variables)}.'
            )

        # Create a mapping from old unsorted_set_of_variables to new unsorted_set_of_variables
        variable_mapping = {
            old_var: new_var for old_var, new_var in zip(old_variables, new_variables)
        }

        modified_constraint = Constraint(
            name=f'{self.name}',
            left_hand_side=[
                Term(
                    coefficient=term.coefficient,
                    variable=variable_mapping.get(term.variable, term.variable),
                    # Replace the variable if it is in the mapping
                )
                for term in self.left_hand_side
            ],
            sign=self.sign,
            right_hand_side=self.right_hand_side,
        )
        return modified_constraint

    def make_non_positive_variable_non_negative(self, variable: Variable) -> Constraint:
        """Change the sign of the coefficient of a variable.

        :param variable: non-positive variable to modify
        :return: the modified constraint
        """
        if variable.sign != SignVariable.NON_POSITIVE:
            error_msg = f'Applies only to non positive unsorted_set_of_variables. Not {variable}.'
            raise ValueError(error_msg)
        new_variable = Variable(name=variable.name, sign=SignVariable.NON_NEGATIVE)
        return Constraint(
            name=self.name,
            left_hand_side=[
                Term(
                    coefficient=(
                        -the_term.coefficient
                        if the_term.variable == variable
                        else the_term.coefficient
                    ),
                    variable=(
                        new_variable
                        if the_term.variable == variable
                        else the_term.variable
                    ),
                )
                for the_term in self.left_hand_side
            ],
            sign=self.sign,
            right_hand_side=self.right_hand_side,
        )

    def make_lesser_or_equal(self) -> list[Constraint]:
        """Generate equivalent constraint(s) of type lesser or equal

        :return: list of constraints
        """
        if self.sign == SignConstraint.LESSER_OR_EQUAL:
            return [self]
        if self.sign == SignConstraint.GREATER_OR_EQUAL:
            return [self.swap_sign()]

        constraint_1 = Constraint(
            name=f'{self.name}_lt',
            left_hand_side=self.left_hand_side,
            sign=SignConstraint.LESSER_OR_EQUAL,
            right_hand_side=self.right_hand_side,
        )
        constraint_2 = Constraint(
            name=f'{self.name}_gt',
            left_hand_side=self.left_hand_side,
            sign=SignConstraint.GREATER_OR_EQUAL,
            right_hand_side=self.right_hand_side,
        )
        return [constraint_1, constraint_2.swap_sign()]

    def add_slack_variable(self) -> Constraint:
        """Add a slack variable to obtain an equality constraint.

        :return: equality constraint
        """
        if self.sign == SignConstraint.EQUAL:
            return self
        slack = Variable(
            name=f's_{self.name}', sign=SignVariable.NON_NEGATIVE, slack=True
        )
        term = (
            Term(coefficient=1, variable=slack)
            if self.sign == SignConstraint.LESSER_OR_EQUAL
            else Term(coefficient=-1, variable=slack)
        )
        name_manager = ConstraintNamer(base_name=self.name)
        return Constraint(
            name=name_manager.slacked_constraint_name,
            left_hand_side=self.left_hand_side + [term],
            sign=SignConstraint.EQUAL,
            right_hand_side=self.right_hand_side,
        )

    def remove_slack_variable(self) -> Constraint:
        """Remove the slack variable, if any

        :return: modified constraint
        """
        terms_with_slack_variable = [
            the_term for the_term in self.left_hand_side if the_term.variable.slack
        ]
        if not terms_with_slack_variable:
            return self
        the_slack_term = terms_with_slack_variable[0]
        if len(terms_with_slack_variable) > 1:
            error_msg = f'The constraint contains {len(terms_with_slack_variable)} slack unsorted_set_of_variables.'
            raise ValueError(error_msg)
        terms_without_slack_variable = [
            the_term for the_term in self.left_hand_side if not the_term.variable.slack
        ]
        sign = (
            SignConstraint.LESSER_OR_EQUAL
            if the_slack_term.coefficient == 1
            else SignConstraint.GREATER_OR_EQUAL
        )
        the_name = ConstraintNamer.get_base_name(name=the_slack_term.variable.name)

        return Constraint(
            name=the_name,
            left_hand_side=terms_without_slack_variable,
            sign=sign,
            right_hand_side=self.right_hand_side,
        )

    def simplify(self) -> Constraint:
        """Merge terms involving the same unsorted_set_of_variables"""
        merged_terms = defaultdict(float)  # default value for coefficients is 0.0

        # Iterate through the list of terms
        for term in self.left_hand_side:
            # Add the coefficient to the variable entry in the dictionary
            merged_terms[term.variable] += term.coefficient

        # Create a new list of terms from the merged dictionary
        new_left_hand_side = [
            Term(coefficient=coefficient, variable=variable)
            for variable, coefficient in merged_terms.items()
        ]
        return Constraint(
            name=self.name,
            left_hand_side=new_left_hand_side,
            sign=self.sign,
            right_hand_side=self.right_hand_side,
        )

    def replace_variable_by_difference(
        self, variables: list[PlusMinusVariables]
    ) -> Constraint:
        """
        Replace an existing variable by the difference of two others.

        :param variables: unsorted_set_of_variables to replace, associated with the "plus" and "minus" variable
        :return: modified constraint
        """
        if variables is None:
            return self
        new_left_hand_side = []
        for term in self.left_hand_side:
            variables_to_treat = [
                the_variable
                for the_variable in variables
                if the_variable.existing == term.variable
            ]
            if not variables_to_treat:
                new_left_hand_side.append(term)
            else:
                if len(variables_to_treat) > 1:
                    error_msg = f'There are {len(variables_to_treat)} unsorted_set_of_variables {term.variable}. Only one is allowed.'
                    raise ValueError(error_msg)

                the_variable_to_treat = variables_to_treat[0]
                new_left_hand_side.append(
                    Term(
                        coefficient=term.coefficient,
                        variable=the_variable_to_treat.plus,
                    )
                )
                new_left_hand_side.append(
                    Term(
                        coefficient=-term.coefficient,
                        variable=the_variable_to_treat.minus,
                    )
                )
        return Constraint(
            name=self.name,
            left_hand_side=new_left_hand_side,
            sign=self.sign,
            right_hand_side=self.right_hand_side,
        )

    def calculate_left_hand_side(
        self, vector_x: np.ndarray, indices: dict[Variable, int]
    ) -> float:
        """
        Calculate the value of the left-hand side of the constraint
        :param vector_x: vector of values for x.
        :param indices: mapping between the unsorted_set_of_variables and the indices.
        :return: value of the lhs.
        """
        total = 0.0
        for term in self.left_hand_side:
            index = indices.get(term.variable)
            assert index is not None, f'Unknown variable {term.variable.name}'
            assert index < len(
                vector_x
            ), f'Index {index} out of range [0, {len(vector_x)-1}]'
            total += term.coefficient * vector_x[index]
        return total

    def calculate_slack(
        self, vector_x: np.ndarray, indices: dict[Variable, int]
    ) -> tuple[float, SignConstraint]:
        """
        Calculate the value of the slack, that is the right-hand-side minus the left-hand side
        :param vector_x: vector of values for x.
        :param indices: mapping between the unsorted_set_of_variables and the indices.
        :return: value of the slack, as well as the sign of the constraint.
        """
        lhs = self.calculate_left_hand_side(vector_x=vector_x, indices=indices)
        return self.right_hand_side - lhs, self.sign

    def is_feasible(self, vector_x: np.ndarray, indices: dict[Variable, int]) -> bool:
        """
        Checks feasibility of a vector
        :param vector_x: vector of interest
        :param indices: mapping between the unsorted_set_of_variables and the indices.
        :return: feasibility status
        """
        lhs = self.calculate_left_hand_side(vector_x=vector_x, indices=indices)

        def check_lesser_or_equal() -> bool:
            slack = self.right_hand_side - lhs
            if np.isclose(slack, 0, atol=np.finfo(float).eps):
                return True
            return slack >= 0.0

        def check_greater_or_equal() -> bool:
            slack = lhs - self.right_hand_side
            if np.isclose(slack, 0, atol=np.finfo(float).eps):
                return True
            return slack >= 0.0

        def check_equal() -> bool:
            slack = lhs - self.right_hand_side
            return np.isclose(slack, 0, atol=np.finfo(float).eps)

        dict_of_check: dict[SignConstraint, Callable[[], bool]] = {
            SignConstraint.GREATER_OR_EQUAL: check_greater_or_equal,
            SignConstraint.LESSER_OR_EQUAL: check_lesser_or_equal,
            SignConstraint.EQUAL: check_equal,
        }
        return dict_of_check[self.sign]()
