"""Definition of canonical and standard unsorted_set_of_variables.

Michel Bierlaire
Sat Mar 16 12:41:41 2024
"""

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

REGULAR_VARIABLE = 'x'
SLACK_VARIABLE = 's'


class SignConstraint(Enum):
    NON_NEGATIVE = auto()
    NON_POSITIVE = auto()
    FREE = auto()


class VariableType(Enum):
    ORIGINAL = auto()
    SLACK = auto()
    POSITIVE_DIFF = auto()
    NEGATIVE_DIFF = auto()


@dataclass
class CanonicalVariable:
    """A non-negative canonical variable is associated to one standard variable, so thst the negative_standard_index
    can be set to None"""

    name: str
    index: int
    positive_standard_index: int | None = None
    negative_standard_index: int | None = None

    def to_standard_values(
        self, canonical_vector: np.array
    ) -> tuple[list[int], list[float]]:
        """Get the corresponding values of the unsorted_set_of_variables in standard form

        :param canonical_vector: value of the canonical variable
        :return: list of indices and list of values of the standard vatiables.
        """
        if self.positive_standard_index is None:
            raise ValueError('No link with standard unsorted_set_of_variables.')
        canonical_value = canonical_vector[self.index]
        if self.negative_standard_index:
            indices = [self.positive_standard_index, self.negative_standard_index]
            values = [
                canonical_value if canonical_value > 0 else 0,
                -canonical_value if canonical_value < 0 else 0,
            ]
            return indices, values
        indices = [self.positive_standard_index]
        values = [canonical_value]
        return indices, values

    def from_standard_values(self, standard_vector: np.array) -> float:
        """Get the value of the canonical variable from the vector of standard unsorted_set_of_variables.

        :param standard_vector: vector of standard unsorted_set_of_variables
        :return: value of the canonical variable
        """
        if self.positive_standard_index is None:
            raise ValueError('No link with standard unsorted_set_of_variables.')

        if self.negative_standard_index:
            return (
                standard_vector[self.positive_standard_index]
                - standard_vector[self.negative_standard_index]
            )
        return standard_vector[self.positive_standard_index]


@dataclass
class StandardVariable:
    name: str
    index: int
    type: VariableType
    canonical_variable: CanonicalVariable | None = None
    canonical_constraint: int | None = None

    def __post_init__(self) -> None:
        if self.canonical_variable is None:
            if self.type == VariableType.SLACK:
                return
            error_msg = f'No canonical variable has been defined for standard variable {self.name}'
            raise ValueError(error_msg)
        if self.type == VariableType.ORIGINAL:
            self.canonical_variable.positive_standard_index = self.index
            self.canonical_variable.negative_standard_index = None
            return
        if self.type == VariableType.POSITIVE_DIFF:
            self.canonical_variable.positive_standard_index = self.index
            return
        if self.type == VariableType.NEGATIVE_DIFF:
            self.canonical_variable.negative_standard_index = self.index
            return
        error_msg = 'Unknown variable type'
        raise ValueError(error_msg)
