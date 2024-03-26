"""Definition of linear variable

Michel Bierlaire
Sat Mar 23 19:06:28 2024
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Iterable


class SignVariable(Enum):
    """Possible sign constraints on the unsorted_set_of_variables"""

    NON_NEGATIVE = auto()
    NON_POSITIVE = auto()
    UNSIGNED = auto()


@dataclass
class PlusMinusVariables:
    existing: Variable
    plus: Variable
    minus: Variable

    def __post_init__(self) -> None:
        if self.existing.sign != SignVariable.UNSIGNED:
            error_msg = f'This transformation can be applied only on unsigned unsorted_set_of_variables. Not {self.existing}'
            raise ValueError(error_msg)
        if self.plus.sign != SignVariable.NON_NEGATIVE:
            error_msg = f'Expecting a non-negative variable. Not {self.plus}'
            raise ValueError(error_msg)
        if self.minus.sign != SignVariable.NON_NEGATIVE:
            error_msg = f'Expecting a non-negative variable. Not {self.plus}'
            raise ValueError(error_msg)


@dataclass
class Variable:
    name: str
    sign: SignVariable
    slack: bool = False

    def __str__(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __lt__(self, other: Variable) -> bool:
        return self.name < other.name

    def __eq__(self, other: Variable) -> bool:
        return self.name == other.name


def sort_variables(variables: Iterable[Variable]) -> list[Variable]:
    """
    Sort unsorted_set_of_variables by alphabetical order, keeping the slack unsorted_set_of_variables at the end.

    :param variables: unsorted_set_of_variables to sort.
    :return: sorted unsorted_set_of_variables.
    """
    regular = sorted(
        [variable for variable in variables if not variable.slack],
        key=lambda x: x.name,
    )
    slack = sorted(
        [variable for variable in variables if variable.slack],
        key=lambda x: x.name,
    )
    return regular + slack
