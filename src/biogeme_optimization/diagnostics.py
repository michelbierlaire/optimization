"""File diagnostics.py

:author: Michel Bierlaire
:date: Wed Jun 28 16:17:43 2023

Classes for algorithm diagnostics
"""

from enum import Enum, auto

import numpy as np

from biogeme_optimization.state import TrustRegionBFGSState


class OptimizationResults(tuple):
    """Result returned by optimization entry points.

    The tuple deliberately retains the historical three-item shape used by
    Biogeme and older callers.  The resumable optimizer state is an attribute,
    rather than a fourth tuple item, so existing unpacking remains valid::

        solution, messages, convergence = result
    """

    state: TrustRegionBFGSState | None

    def __new__(  # noqa: PYI034 - Self is unavailable on Python 3.10
        cls,
        solution: np.ndarray,
        messages: dict[str, object],
        convergence: bool,
        state: TrustRegionBFGSState | None = None,
    ) -> 'OptimizationResults':
        result = super().__new__(cls, (solution, messages, convergence))
        result.state = state
        return result

    @property
    def solution(self) -> np.ndarray:
        """The solution vector."""
        return self[0]

    @property
    def messages(self) -> dict[str, object]:
        """The optimizer diagnostic messages."""
        return self[1]

    @property
    def convergence(self) -> bool:
        """Whether the optimizer converged."""
        return self[2]


class Diagnostic(Enum):
    pass


class DoglegDiagnostic(Diagnostic):
    """Possible outcomes of the dogleg method"""

    NEGATIVE_CURVATURE = auto()
    CAUCHY = auto()
    PARTIAL_CAUCHY = auto()
    NEWTON = auto()
    PARTIAL_NEWTON = auto()
    DOGLEG = auto()


class ConjugateGradientDiagnostic(Diagnostic):
    """Possible outcomes of the conjugate gradient method"""

    CONVERGENCE = auto()
    OUT_OF_TRUST_REGION = auto()
    NEGATIVE_CURVATURE = auto()
    NUMERICAL_PROBLEM = auto()
