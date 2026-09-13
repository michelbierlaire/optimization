"""File diagnostics.py

:author: Michel Bierlaire
:date: Wed Jun 28 16:17:43 2023

Classes for algorithm diagnostics
"""

from enum import Enum, auto
from typing import NamedTuple

import numpy as np

from biogeme_optimization.state import TrustRegionBFGSState


class OptimizationResults(NamedTuple):
    """Typed result returned by the public optimization entry points."""

    solution: np.ndarray
    messages: dict[str, object]
    convergence: bool
    state: TrustRegionBFGSState | None = None


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
