"""File diagnostics.py

:author: Michel Bierlaire
:date: Wed Jun 28 16:17:43 2023

Classes for alogithm diagnostics
"""
from enum import Enum, auto


class DoglegDiagnostic(Enum):
    """Possible aoutcomes of the dogleg method"""

    NEGATIVE_CURVATURE = auto()
    CAUCHY = auto()
    PARTIAL_CAUCHY = auto()
    NEWTON = auto()
    PARTIAL_NEWTON = auto()
    DOGLEG = auto()


class ConjugateGradientDiagnostic(Enum):
    """Possible outcomes of the conjugate gradient method"""

    CONVERGENCE = auto()
    OUT_OF_TRUST_REGION = auto()
    NEGATIVE_CURVATURE = auto()
    NUMERICAL_PROBLEM = auto()
