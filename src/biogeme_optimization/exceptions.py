"""Defines a generic exception for the optimization package

:author: Michel Bierlaire

:date: Wed Jun 21 10:34:51 2023

"""


class OptimizationError(Exception):
    """Defines a generic exception ."""


class CheckpointError(OptimizationError):
    """Raised when a trust-region optimizer checkpoint is invalid or fails."""
