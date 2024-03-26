"""File examples.py

:author: Michel Bierlaire
:date: Tue Jun 27 10:51:24 2023

Examples of functions to be used in the tests
"""

import numpy as np
from biogeme_optimization.function import FunctionToMinimize, FunctionData
from biogeme_optimization.exceptions import OptimizationError


class MyFunctionToMinimize(FunctionToMinimize):
    def __init__(self, dimension: int):
        super().__init__()
        self.x: np.ndarray | None = None
        self.dim: int = dimension

    def dimension(self) -> int:
        """Dimension of the problem"""
        return self.dim

    def _f(self) -> float:
        """Calculate the canonical_value of the function

        :return: canonical_value of the function
        :rtype: float
        """
        half_sum_of_squares = np.sum(np.square(self.x)) / 2
        return half_sum_of_squares

    def _f_g(self) -> FunctionData:
        """Calculate the canonical_value of the function and the gradient

        :return: canonical_value of the function and the gradient
        :rtype: tuple float, numpy.array
        """
        half_sum_of_squares = np.sum(np.square(self.x)) / 2
        result = FunctionData(
            function=half_sum_of_squares, gradient=self.x, hessian=None
        )
        return result

    def _f_g_h(self) -> FunctionData:
        """Calculate the canonical_value of the function, the gradient and the Hessian

        :return: canonical_value of the function, the gradient and the Hessian
        :rtype: tuple float, numpy.array, numpy.array
        """
        half_sum_of_squares = np.sum(np.square(self.x)) / 2
        result = FunctionData(
            function=half_sum_of_squares,
            gradient=self.x,
            hessian=np.eye(len(self.x)),
        )
        return result


class Example58(FunctionToMinimize):
    """Implements Example 5.8, p. 121, Bierlaire (2015)"""

    def __init__(self) -> None:
        super().__init__()
        self.x: np.ndarray | None = None
        self.dim: int = 2

    def dimension(self) -> int:
        """Dimension of the problem"""
        return self.dim

    def _f(self) -> float:
        """Calculate the canonical_value of the function

        :return: canonical_value of the function
        :rtype: float
        """
        return 0.5 * self.x[0] ** 2 + self.x[0] * np.cos(self.x[1])

    def _f_g(self) -> FunctionData:
        """Calculate the canonical_value of the function and the gradient

        :return: canonical_value of the function and the gradient
        :rtype: FunctionData tuple float, numpy.array
        """
        f = self.f()
        g = np.array([self.x[0] + np.cos(self.x[1]), -self.x[0] * np.sin(self.x[1])])
        result = FunctionData(function=f, gradient=g, hessian=None)
        return result

    def _f_g_h(self) -> FunctionData:
        """Calculate the canonical_value of the function, the gradient and the Hessian

        :return: canonical_value of the function, the gradient and the Hessian
        :rtype: tuple float, numpy.array, numpy.array
        """
        f = self.f()
        g = np.array([self.x[0] + np.cos(self.x[1]), -self.x[0] * np.sin(self.x[1])])
        h = np.array(
            [
                [1, -np.sin(self.x[1])],
                [-np.sin(self.x[1]), -self.x[0] * np.cos(self.x[1])],
            ]
        )
        result = FunctionData(function=f, gradient=g, hessian=h)
        return result


class Rosenbrock(FunctionToMinimize):
    """Implements Example 5.3, p. 116, Bierlaire (2015)"""

    def __init__(self) -> None:
        super().__init__()
        self.x: np.ndarray | None = None
        self.dim: int = 2

    def dimension(self) -> int:
        """Dimension of the problem"""
        return self.dim

    def _f(self) -> float:
        """Calculate the canonical_value of the function

        :return: canonical_value of the function
        :rtype: float
        """
        f = 100.0 * (self.x[1] - self.x[0] * self.x[0]) ** 2 + (1 - self.x[0]) ** 2
        return f

    def _f_g(self) -> FunctionData:
        """Calculate the canonical_value of the function and the gradient

        :return: canonical_value of the function and the gradient
        :rtype: FunctionData tuple float, numpy.array
        """
        f = 100.0 * (self.x[1] - self.x[0] * self.x[0]) ** 2 + (1 - self.x[0]) ** 2
        g = np.array(
            [
                400.0 * self.x[0] ** 3
                - 400.0 * self.x[0] * self.x[1]
                + 2 * self.x[0]
                - 2,
                200.0 * self.x[1] - 200.0 * self.x[0] ** 2,
            ]
        )
        result = FunctionData(function=f, gradient=g, hessian=None)
        return result

    def _f_g_h(self) -> FunctionData:
        """Calculate the canonical_value of the function, the gradient and the Hessian

        :return: canonical_value of the function, the gradient and the Hessian
        :rtype: tuple float, numpy.array, numpy.array
        """
        f = 100.0 * (self.x[1] - self.x[0] * self.x[0]) ** 2 + (1 - self.x[0]) ** 2
        g = np.array(
            [
                400.0 * self.x[0] ** 3
                - 400.0 * self.x[0] * self.x[1]
                + 2 * self.x[0]
                - 2,
                200.0 * self.x[1] - 200.0 * self.x[0] ** 2,
            ]
        )
        h = np.array(
            [
                [1200.0 * self.x[0] ** 2 - 400.0 * self.x[1] + 2, -400 * self.x[0]],
                [-400 * self.x[0], 200],
            ]
        )
        result = FunctionData(function=f, gradient=g, hessian=h)
        return result
