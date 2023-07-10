"""File function.py

:author: Michel Bierlaire
:date: Thu Jun 22 16:42:46 2023

Abstract class for the function to minimize
"""
import logging
from abc import ABC, abstractmethod
from typing import final
from collections import namedtuple
import numpy as np

FunctionData = namedtuple('FunctionData', ['function', 'gradient', 'hessian'])

logger = logging.getLogger(__name__)


class FunctionToMinimize(ABC):
    """This is an abstract class. The actual function to minimize
    must be implemented in a concrete class deriving from this one.

    """

    def __init__(self, epsilon=None, steptol=None):
        """Constructor
        :param epsilon: tolerance on the relative gradient for convergence
        :type epsilon: float

        :param steptol: tolerance of the step length. If the distance
            between two successive iterates is less than steptol, the
            iterations should be interrupted.
        :type  steptol: float

        """
        self.epsilon = (
            np.finfo(np.float64).eps ** 0.3333 if epsilon is None else epsilon
        )
        self.steptol = 1.0e-5 if steptol is None else steptol
        self.reset()

    @final
    def reset(self):
        """Erase all stored values"""
        self.stored_values = {}
        self.stored_gradient = {}
        self.stored_hessian = {}
        self.x = None
        self.x_tuple = None
        self.typx = None
        self.typf = None
        self.relgrad = None
        self.messages = {}

    @final
    def needs_reset(self):
        """Checks if the function needs to be reset

        :return: True if calculated values of the function are stored
        :rtype: bool
        """
        return bool(self.stored_values)

    @abstractmethod
    def dimension(self):
        """Provides the number of variables of the problem"""

    def calculate_relative_projected_gradient(self, bounds=None):
        """Calculates the relative projected gradient

        :param bounds: object describing the bound constraints
        :type bounds: Bounds

        :return: True is optimal, False otherwise
        :rtype: bool

        """
        evaluation = self.f_g()
        value_function = evaluation.function
        gradient = evaluation.gradient

        projected_gradient = (
            gradient if bounds is None else bounds.project(self.x - gradient) - self.x
        )

        if self.typx is None:
            self.typx = np.ones(np.asarray(self.x).shape)
        if self.typf is None:
            self.typf = max(np.abs(evaluation.function), 1.0)
        self.relgrad = relative_gradient(
            self.x, value_function, projected_gradient, self.typx, self.typf
        )

    def check_optimality(self, bounds=None):
        """Verifies the optimality of the current iterate

        :param bounds: object describing the bound constraints
        :type bounds: Bounds

        :return: True is optimal, False otherwise
        :rtype: bool

        """
        self.calculate_relative_projected_gradient(bounds)
        if self.relgrad <= self.epsilon:
            message = f'Relative gradient = {self.relgrad:.2g} <= {self.epsilon:.2g}'
            self.messages['Relative gradient'] = self.relgrad
            self.messages['Cause of termination'] = message
            self.messages[
                'Number of function evaluations'
            ] = self.nbr_function_evaluations()
            self.messages[
                'Number of gradient evaluations'
            ] = self.nbr_gradient_evaluations()
            self.messages[
                'Number of hessian evaluations'
            ] = self.nbr_hessian_evaluations()
            return True
        return False

    def check_insufficient_progress(self, previous_iterate):
        """Verifies if the algorithm is making sufficient progress. If
            not, the algorithm is stopped

        :param previous_iterate: previous iterate
        :type previous_iterate: numpy.array (n x 1)

        :return: True if progress is insufficient, False otherwise
        :rtype: bool

        """
        if self.typx is None:
            self.typx = np.ones(np.asarray(self.x).shape)
        change = relative_change(self.x, previous_iterate, self.typx)
        if change <= self.steptol:
            message = f'Relative change = ' f'{change:.3g} <= {self.steptol:.2g}'
            self.messages['Cause of termination'] = message
            self.messages[
                'Number of function evaluations'
            ] = self.nbr_function_evaluations()
            self.messages[
                'Number of gradient evaluations'
            ] = self.nbr_gradient_evaluations()
            self.messages[
                'Number of hessian evaluations'
            ] = self.nbr_hessian_evaluations()
            return True
        return False

    @final
    def set_variables(self, x):
        """Set the values of the variables for which the function
        has to be calculated.

        :param x: values
        :type x: numpy.array
        """
        self.x = x
        self.x_tuple = tuple(x)

    @final
    def f(self):
        """Retrieve or calculate the value of the function

        :return: value of the function
        :rtype: float
        """
        value = self.stored_values.get(self.x_tuple)
        if value is None:
            value = self._f()
            self.stored_values[self.x_tuple] = value
        return value

    @abstractmethod
    def _f(self):

        """Calculate the value of the function

        :return: value of the function
        :rtype: float
        """

    @final
    def f_g(self):
        """Retrieve or calculate the value of the function and the gradient

        :return: value of the function and the gradient
        :rtype: FunctionData
        """
        function_data = self.stored_gradient.get(self.x_tuple)
        if function_data is None:
            function_data = self._f_g()
            self.stored_gradient[self.x_tuple] = function_data
            self.stored_values[self.x_tuple] = function_data.function
        return function_data

    @abstractmethod
    def _f_g(self):
        """Calculate the value of the function and the gradient

        :return: value of the function and the gradient
        :rtype: FunctionData
        """

    @final
    def f_g_h(self):
        """Calculate the value of the function, the gradient and the Hessian

        :return: value of the function, the gradient and the Hessian
        :rtype: FunctionData
        """
        function_data = self.stored_hessian.get(self.x_tuple)
        if function_data is None:
            function_data = self._f_g_h()
            self.stored_hessian[self.x_tuple] = function_data
            self.stored_gradient[self.x_tuple] = function_data
            self.stored_values[self.x_tuple] = function_data.function
        return function_data

    @abstractmethod
    def _f_g_h(self):
        """Calculate the value of the function, the gradient and the Hessian

        :return: value of the function, the gradient and the Hessian
        :rtype: FunctionData
        """

    @final
    def nbr_function_evaluations(self):
        """Obtain the total number of function evaluations"""
        return len(self.stored_values)

    @final
    def nbr_gradient_evaluations(self):
        """Obtain the total number of gradient evaluations"""
        return len(self.stored_gradient)

    @final
    def nbr_hessian_evaluations(self):
        """Obtain the total number of hessian evaluations"""
        return len(self.stored_hessian)

    def finite_differences_gradient(self, x):
        """Calculates the gradient of the function using finite differences

        :param x: argument of the function
        :type x: numpy.array

        :return: numpy vector, same dimension as x, containing the gradient
        calculated by finite differences.
        :rtype: numpy.array

        """
        x = x.astype(float)
        tau = 0.0000001
        n = len(x)
        g = np.zeros(n)
        self.set_variables(x)
        f = self.f()
        for i in range(n):
            xi = x[i]
            xp = x.copy()
            if abs(xi) >= 1:
                s = tau * xi
            elif xi >= 0:
                s = tau
            else:
                s = -tau
            xp[i] = xi + s
            self.set_variables(xp)
            fp = self.f()
            g[i] = (fp - f) / s
        return g.astype(float)

    def finite_differences_hessian(self, x):
        """Calculates the hessian of the function using finite differences

        :param x: argument of the function
        :type x: numpy.array

        :return: numpy matrix containing the hessian calculated by
             finite differences.
        :rtype: numpy.array

        """
        tau = 1.0e-7
        n = len(x)
        hessian = np.zeros((n, n))
        self.set_variables(x)
        evaluation = self.f_g()
        g = evaluation.gradient
        eye = np.eye(n)
        for i in range(n):
            xi = x[i]
            if abs(xi) >= 1:
                s = tau * xi
            elif xi >= 0:
                s = tau
            else:
                s = -tau
            ei = eye[i]
            self.set_variables(x + s * ei)
            diff_evaluation = self.f_g()
            gp = diff_evaluation.gradient
            hessian[:, i] = (gp - g) / s
        return hessian.astype(float)

    def check_derivatives(self, x, names=None):
        """Verifies the analytical derivatives of a function by comparing
        them with finite difference approximations.

        :param x: arguments of the function
        :type x: numpy.array

        :param names: the names of the entries of x (for reporting).
        :type names: list(string)

        :return: tuple f, g, h, gdiff, hdiff where

          - f is the value of the function at x,
          - g is the analytical gradient,
          - h is the analytical hessian,
          - gdiff is the difference between the analytical gradient
            and the finite difference approximation
          - hdiff is the difference between the analytical hessian
            and the finite difference approximation

        :rtype: float, numpy.array,numpy.array,  numpy.array,numpy.array

        """
        x = np.array(x, dtype=float)
        self.set_variabls(x)
        evaluation = self.f_g_h(x)
        f = evaluation.function
        g = evaluation.gradient
        h = evaluation.hessian
        g_num = self.finite_differences_gradient(x)
        gdiff = g - g_num
        if names is None:
            names = [f'x[{i}]' for i in range(len(x))]
        logger.info('x\t\tGradient\tFinDiff\t\tDifference')
        for k, v in enumerate(gdiff):
            logger.info(f'{names[k]:15}\t{g[k]:+E}\t{g_num[k]:+E}\t{v:+E}')

        h_num = self.finite_differences_hessian(x)
        hdiff = h - h_num
        logger.info('Row\t\tCol\t\tHessian\tFinDiff\t\tDifference')
        for row in range(len(hdiff)):
            for col in range(len(hdiff)):
                logger.info(
                    f'{names[row]:15}\t{names[col]:15}\t{h[row,col]:+E}\t'
                    f'{h_num[row,col]:+E}\t{hdiff[row,col]:+E}'
                )
        return f, g, h, gdiff, hdiff


def relative_gradient(x, f, g, typx, typf):
    """Calculates the relative gradients.

    It is typically used as stopping criterion.

    :param x: current iterate.
    :type x: numpy.array
    :param f: value of f(x)
    :type f: float
    :param g: :math:`\\nabla f(x)`, gradient of f at x
    :type g: numpy.array
    :param typx: typical value for x.
    :type typx: numpy.array
    :param typf: typical value for f.
    :type typf: float

    :return: relative gradient

    .. math:: \\max_{i=1,\\ldots,n}\\frac{(\\nabla f(x))_i
              \\max(x_i,\\text{typx}_i)}
              {\\max(|f(x)|, \\text{typf})}

    :rtype: float
    """
    abs_x_typx = np.maximum(np.abs(x), typx)
    max_abs_f_typf = np.maximum(np.abs(f), typf)

    relgrad = np.abs(g * abs_x_typx / max_abs_f_typf)
    result = np.max(relgrad)

    if np.isfinite(result):
        return result

    return np.finfo(float).max


def relative_change(x, xpred, typx):
    """Calculates the relative change.

    It is typically used as stopping criterion.

    :param x: current iterate.
    :type x: numpy.array

    :param xpred: previous iterate.
    :type xpred: numpy.array

    :param typx: typical value for x.
    :type typx: numpy.array

    :return: relative change:

    .. math:: \\max_i \\frac{|(x_k)_i - (x_{k-1})_i|}{\\max(|(x_k)_i|,
              \\text{typx}_i)}

    :rtype: float

    """

    relx = np.array(
        [abs(x[i] - xpred[i]) / max(abs(x[i]), typx[i]) for i in range(len(x))]
    )
    result = relx.max()
    if np.isfinite(result):
        return result

    return np.finfo(float).max
