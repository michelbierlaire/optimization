"""File function.py

:author: Michel Bierlaire
:date: Thu Jun 22 16:42:46 2023

Abstract class for the function to minimize
"""
from abc import abstractmethod
import numpy as np


class FunctionToMinimize:
    """This is an abstract class. The actual function to minimize
    must be implemented in a concrete class deriving from this one.

    """

    @abstractmethod
    def set_variables(self, x):
        """Set the values of the variables for which the function
        has to be calculated.

        :param x: values
        :type x: numpy.array
        """

    @abstractmethod
    def f(self, batch=None):
        """Calculate the value of the function

        :param batch: for data driven functions (such as a log
                      likelikood function), it is possible to
                      approximate the value of the function using a
                      sample of the data called a batch. This argument
                      is a value between 0 and 1 representing the
                      percentage of the data that should be used for
                      thre random batch. If None, the full data set is
                      used. Default: None pass
        :type batch: float

        :return: value of the function
        :rtype: float
        """

    @abstractmethod
    def f_g(self, batch=None):
        """Calculate the value of the function and the gradient

        :param batch: for data driven functions (such as a log
                      likelikood function), it is possible to
                      approximate the value of the function using a
                      sample of the data called a batch. This argument
                      is a value between 0 and 1 representing the
                      percentage of the data that should be used for
                      the random batch. If None, the full data set is
                      used. Default: None pass
        :type batch: float

        :return: value of the function and the gradient
        :rtype: tuple float, numpy.array
        """

    @abstractmethod
    def f_g_h(self, batch=None):
        """Calculate the value of the function, the gradient and the Hessian

        :param batch: for data driven functions (such as a log
                      likelikood function), it is possible to
                      approximate the value of the function using a
                      sample of the data called a batch. This argument
                      is a value between 0 and 1 representing the
                      percentage of the data that should be used for
                      the random batch. If None, the full data set is
                      used. Default: None pass
        :type batch: float

        :return: value of the function, the gradient and the Hessian
        :rtype: tuple float, numpy.array, numpy.array
        """

    @abstractmethod
    def f_g_bhhh(self, batch=None):
        """Calculate the value of the function, the gradient and
        the BHHH matrix

        :param batch: for data driven functions (such as a log
                      likelikood function), it is possible to
                      approximate the value of the function using a
                      sample of the data called a batch. This argument
                      is a value between 0 and 1 representing the
                      percentage of the data that should be used for
                      the random batch. If None, the full data set is
                      used. Default: None pass
        :type batch: float

        :return: value of the function, the gradient and the BHHH
        :rtype: tuple float, numpy.array, numpy.array
        """


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
