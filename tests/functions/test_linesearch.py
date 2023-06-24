"""File test_linesearch.py

:author: Michel Bierlaire
:date: Thu Jun 22 16:04:39 2023

Tests for the linesearch module
"""
import unittest
import numpy as np
from biogeme_optimization.exceptions import (
    OptimizationError,
)  # Import the required exception class
from biogeme_optimization.linesearch import (
    linesearch,
    newton_linesearch,
)
from biogeme_optimization.function import FunctionToMinimize


class my_function_to_minimize(FunctionToMinimize):
    def __init__(self):
        self.x = None

    def set_variables(self, x):
        """Set the values of the variables for which the function
        has to be calculated.

        :param x: values
        :type x: numpy.array
        """
        self.x = x

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
        half_sum_of_squares = np.sum(np.square(self.x)) / 2
        return half_sum_of_squares

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
        half_sum_of_squares = np.sum(np.square(self.x)) / 2
        return half_sum_of_squares, self.x

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
        half_sum_of_squares = np.sum(np.square(self.x)) / 2
        return half_sum_of_squares, self.x, np.eye(len(self.x))

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
        raise OptimizationError('BHHH is irrelevant in this context')


class LineSearchTestCase(unittest.TestCase):
    def test_linesearch(self):
        fct = my_function_to_minimize()
        iterate = np.array([1.0, 1.0])
        descent_direction = np.array([-1.0, -1.0])
        alpha0 = 1.0
        beta1 = 1.0e-4
        beta2 = 0.99
        lbd = 2.0

        result = linesearch(fct, iterate, descent_direction, alpha0, beta1, beta2, lbd)
        alpha, nbr_function_evaluations = result

        # Perform assertions to validate the output
        self.assertIsInstance(alpha, float)
        self.assertEqual(alpha, 1)
        self.assertIsInstance(nbr_function_evaluations, int)
        self.assertGreater(nbr_function_evaluations, 0)

    def test_linesearch_wolfe_1(self):
        fct = my_function_to_minimize()
        iterate = np.array([1.0, 1.0])
        descent_direction = np.array([-1.0, -1.0])
        alpha0 = 1.0
        beta1 = 0.98
        beta2 = 0.99
        lbd = 2.0

        result = linesearch(fct, iterate, descent_direction, alpha0, beta1, beta2, lbd)
        alpha, nbr_function_evaluations = result

        # Perform assertions to validate the output
        self.assertIsInstance(alpha, float)
        self.assertEqual(alpha, 0.03125)
        self.assertIsInstance(nbr_function_evaluations, int)
        self.assertGreater(nbr_function_evaluations, 0)

    def test_linesearch_wolfe_2(self):
        fct = my_function_to_minimize()
        iterate = np.array([1.0, 1.0])
        descent_direction = np.array([-1.0, -1.0])
        alpha0 = 1.0e-10
        beta1 = 1.0e-5
        beta2 = 1.0e-4
        lbd = 2.0

        result = linesearch(fct, iterate, descent_direction, alpha0, beta1, beta2, lbd)
        alpha, nbr_function_evaluations = result

        # Perform assertions to validate the output
        self.assertIsInstance(alpha, float)
        self.assertEqual(alpha, 1.7179869184)
        self.assertIsInstance(nbr_function_evaluations, int)
        self.assertGreater(nbr_function_evaluations, 0)

    def test_linesearch_lambda_less_than_one(self):
        fct = my_function_to_minimize()
        iterate = np.array([1.0, 1.0])
        descent_direction = np.array([-1.0, -1.0])
        alpha0 = 1.0
        beta1 = 1.0e-4
        beta2 = 0.99
        lbd = 0.5

        with self.assertRaises(OptimizationError):
            linesearch(fct, iterate, descent_direction, alpha0, beta1, beta2, lbd)

    def test_linesearch_alpha0_less_than_zero(self):
        fct = my_function_to_minimize()
        iterate = np.array([1.0, 1.0])
        descent_direction = np.array([-1.0, -1.0])
        alpha0 = -1.0
        beta1 = 1.0e-4
        beta2 = 0.99
        lbd = 2.0

        with self.assertRaises(OptimizationError):
            linesearch(fct, iterate, descent_direction, alpha0, beta1, beta2, lbd)

    def test_linesearch_beta1_greater_than_beta2(self):
        fct = my_function_to_minimize()
        iterate = np.array([1.0, 1.0])
        descent_direction = np.array([-1.0, -1.0])
        alpha0 = 1.0
        beta1 = 0.99
        beta2 = 0.5
        lbd = 2.0

        with self.assertRaises(OptimizationError):
            linesearch(fct, iterate, descent_direction, alpha0, beta1, beta2, lbd)

    def test_linesearch_non_descent_direction(self):
        fct = my_function_to_minimize()
        iterate = np.array([1.0, 1.0])
        descent_direction = np.array([1.0, 1.0])
        alpha0 = 1.0
        beta1 = 1.0e-4
        beta2 = 0.99
        lbd = 2.0

        with self.assertRaises(OptimizationError):
            linesearch(fct, iterate, descent_direction, alpha0, beta1, beta2, lbd)

    def test_newton_linesearch(self):
        fct = my_function_to_minimize()

        # Set the starting point
        starting_point = np.array([2.0, 2.0])

        # Set the expected solution
        expected_solution = np.array([0.0, 0.0])

        # Call the newton_linesearch function
        solution, messages = newton_linesearch(fct, starting_point)

        # Check the solution
        np.testing.assert_allclose(solution, expected_solution, atol=1e-6)

        # Check the termination message
        self.assertEqual(
            messages['Cause of termination'], 'Relative gradient = 0 <= 6.1e-06'
        )


if __name__ == '__main__':
    unittest.main()
