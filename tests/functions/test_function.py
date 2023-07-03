"""File test_function.py

:author: Michel Bierlaire
:date: Thu Jun 22 17:08:01 2023

Tests for the function module
"""

import unittest
import numpy as np

from biogeme_optimization.function import relative_gradient
from examples import MyFunctionToMinimize, Example58, Rosenbrock


class RelativeGradientTestCase(unittest.TestCase):
    def test_relative_gradient(self):
        x = np.array([1.0, 2.0, 3.0])
        f = 10.0
        g = np.array([0.1, 0.2, 0.3])
        typx = np.array([0.5, 1.0, 1.5])
        typf = 5.0

        result = relative_gradient(x, f, g, typx, typf)

        # Expected relative gradients for each component
        expected_relgrad = np.array([0.2, 0.4, 0.6])
        # Expected maximum relative gradient
        expected_result = 0.09

        # Perform assertions to validate the output
        np.testing.assert_allclose(result, expected_result, rtol=1e-6)

    def test_relative_gradient_finite_result(self):
        x = np.array([1.0, 2.0, 3.0])
        f = 10.0
        g = np.array([0.1, 0.2, 0.3])
        typx = np.array([0.5, 1.0, 1.5])
        typf = 5.0

        # Set one component of g to infinity
        g[1] = np.inf

        result = relative_gradient(x, f, g, typx, typf)

        # Expected result when one component of g is infinite
        expected_result = np.finfo(float).max

        # Perform assertions to validate the output
        self.assertEqual(result, expected_result)


class TestFiniteDifferencesGradient(unittest.TestCase):
    def test_finite_differences_gradient(self):

        instances = (MyFunctionToMinimize(dimension=2), Example58(), Rosenbrock())

        for instance in instances:
            # Test case
            x = np.array([1, 2])
            instance.set_variables(x)
            evaluation = instance.f_g()
            expected_gradient = evaluation.gradient

            # Call the function being tested
            fin_diff_gradient = instance.finite_differences_gradient(x)

            # Assertion
            np.testing.assert_array_almost_equal(
                expected_gradient, fin_diff_gradient, decimal=3
            )


class TestFiniteDifferencesHessian(unittest.TestCase):
    def test_finite_differences_hessian(self):
        # Create a mock class that mimics the behavior of the original class
        instances = (MyFunctionToMinimize(dimension=2), Example58(), Rosenbrock())

        for instance in instances:
            # Test case
            x = np.array([1, 2])
            instance.set_variables(x)
            evaluation = instance.f_g_h()
            expected_hessian = evaluation.hessian

            # Call the function being tested
            fin_diff_hessian = instance.finite_differences_hessian(x)

            # Assertion
            np.testing.assert_array_almost_equal(
                expected_hessian, fin_diff_hessian, decimal=3
            )


if __name__ == '__main__':
    unittest.main()
