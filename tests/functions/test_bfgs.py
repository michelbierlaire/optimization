"""File test_bfgs.py

:author: Michel Bierlaire
:date: Sat Jun 24 17:44:39 2023

Tests for the bfgs module
"""
import unittest
import numpy as np
import scipy.linalg as la
from biogeme_optimization.exceptions import (
    OptimizationError,
)  # Import the required exception class
from biogeme_optimization.bfgs import (
    bfgs,
    inverse_bfgs,
)
from biogeme_optimization.function import FunctionToMinimize


class BfgsTest(unittest.TestCase):
    def test_bfgs_1(self):
        # Test case 1: Basic example
        hessian_approx = np.array([[2, 0], [0, 2]])
        delta_x = np.array([1, 1])
        delta_g = np.array([1, 1])

        expected_result = np.array([[1.5, -0.5], [-0.5, 1.5]])

        result = bfgs(hessian_approx, delta_x, delta_g)
        np.testing.assert_array_almost_equal(result, expected_result, decimal=6)

    def test_bfgs_2(self):
        # Test case 2: Larger matrices
        hessian_approx = np.array([[2, 1, 0], [1, 3, -1], [0, -1, 4]])
        delta_x = np.array([1, -1, 1])
        delta_g = np.array([4, 0, -1])

        expected_result = np.array(
            [
                [7.22222222, 1.33333333, -1.88888888],
                [1.33333333, 2.0, 0.66666666],
                [-1.88888888, 0.66666666, 1.55555555],
            ]
        )

        result = bfgs(hessian_approx, delta_x, delta_g)
        np.testing.assert_array_almost_equal(result, expected_result, decimal=6)

    def test_bfgs_3(self):
        # Test case 3:delta_x and delta_g are orthogonal
        hessian_approx = np.array([[2, 0], [0, 2]])
        delta_x = np.array([1, -1])
        delta_g = np.array([-1, 1])

        expected_result = np.array([[1.2, 0.8], [0.8, 1.2]])

        result = bfgs(hessian_approx, delta_x, delta_g)
        np.testing.assert_array_almost_equal(result, expected_result, decimal=6)

    def test_inverse_bfgs_1(self):
        # Test case 1: Basic example
        hessian_approx = np.array([[2, 0], [0, 2]])
        inverse_hessian_approx = np.linalg.inv(hessian_approx)
        delta_x = np.array([1, 1])
        delta_g = np.array([1, 1])

        expected_result_bfgs = np.array([[1.5, -0.5], [-0.5, 1.5]])
        expected_result = np.linalg.inv(expected_result_bfgs)

        result = inverse_bfgs(inverse_hessian_approx, delta_x, delta_g)
        np.testing.assert_array_almost_equal(result, expected_result, decimal=6)

    def test_inverse_bfgs_2(self):
        # Test case 2: Larger matrices
        hessian_approx = np.array([[2, 1, 0], [1, 3, -1], [0, -1, 4]])
        inverse_hessian_approx = np.linalg.inv(hessian_approx)
        delta_x = np.array([1, -1, 1])
        delta_g = np.array([4, 0, -1])

        expected_result_bfgs = np.array(
            [
                [7.22222222, 1.33333333, -1.88888888],
                [1.33333333, 2.0, 0.66666666],
                [-1.88888888, 0.66666666, 1.55555555],
            ]
        )
        expected_result = np.linalg.inv(expected_result_bfgs)
        result = inverse_bfgs(inverse_hessian_approx, delta_x, delta_g)
        np.testing.assert_array_almost_equal(result, expected_result, decimal=6)

    def test_inverse_bfgs_3(self):
        # Test case 3:delta_x and delta_g are orthogonal. Note that,
        # in that case, the "inverse" formula is not valid.
        hessian_approx = np.array([[2, 0], [0, 2]])
        inverse_hessian_approx = np.linalg.inv(hessian_approx)
        delta_x = np.array([1, -1])
        delta_g = np.array([-1, 1])

        with self.assertRaises(OptimizationError):
            result = inverse_bfgs(inverse_hessian_approx, delta_x, delta_g)


if __name__ == '__main__':
    unittest.main()
