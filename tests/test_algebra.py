"""File test_algebra.py

:author: Michel Bierlaire
:date: Thu Jun 22 12:36:34 2023

Tests for the algebra module
"""
import unittest
import numpy as np
from biogeme_optimization.exceptions import (
    OptimizationError,
)  # Import the required exception class
from biogeme_optimization.algebra import (
    schnabel_eskow,
    schnabel_eskow_direction,
)


class SchnabelEskowTestCase(unittest.TestCase):
    def test_schnabelEskow(self):
        A = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
        L, E, P = schnabel_eskow(A)

        # Verify that L is lower triangular
        self.assertTrue(np.all(np.tril(L) == L))

        # Verify that A + E = PLL^TP^T
        result = P @ L @ L.T @ P.T + E
        np.testing.assert_array_almost_equal(A, result, decimal=5)

        # Test case 1: Identity matrix
        A = np.eye(3)
        L, E, P = schnabel_eskow(A)
        # Verify that L is lower triangular
        self.assertTrue(np.all(np.tril(L) == L))

        expected_L = np.eye(3)
        expected_E = np.zeros(3)
        expected_P = np.eye(3)
        self.assertTrue(np.allclose(L, expected_L))
        self.assertTrue(np.allclose(E, expected_E))
        self.assertTrue(np.allclose(P, expected_P))

        # Verify that A + E = PLL^TP^T
        result = P @ L @ L.T @ P.T + E
        np.testing.assert_array_almost_equal(A, result, decimal=5)

        # Test case 2: Positive definite matrix
        A = np.array([[4, 2, 1], [2, 5, 2], [1, 2, 6]])
        L, E, P = schnabel_eskow(A)
        # Verify that L is lower triangular
        self.assertTrue(np.all(np.tril(L) == L))

        # Verify that A + E = PLL^TP^T
        result = P @ L @ L.T @ P.T + E
        np.testing.assert_array_almost_equal(A, result, decimal=5)

        # Test case 3: Non-positive definite matrix
        A = np.array([[1, 1, 1], [1, 2, 2], [1, 2, 3]])
        L, E, P = schnabel_eskow(A)
        # Verify that L is lower triangular
        self.assertTrue(np.all(np.tril(L) == L))

        # Verify that A + E = PLL^TP^T
        result = P @ L @ L.T @ P.T + E
        np.testing.assert_array_almost_equal(A, result, decimal=5)

        # Test case 4: Larger matrix
        A = np.array([[2, -1, 0, 0], [-1, 2, -1, 0], [0, -1, 2, -1], [0, 0, -1, 2]])
        L, E, P = schnabel_eskow(A)
        # Verify that L is lower triangular
        self.assertTrue(np.all(np.tril(L) == L))

        # Verify that A + E = PLL^TP^T
        result = P @ L @ L.T @ P.T + E
        np.testing.assert_array_almost_equal(A, result, decimal=5)

        A = np.array(
            [
                [0.3571, -0.1030, 0.0274, -0.0459],
                [-0.1030, 0.2525, 0.0736, -0.3845],
                [0.0274, 0.0736, 0.2340, -0.2878],
                [-0.0459, -0.3845, -0.2878, 0.5549],
            ]
        )
        L, E, P = schnabel_eskow(A)
        self.assertAlmostEqual(L[0][0], 0.7449161, 5)
        diff = P @ L @ L.T @ P.T - E - A
        for i in list(diff.flatten()):
            self.assertAlmostEqual(i, 0.0, 5)

        A = np.array(
            [
                [1890.3, -1705.6, -315.8, 3000.3],
                [-1705.6, 1538.3, 284.9, -2706.6],
                [-315.8, 284.9, 52.5, -501.2],
                [3000.3, -2706.6, -501.2, 4760.8],
            ]
        )
        L, E, P = schnabel_eskow(A)
        self.assertAlmostEqual(L[0][0], 6.89985507e01, 5)
        diff = P @ L @ L.T @ P.T - E - A
        for i in list(diff.flatten()):
            self.assertAlmostEqual(i, 0.0, 5)

    def test_schnabel_eskow_exceptions(self):
        # Test case 1: Non-square matrix
        A = np.array([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(OptimizationError):
            schnabel_eskow(A)

        # Test case 2: Non-symmetric matrix
        A = np.array([[1, 2, 3], [2, 4, 5], [13, 5, 6]])
        with self.assertRaises(OptimizationError):
            schnabel_eskow(A)


import unittest
import numpy as np
from scipy.linalg import cholesky, solve_triangular


class SchnabelEskowDirectionTestCase(unittest.TestCase):
    def test_schnabel_eskow(self):
        # Define input matrix A
        A = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        # Call the schnabel_eskow function
        L, E, P = schnabel_eskow(A)
        # Define the expected results
        expected_L = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        expected_E = np.zeros_like(expected_L)
        expected_P = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        # Compare the results with the expected values
        self.assertTrue(np.allclose(L, expected_L))
        self.assertTrue(np.allclose(E, expected_E))
        self.assertTrue(np.allclose(P, expected_P))

    def test_schnabel_eskow_complex(self):
        expected_L = np.array(
            [
                [8.71779789, 0.0, 0.0],
                [0.45883147, 0.37674965, 0.0],
                [2.63828094, 2.09548617, 0.02963117],
            ]
        )
        expected_E = np.array(
            [[0.0, 0.0, 0.0], [0.0, 1.35246662, 0.0], [0.0, 0.0, 1.35246662]]
        )
        expected_P = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        A = expected_P @ expected_L @ expected_L.T @ expected_P.T - expected_E
        L, E, P = schnabel_eskow(A)
        # Compare the results with the expected values
        self.assertTrue(np.allclose(L, expected_L))
        self.assertTrue(np.allclose(E, expected_E))
        self.assertTrue(np.allclose(P, expected_P))

    def test_schnabel_eskow_non_convex(self):
        # Define input matrix A
        A = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])

        # Call the schnabel_eskow function
        L, E, P = schnabel_eskow(A)

        # Define the expected results
        positive_values_exist = np.any(E > 0)
        self.assertTrue(positive_values_exist)

    def test_schnabel_eskow_direction(self):
        gradient = np.array([-2.0, 3.0, -1.0])
        hessian = np.array([[4.0, -1.0, 2.0], [-1.0, 6.0, -4.0], [2.0, -4.0, 7.0]])

        result = schnabel_eskow_direction(gradient, hessian)
        # Expected descent direction computed manually
        expected_direction = np.array([0.52808989, -0.6741573, -0.39325843])

        # Perform assertions to validate the output
        np.testing.assert_allclose(result, expected_direction, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
