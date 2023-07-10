"""File test_hybrid_function.py

:author: Michel Bierlaire
:date: Sun Jul  2 16:14:20 2023

Tests for the hybrid_function module
"""
import numpy as np
import unittest
from biogeme_optimization.bfgs import bfgs
from biogeme_optimization.bounds import Bounds
from biogeme_optimization.function import FunctionData
from biogeme_optimization.hybrid_function import HybridFunction
from examples import MyFunctionToMinimize


class TestHybridFunction(unittest.TestCase):
    def setUp(self):
        # Create an instance of MyFunctionToMinimize with dimension 3
        self.dimension = 3
        self.function = MyFunctionToMinimize(self.dimension)
        self.proportion = 0.5
        self.bounds = Bounds(
            [
                (None, None),
                (None, None),
                (None, None),
            ]
        )
        self.hybrid_func = HybridFunction(self.function, self.proportion, self.bounds)

    def test_calculate_function_and_derivatives_analytical(self):
        # Test when analytical Hessian is used
        iterate = np.array([1, 2, 3])
        function_data = self.hybrid_func._calculate_function_and_derivatives_analytical(
            iterate
        )
        expected_function_data = FunctionData(
            function=7,
            gradient=iterate,
            hessian=np.eye(self.dimension),
        )
        self.assertEqual(function_data.function, expected_function_data.function)
        np.testing.assert_array_almost_equal(
            function_data.gradient, expected_function_data.gradient
        )
        np.testing.assert_array_almost_equal(
            function_data.hessian, expected_function_data.hessian
        )
        self.assertEqual(self.hybrid_func.number_of_analytical_hessians, 1)
        self.assertEqual(self.hybrid_func.number_of_matrices, 1)
        # Add more assertions for the specific behavior and expected values

    def test_calculate_function_and_derivatives_bfgs(self):
        # Test when BFGS approximation is used
        iterate = np.array([1, 2, 3])
        self.function.x = iterate
        function_data = self.hybrid_func._calculate_function_and_derivatives_bfgs(
            iterate
        )
        expected_function_data = FunctionData(
            function=7,
            gradient=iterate,
            hessian=np.eye(self.dimension),
        )
        self.assertEqual(function_data.function, expected_function_data.function)
        np.testing.assert_array_almost_equal(
            function_data.gradient, expected_function_data.gradient
        )
        np.testing.assert_array_almost_equal(
            function_data.hessian, expected_function_data.hessian
        )
        self.assertEqual(self.hybrid_func.number_of_matrices, 1)

        iterate = np.array([0, 0, 0])
        self.function.x = iterate
        function_data = self.hybrid_func._calculate_function_and_derivatives_bfgs(
            iterate
        )
        expected_function_data = FunctionData(
            function=0,
            gradient=iterate,
            hessian=np.eye(self.dimension),
        )
        # The solution is optimal
        self.assertIsNone(function_data)

    def test_calculate_function_and_derivatives(self):
        # Test when analytical Hessian is used
        iterate = np.array([1, 2, 3])
        self.function.x = iterate
        self.hybrid_func.calculate_function_and_derivatives(iterate)
        self.assertEqual(self.hybrid_func.number_of_analytical_hessians, 1)
        self.assertEqual(self.hybrid_func.number_of_matrices, 1)
        # Add more assertions for the specific behavior and expected values

        # Test when BFGS approximation is used
        iterate = np.array([4, 5, 6])
        self.function.x = iterate
        self.hybrid_func.calculate_function_and_derivatives(iterate)
        self.assertEqual(self.hybrid_func.number_of_matrices, 2)
        # Add more assertions for the specific behavior and expected values

    def test_message(self):
        # Test when there are no numerical issues
        self.hybrid_func.number_of_analytical_hessians = 2
        self.hybrid_func.number_of_matrices = 5
        self.assertEqual(self.hybrid_func.message(), '2/5 = 40.0%')

        # Test when there are numerical issues
        self.hybrid_func.number_of_numerical_issues = 1
        self.assertEqual(self.hybrid_func.message(), '2/5 = 40.0% [1 numerical issues]')

        # Test when there are no matrices involved
        self.hybrid_func.number_of_matrices = 0
        self.assertEqual(self.hybrid_func.message(), '')


if __name__ == '__main__':
    unittest.main()
