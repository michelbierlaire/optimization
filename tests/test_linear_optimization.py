"""Test LinearOptimization class

Michel Bierlaire
Mon Apr 1 14:47:49 2024

"""

import unittest

import numpy as np

from biogeme_optimization.teaching.linear_optimization import LinearOptimization


class TestLinearOptimization(unittest.TestCase):
    def setUp(self):
        # Example correct inputs
        self.objective = np.array([1, 2, 3, 4])
        self.constraint_matrix = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
        self.right_hand_side = np.array([4, 5, 6])

        # Instantiate the class with correct inputs for reuse
        self.optimization_problem = LinearOptimization(
            self.objective, self.constraint_matrix, self.right_hand_side
        )

    def test_initialization_success(self):
        """Test successful class initialization."""
        self.assertIsInstance(self.optimization_problem, LinearOptimization)

    def test_initialization_failure_due_to_incompatible_sizes(self):
        """Test failure due to incompatible sizes between objective and constraint matrix."""
        with self.assertRaises(ValueError):
            _ = LinearOptimization(
                np.array([1, 2]), self.constraint_matrix, self.right_hand_side
            )

    def test_initialization_failure_due_to_wrong_input_types(self):
        """Test initialization failure due to wrong input types."""
        with self.assertRaises(AttributeError):
            _ = LinearOptimization(
                'not_an_array', 'also_not_an_array', 'still_not_an_array'
            )

    def test_n_variables(self):
        """Test the n_variables property."""
        self.assertEqual(self.optimization_problem.n_variables, 4)

    def test_n_constraints(self):
        """Test the n_constraints property."""
        self.assertEqual(self.optimization_problem.n_constraints, 3)

    def test_basic_indices_setter_and_getter(self):
        """Test setting and getting basic indices."""
        self.optimization_problem.basic_indices = [0, 1, 2]
        self.assertEqual(self.optimization_problem.basic_indices, [0, 1, 2])

    def test_reduced_cost_calculation(self):
        """Test the calculation of reduced cost."""
        self.optimization_problem.basic_indices = [
            0,
            1,
            2,
        ]
        reduced_cost = self.optimization_problem.reduced_cost(variable_index=3)
        expected_value = -2
        self.assertEqual(reduced_cost, expected_value)
        basic_direction = self.optimization_problem.constraints.basic_direction(
            non_basic_index=3
        )
        second_expected_value = np.inner(
            self.optimization_problem.objective, basic_direction
        )
        self.assertEqual(reduced_cost, second_expected_value)

    def test_reduced_cost_without_basic_indices(self):
        """Test reduced cost calculation without basic indices set."""
        optimization_problem = LinearOptimization(
            self.objective, self.constraint_matrix, self.right_hand_side
        )
        with self.assertRaises(EnvironmentError):
            _ = optimization_problem.reduced_cost(variable_index=2)


if __name__ == '__main__':
    unittest.main()
