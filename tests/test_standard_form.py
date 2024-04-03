"""Tests for the optimization problem in standard form

Michel Bierlaire
Sat Mar 30 18:18:29 2024

"""

import unittest
from unittest.mock import patch

import numpy as np
from biogeme_optimization.teaching.linear_constraints import StandardForm


class TestStandardForm(unittest.TestCase):

    def setUp(self):
        """Set up a StandardForm object with a predefined matrix and vector."""
        self.matrix = np.array([[1, 2, 0], [0, 1, -1]])
        self.vector = np.array([3, 1])
        self.problem = StandardForm(self.matrix, self.vector)
        self.problem.basic_indices = [0, 1]

    def test_init_valid_input(self):
        """Test initialization with valid input."""
        self.assertIsInstance(self.problem, StandardForm)
        self.assertEqual(self.problem.n_constraints, 2)
        self.assertEqual(self.problem.n_variables, 3)

    def test_init_dimension_mismatch(self):
        """Test initialization with dimension mismatch between matrix and vector."""
        matrix = np.array([[1, 2], [3, 4]])
        vector = np.array([5])
        with self.assertRaises(ValueError):
            StandardForm(matrix, vector)

    def test_basic_indices_setter_valid(self):
        """Test setting valid basic indices."""
        matrix = np.array([[1, 2], [3, 4]])
        vector = np.array([5, 6])
        problem = StandardForm(matrix, vector)
        problem.basic_indices = [0, 1]
        self.assertEqual(problem.basic_indices, [0, 1])

    def test_basic_indices_setter_invalid_length(self):
        """Test setting basic indices with invalid length."""
        matrix = np.array([[1, 2], [3, 4]])
        vector = np.array([5, 6])
        problem = StandardForm(matrix, vector)
        with self.assertRaises(ValueError):
            problem.basic_indices = [0]

    def test_basic_indices_setter_out_of_range(self):
        """Test setting basic indices with out-of-range index."""
        matrix = np.array([[1, 2], [3, 4]])
        vector = np.array([5, 6])
        problem = StandardForm(matrix, vector)
        with self.assertRaises(ValueError):
            problem.basic_indices = [0, 2]

    def test_basic_matrix(self):
        """Test extraction of the basic matrix."""
        matrix = np.array([[1, 2], [3, 4]])
        vector = np.array([5, 6])
        problem = StandardForm(matrix, vector)
        problem.basic_indices = [0, 1]
        np.testing.assert_array_equal(problem.basic_matrix, matrix)

    def test_basic_part_basic_solution(self):
        """Test calculation of the basic part of the solution."""
        matrix = np.array([[1, 0], [0, 1]])
        vector = np.array([5, 6])
        problem = StandardForm(matrix, vector)
        problem.basic_indices = [0, 1]
        np.testing.assert_array_equal(problem.basic_part_basic_solution, vector)

    def test_feasible_basic_solution(self):
        """Test checking of the feasibility of the basic solution."""
        matrix = np.array([[1, 0], [0, 1]])
        vector = np.array([5, 6])
        problem = StandardForm(matrix, vector)
        problem.basic_indices = [0, 1]
        self.assertTrue(problem.is_basic_solution_feasible)

    def test_basic_solution_full_dimension(self):
        """Test the basic solution in the full dimension."""
        matrix = np.array([[1, 0], [0, 1]])
        vector = np.array([5, 6])
        problem = StandardForm(matrix, vector)
        problem.basic_indices = [0, 1]
        expected_solution = np.array([5, 6])
        np.testing.assert_array_equal(problem.basic_solution, expected_solution)

    def test_basic_direction_with_valid_non_basic_index(self):
        """Test basic_direction with a valid non-basic index."""
        non_basic_index = 2
        expected_direction = np.array([-2, 1, 1])
        np.testing.assert_array_almost_equal(
            self.problem.basic_direction(non_basic_index), expected_direction
        )

    def test_basic_direction_with_negative_index(self):
        """Test basic_direction with a negative non-basic index."""
        with self.assertRaises(ValueError):
            self.problem.basic_direction(-1)

    def test_basic_direction_with_out_of_range_index(self):
        """Test basic_direction with a non-basic index out of range."""
        with self.assertRaises(ValueError):
            self.problem.basic_direction(self.problem.n_variables)  # Out of range index

    def test_basic_direction_with_basic_index(self):
        """Test basic_direction with a basic index."""
        with self.assertRaises(ValueError):
            self.problem.basic_direction(self.problem.basic_indices[0])  # A basic index

    def test_basic_direction_without_basis_selected(self):
        """Test basic_direction when no basis has been selected."""
        matrix = np.array([[1, 0], [0, 1]])
        vector = np.array([5, 6])
        problem = StandardForm(matrix, vector)
        with patch('logging.Logger.warning') as mock_warning:
            self.assertIsNone(problem.basic_direction(2))
            mock_warning.assert_called_once_with('No basis has been selected.')

    def test_basic_direction_with_singular_basic_matrix(self):
        """Test basic_direction when the basic matrix is singular."""
        # Making the basic matrix singular by setting dependent rows
        matrix = np.array([[1, 0, 1], [1, 0, 1]])
        vector = np.array([3, 6])
        problem = StandardForm(matrix=matrix, vector=vector)
        problem.basic_indices = [0, 1]
        self.assertIsNone(problem.basic_direction(2))  # Assuming column 2 is non-basic


# This allows the test script to be run directly
if __name__ == '__main__':
    unittest.main()
