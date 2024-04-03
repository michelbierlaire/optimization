"""Test the simplex tableau

Michel Bierlaire
Wed Apr 3 13:23:11 2024
"""

import unittest
import numpy as np
from biogeme_optimization.teaching.tableau import SimplexTableau, RowColumn


class TestSimplexTableau(unittest.TestCase):
    def setUp(self):
        # Define a known tableau for testing
        self.tableau = np.array(
            [
                [0, 1.5, 1, 1, -0.5, 0, 10],
                [1, 0.5, 1, 0, 0.5, 0, 10],
                [0, 1, -1, 0, -1, 1, 0],
                [0, -7, -2, 0, 5, 0, 100],
            ]
        )
        self.simplex_tableau = SimplexTableau(self.tableau)

    def test_initialization(self):
        """Test correct initialization and attributes."""
        self.assertEqual(self.simplex_tableau.n_rows, 4)
        self.assertEqual(self.simplex_tableau.n_columns, 7)
        self.assertEqual(self.simplex_tableau.n_variables, 6)
        self.assertEqual(self.simplex_tableau.n_constraints, 3)

    def test_feasible_basic_solution(self):
        """Test the feasible basic solution calculation."""
        solution = self.simplex_tableau.feasible_basic_solution
        expected_solution = np.zeros(6)
        expected_solution[self.simplex_tableau.basic_indices] = self.tableau[:3, -1]
        np.testing.assert_array_equal(solution, expected_solution)

    def test_value_objective_function(self):
        """Test the objective function value calculation."""
        objective_value = self.simplex_tableau.value_objective_function
        self.assertEqual(objective_value, -100)

    def test_reduced_costs(self):
        """Test the reduced costs calculation."""
        reduced_costs = self.simplex_tableau.reduced_costs
        expected_results = {1: -7.0, 2: -2.0, 4: 5.0}
        self.assertDictEqual(reduced_costs, expected_results)

    def test_pivoting_invalid_row(self):
        """Test pivoting with an invalid row index."""
        with self.assertRaises(ValueError):
            self.simplex_tableau.pivoting(RowColumn(row=10, column=1))

    def test_pivoting_invalid_column(self):
        """Test pivoting with an invalid column index."""
        with self.assertRaises(ValueError):
            self.simplex_tableau.pivoting(RowColumn(row=1, column=10))

    def test_pivoting_zero_pivot(self):
        """Test pivoting with a pivot value too close to zero."""
        with self.assertRaises(ValueError):
            self.simplex_tableau.pivoting(RowColumn(row=0, column=0))

    def test_pivoting_success(self):
        """Test successful pivoting."""
        pivot = RowColumn(row=2, column=1)
        try:
            self.simplex_tableau.pivoting(pivot)
        except ValueError as e:
            self.fail(f"Pivoting raised an unexpected ValueError: {e}")

        expected_result = np.array(
            [
                [0, 0, 2.5, 1, 1, -1.5, 10],
                [1, 0, 1.5, 0, 1, -0.5, 10],
                [0, 1, -1, 0, -1, 1, 0],
                [0, 0, -9, 0, -2, 7, 100],
            ]
        )
        np.testing.assert_array_equal(self.simplex_tableau.tableau, expected_result)


if __name__ == '__main__':
    unittest.main()
