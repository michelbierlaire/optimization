"""Test simplex algorithm

Michel Bierlaire
Tue Apr 2 16:48:13 2024

"""

import unittest
import numpy as np
from biogeme_optimization.teaching.simplex import (
    SimplexAlgorithm,
    CauseInterruptionIterations,
    LinearOptimization,
)


class TestSimplexAlgorithm(unittest.TestCase):

    def test_initialization(self):
        """Test correct initialization of the SimplexAlgorithm class."""
        standard_a = np.array([[2, 1, 1, 0], [1, -1, 0, 1]])
        standard_b = np.array([6, 2])
        standard_c = np.array([-4, 3, 0, 0])
        optimization_problem = LinearOptimization(
            objective=standard_c,
            constraint_matrix=standard_a,
            right_hand_side=standard_b,
        )
        initial_basis = [2, 3]  # Example basis
        algo = SimplexAlgorithm(optimization_problem, initial_basis)
        self.assertIsInstance(algo, SimplexAlgorithm)

    def test_invalid_initial_basis(self):
        """Test initialization with an invalid initial basis."""
        standard_a = np.array([[2, 1, 1, 0], [1, -1, 0, 1]])
        standard_b = np.array([6, 2])
        standard_c = np.array([-4, 3, 0, 0])
        optimization_problem = LinearOptimization(
            objective=standard_c,
            constraint_matrix=standard_a,
            right_hand_side=standard_b,
        )
        initial_basis = [2, 3, 4]  # Example basis
        with self.assertRaises(ValueError):
            SimplexAlgorithm(optimization_problem, initial_basis)

    def test_infeasible_starting_solution(self):
        """Test handling of an infeasible starting solution."""
        # This test assumes the existence of an `is_basic_solution_feasible` property
        # You'll need to mock or modify LinearOptimization to return False for `is_basic_solution_feasible`
        standard_a = np.array([[2, 1, 1, 0], [1, -1, 0, 1]])
        standard_b = np.array([6, 2])
        standard_c = np.array([-4, 3, 0, 0])
        optimization_problem = LinearOptimization(
            objective=standard_c,
            constraint_matrix=standard_a,
            right_hand_side=standard_b,
        )
        initial_basis = [1, 2]  # Example of infeasible basis
        with self.assertRaises(ValueError):
            SimplexAlgorithm(optimization_problem, initial_basis)

    def test_optimality_detection(self):
        """Test detection of optimality."""
        standard_a = np.array([[2, 1, 1, 0], [1, -1, 0, 1]])
        standard_b = np.array([6, 2])
        standard_c = np.array([-4, 3, 0, 0])
        optimization_problem = LinearOptimization(
            objective=standard_c,
            constraint_matrix=standard_a,
            right_hand_side=standard_b,
        )
        initial_basis = [2, 3]  # Example basis
        algo = SimplexAlgorithm(optimization_problem, initial_basis)
        for _ in algo:
            pass
        self.assertEqual(algo.stopping_cause, CauseInterruptionIterations.OPTIMAL)

    def test_unbounded_detection(self):
        """Test detection of optimality."""
        standard_a = np.array([[1, 0, -1, 0], [0, 1, 0, 1]])
        standard_b = np.array([6, 2])
        standard_c = np.array([-1, 0, 0, 0])
        optimization_problem = LinearOptimization(
            objective=standard_c,
            constraint_matrix=standard_a,
            right_hand_side=standard_b,
        )
        initial_basis = [0, 1]  # Example basis
        algo = SimplexAlgorithm(optimization_problem, initial_basis)
        for _ in algo:
            pass
        self.assertEqual(algo.stopping_cause, CauseInterruptionIterations.UNBOUNDED)


if __name__ == '__main__':
    unittest.main()
