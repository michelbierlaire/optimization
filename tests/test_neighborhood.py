"""File test_neighborhood.py

:author: Michel Bierlaire
:date: Wed Jul  5 15:58:49 2023

Tests for the neighborhood module
"""
import unittest
import os
import tempfile
import numpy as np
from unittest.mock import patch
from biogeme_optimization.exceptions import (
    OptimizationError,
)  # Import the required exception class
from biogeme_optimization.neighborhood import OperatorsManagement


class TestOperatorsManagement(unittest.TestCase):
    def setUp(self):
        self.operators = {
            'operator1': lambda solution, neighborhood_size: solution
            + neighborhood_size,
            'operator2': lambda solution, neighborhood_size: solution
            - neighborhood_size,
        }
        self.op_management = OperatorsManagement(self.operators)

    def test_increase_score_last_operator(self):
        self.op_management.last_operator_name = 'operator1'
        self.op_management.increase_score_last_operator()
        self.assertEqual(self.op_management.scores['operator1'], 1)

    def test_increase_score_last_operator_unknown_operator(self):
        self.op_management.last_operator_name = 'unknown_operator'
        with self.assertRaises(OptimizationError):
            self.op_management.increase_score_last_operator()

    def test_decrease_score_last_operator(self):
        self.op_management.last_operator_name = 'operator1'
        self.op_management.scores['operator1'] = 2
        self.op_management.decrease_score_last_operator()
        self.assertEqual(self.op_management.scores['operator1'], 1)

    def test_decrease_score_last_operator_minimum_score(self):
        self.op_management.last_operator_name = 'operator1'
        self.op_management.scores['operator1'] = 0
        self.op_management.decrease_score_last_operator()
        self.assertEqual(self.op_management.scores['operator1'], -1)

    def test_decrease_score_last_operator_unknown_operator(self):
        self.op_management.last_operator_name = 'unknown_operator'
        with self.assertRaises(OptimizationError):
            self.op_management.decrease_score_last_operator()

    def test_select_operator(self):
        # Mocking np.random.choice to always return 'operator1'
        chosen = ('operator1',)
        with patch('numpy.random.choice', return_value=chosen):
            operator = self.op_management.select_operator()
            self.assertEqual(operator, self.operators['operator1'])

    def test_select_operator_with_unavailable_operators(self):
        self.op_management.available['operator2'] = False
        # Mocking np.random.choice to always return 'operator1'
        chosen = ('operator1',)
        with patch('numpy.random.choice', return_value=chosen):
            operator = self.op_management.select_operator()
            self.assertEqual(operator, self.operators['operator1'])

    def test_select_operator_with_min_probability(self):
        # Mocking np.random.choice to always return 'operator1'
        chosen = ('operator1',)
        with patch('numpy.random.choice', return_value=chosen):
            self.op_management.min_probability = 0.5
            operator = self.op_management.select_operator()
            self.assertEqual(operator, self.operators['operator1'])

    def test_select_operator_with_large_min_probability(self):
        with self.assertRaises(OptimizationError):
            self.op_management.min_probability = 0.7
            self.op_management.select_operator()

    def test_probability(self):
        self.op_management.available = [True, True, True]
        prob = np.array([0.0, 0.1, 0.45, 0.45])
        processed_prob = OperatorsManagement.enforce_minimum_probability(
            prob=prob, min_probability=0.2
        )
        expected_output = np.array([0, 0.2, 0.4, 0.4])
        np.testing.assert_array_equal(processed_prob, expected_output)

        prob = np.array([0.0, 0.5, 0.5])
        with self.assertRaises(OptimizationError):
            processed_prob = OperatorsManagement.enforce_minimum_probability(
                prob=prob, min_probability=0.6
            )

        prob = np.array([0.0, 0.5, 0.7])
        with self.assertRaises(OptimizationError):
            processed_prob = OperatorsManagement.enforce_minimum_probability(
                prob=prob, min_probability=0.6
            )

        prob = np.array([0.0, -0.5, 0.7])
        with self.assertRaises(OptimizationError):
            processed_prob = OperatorsManagement.enforce_minimum_probability(
                prob=prob, min_probability=0.6
            )

        prob = np.array([0.0, 0.1, 0.1, 0.8])
        with self.assertRaises(OptimizationError):
            processed_prob = OperatorsManagement.enforce_minimum_probability(
                prob=prob, min_probability=0.6
            )


if __name__ == '__main__':
    unittest.main()
