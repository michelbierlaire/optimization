"""File test_matrices.py

:author: Michel Bierlaire
:date: Mon Mar 18 17:32:24 2024

Tests for the polyhedron class
"""

import unittest

import numpy as np

from biogeme_optimization.teaching.matrices import (
    find_opposite_columns,
    find_columns_multiple_identity,
)


class TestFindOppositeColumns(unittest.TestCase):

    def test_no_opposite_columns(self):
        matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.assertEqual(find_opposite_columns(matrix), [])

    def test_single_pair_of_opposite_columns(self):
        matrix = np.array([[1, -1], [2, -2], [3, -3]])
        self.assertEqual(find_opposite_columns(matrix), [(0, 1)])

    def test_multiple_pairs_of_opposite_columns(self):
        matrix = np.array([[1, -1, -2, 2], [-3, 3, 4, -4]])
        self.assertEqual(
            sorted(find_opposite_columns(matrix)), sorted([(0, 1), (2, 3)])
        )

    def test_empty_matrix(self):
        matrix = np.array([[]])
        self.assertEqual(find_opposite_columns(matrix), [])

    def test_matrix_with_one_column(self):
        matrix = np.array([[1], [-1], [1]])
        self.assertEqual(find_opposite_columns(matrix), [])

    def test_matrix_with_repeated_columns(self):
        matrix = np.array([[1, 1, -1], [2, 2, -2], [3, 3, -3]])
        self.assertEqual(find_opposite_columns(matrix), [(0, 2), (1, 2)])

    def test_matrix_with_floats(self):
        matrix = np.array([[1.0, -1.0], [2.2, -2.2], [3.333, -3.333]])
        self.assertEqual(find_opposite_columns(matrix), [(0, 1)])


class TestFindSpecialColumns(unittest.TestCase):

    def test_empty_matrix(self):
        matrix = np.array([[]])
        result = find_columns_multiple_identity(matrix)
        self.assertEqual(result, [])

    def test_no_special_columns(self):
        matrix = np.array([[1, 2], [3, 4]])
        result = find_columns_multiple_identity(matrix)
        self.assertEqual(result, [])

    def test_all_special_columns(self):
        matrix = np.array([[0, 5, 0], [0, 0, 6], [7, 0, 0]])
        result = sorted(find_columns_multiple_identity(matrix))
        expected = sorted([(0, 2), (1, 0), (2, 1)])
        self.assertEqual(result, expected)

    def test_some_special_columns(self):
        matrix = np.array([[0, 4, 0], [5, 0, 5], [0, 0, 6]])
        result = sorted(find_columns_multiple_identity(matrix))
        expected = sorted([(0, 1), (1, 0)])
        self.assertEqual(result, expected)

    def test_single_column_matrix(self):
        matrix = np.array([[0], [0], [5]])
        result = find_columns_multiple_identity(matrix)
        self.assertEqual(result, [(0, 2)])

    def test_single_row_matrix(self):
        matrix = np.array([[0, 2, 0]])
        result = find_columns_multiple_identity(matrix)
        self.assertEqual(result, [(1, 0)])

    def test_matrix_with_zeros(self):
        matrix = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        result = find_columns_multiple_identity(matrix)
        self.assertEqual(result, [])

    def test_matrix_with_multiple_non_zero(self):
        matrix = np.array([[1, 7, 0], [0, 2, 3], [3, 0, 4]])
        result = find_columns_multiple_identity(matrix)
        self.assertEqual(result, [])
