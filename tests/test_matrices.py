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
    print_string_matrix,
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


class TestPrintStringMatrix(unittest.TestCase):
    def test_basic_functionality(self):
        """Test the basic functionality with rows and headers."""
        matrix = [['Alice', '30', 'Engineer'], ['Bob', '25', 'Designer']]
        headers = ['Name', 'Age', 'Occupation']
        expected_output = (
            'Name  | Age | Occupation\n'
            '------------------------\n'
            'Alice | 30  | Engineer  \n'
            'Bob   | 25  | Designer  '
        )
        self.assertEqual(print_string_matrix(matrix, headers), expected_output)

    def test_with_no_headers(self):
        """Test functionality with rows but no headers."""
        matrix = [['Alice', '30', 'Engineer'], ['Bob', '25', 'Designer']]
        expected_output = 'Alice | 30 | Engineer\nBob   | 25 | Designer'
        self.assertEqual(print_string_matrix(matrix), expected_output)

    def test_empty_matrix(self):
        """Test with an empty matrix."""
        self.assertEqual(print_string_matrix([]), '')

    def test_headers_length_mismatch(self):
        """Test error raised when headers length doesn't match the number of columns."""
        matrix = [['Alice', '30', 'Engineer']]
        headers = ['Name', 'Age']
        with self.assertRaises(AssertionError):
            print_string_matrix(matrix, headers)

    def test_varying_string_lengths(self):
        """Test with varying lengths of strings within the matrix."""
        matrix = [
            ['Alice', '3000', 'Software Engineer'],
            ['Bob', '25', 'UI/UX Designer'],
        ]
        headers = ['Name', 'Age', 'Occupation']
        expected_output = (
            'Name  | Age  | Occupation       \n'
            '--------------------------------\n'
            'Alice | 3000 | Software Engineer\n'
            'Bob   | 25   | UI/UX Designer   '
        )
        self.assertEqual(print_string_matrix(matrix, headers), expected_output)


if __name__ == '__main__':
    unittest.main()
