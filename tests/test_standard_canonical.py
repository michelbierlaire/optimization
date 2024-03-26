"""Tests for the standard and canonical forms of the constraints representation

Michel Bierlaire
Tue Mar 26 18:58:08 2024

"""

import unittest
import numpy as np
from biogeme_optimization.teaching.linear_constraints import (
    AllConstraints,
    Constraint,
    Term,
    Variable,
    StandardCanonicalForms,
    SignVariable,
    SignConstraint,
)


class TestStandardCanonicalForms(unittest.TestCase):

    def setUp(self):
        # Setting up example constraints and variables for the tests
        self.x0 = Variable(name='x0', sign=SignVariable.NON_NEGATIVE)
        self.x1 = Variable(name='x1', sign=SignVariable.NON_POSITIVE)
        self.x2 = Variable(name='x2', sign=SignVariable.UNSIGNED)

        self.c0 = Constraint(
            name='c0',
            left_hand_side=[Term(1.0, self.x0), Term(-1, self.x1)],
            sign=SignConstraint.LESSER_OR_EQUAL,
            right_hand_side=1,
        )
        self.c1 = Constraint(
            name='c1',
            left_hand_side=[Term(-1.0, self.x0), Term(1, self.x2)],
            sign=SignConstraint.GREATER_OR_EQUAL,
            right_hand_side=2,
        )
        self.c2 = Constraint(
            name='c2',
            left_hand_side=[Term(-1.0, self.x0), Term(-2.0, self.x1), Term(1, self.x2)],
            sign=SignConstraint.EQUAL,
            right_hand_side=2,
        )
        all_constraints = AllConstraints([self.c0, self.c1, self.c2])
        self.forms = StandardCanonicalForms(all_constraints)

    def test_canonical_to_standard(self):
        # Test transforming a canonical vector to standard form
        canonical_x = np.array([1, 2, 3])
        expected_standard_x = np.array([1.0, 2.0, 3.0, 0.0, 2.0, 0.0])
        standard_x = self.forms.from_canonical_to_standard(canonical_x)
        np.testing.assert_array_almost_equal(standard_x, expected_standard_x)

    def test_standard_to_canonical(self):
        # Test transforming a standard vector to canonical form
        standard_x = np.array([1.0, 2.0, 3.0, 0.0, 2.0, 0.0])
        expected_canonical_x = np.array([1, 2, 3])  # Adjust based on expected behavior
        canonical_x = self.forms.from_standard_to_canonical(standard_x)
        np.testing.assert_array_almost_equal(canonical_x, expected_canonical_x)

    def test_dimension_mismatch_error(self):
        # Test for dimension mismatch error
        wrong_dimension_x = np.array([1, 2])
        with self.assertRaises(AssertionError):
            self.forms.from_canonical_to_standard(wrong_dimension_x)
        with self.assertRaises(AssertionError):
            self.forms.from_standard_to_canonical(wrong_dimension_x)

    def test_nan_entries_error(self):
        # Test for NaN entries error after conversion
        correct_dimension_x = np.array([1, np.nan, 3])
        with self.assertRaises(AssertionError):
            standard_x = self.forms.from_canonical_to_standard(correct_dimension_x)
            assert not np.any(np.isnan(standard_x)), 'The vector contains NaN entries.'
        with self.assertRaises(AssertionError):
            canonical_x = self.forms.from_standard_to_canonical(correct_dimension_x)
            assert not np.any(np.isnan(canonical_x)), 'The vector contains NaN entries.'


if __name__ == '__main__':
    unittest.main()
