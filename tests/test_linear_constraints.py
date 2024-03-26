"""File test_linear_constraints.py

:author: Michel Bierlaire
:date: Tue Mar 19 08:06:56 2024

Tests for the linear constraints representation
"""

import unittest

import numpy as np

from biogeme_optimization.teaching.linear_constraints import (
    AllConstraints,
    Constraint,
    Variable,
    Term,
    SignConstraint,
    SignVariable,
    Widths,
)


class TestAllConstraints(unittest.TestCase):

    def setUp(self):
        # Common setup for the tests
        self.var1 = Variable(name="x1", sign=SignVariable.NON_NEGATIVE)
        self.var2 = Variable(name="x2", sign=SignVariable.UNSIGNED)
        self.const1 = Constraint(
            name="c1",
            left_hand_side=[Term(1, self.var1)],
            sign=SignConstraint.LESSER_OR_EQUAL,
            right_hand_side=10,
        )
        self.const2 = Constraint(
            name="c2",
            left_hand_side=[Term(1, self.var2)],
            sign=SignConstraint.EQUAL,
            right_hand_side=5,
        )

    def test_unique_constraint_names(self):
        # Verify that duplicate constraint names raise a ValueError
        with self.assertRaises(ValueError):
            AllConstraints([self.const1, self.const1])

    def test_from_canonical_form_structure(self):
        # Test structure and values after conversion from canonical form
        matrix = np.array([[1, 0], [0, 1]])
        vector = np.array([3, 5])
        ac = AllConstraints.from_canonical_form(matrix, vector)
        self.assertEqual(
            len(ac.constraints), 2
        )  # Assuming no sign constraints detected
        self.assertEqual(ac.n_variables, 2)
        # Additional checks for variables and constraints in ac
        for constraint in ac:
            self.assertEqual(constraint.sign, SignConstraint.LESSER_OR_EQUAL)

    def test_from_standard_form_structure(self):
        # Test structure and values after conversion from standard form
        matrix = np.array([[1, -1], [-1, 1]])
        vector = np.array([4, 6])
        ac = AllConstraints.from_standard_form(matrix, vector)
        self.assertEqual(len(ac.constraints), 2)
        self.assertEqual(ac.n_variables, 2)
        # Additional checks for variables and constraints in ac
        for variable in ac.unsorted_set_of_variables:
            self.assertEqual(variable.sign, SignVariable.NON_NEGATIVE)

    def test_remove_unsigned_variables(self):
        ac = AllConstraints([self.const2])
        new_ac = ac.remove_unsigned_variables()
        self.assertNotEqual(ac, new_ac)
        # Check that unsigned variables have been replaced correctly
        unsigned_ac = [
            variable
            for variable in ac.unsorted_set_of_variables
            if variable.sign == SignVariable.UNSIGNED
        ]
        self.assertEqual(len(unsigned_ac), 1)
        unsigned_new_ac = [
            variable
            for variable in new_ac.unsorted_set_of_variables
            if variable.sign == SignVariable.UNSIGNED
        ]
        self.assertEqual(len(unsigned_new_ac), 0)

    def test_make_variables_non_negative(self):
        ac = AllConstraints([self.const2])
        non_negative_ac = [
            variable
            for variable in ac.unsorted_set_of_variables
            if variable.sign == SignVariable.NON_NEGATIVE
        ]
        new_ac = ac.make_variables_non_negative()
        non_negative_new_ac = [
            variable
            for variable in new_ac.unsorted_set_of_variables
            if variable.sign == SignVariable.NON_NEGATIVE
        ]
        self.assertNotEqual(ac, new_ac)
        # Check that variables have been made non-negative correctly
        self.assertEqual(len(non_negative_ac), 0)
        self.assertEqual(
            len(non_negative_new_ac), len(new_ac.unsorted_set_of_variables)
        )

    def test_variable_value_errors(self):
        ac = AllConstraints([self.const1])
        wrong_vector_length = np.array([1, 1, 1, 1, 1, 1])
        with self.assertRaises(AssertionError):
            ac.variable_value(self.var1, wrong_vector_length)
        # Additional error checks for incompatible vectors

    def test_get_constraint_by_name(self):
        ac = AllConstraints([self.const1, self.const2])
        self.assertEqual(ac.get_constraint_by_name("c1"), self.const1)
        self.assertIsNone(ac.get_constraint_by_name("nonexistent"))

    def test_calculate_column_width(self):
        # Test the column width calculation logic
        ac = AllConstraints([self.const1, self.const2])
        widths = ac.calculate_column_width()
        expected_widths = Widths(coefficient=2, variable=2)
        self.assertEqual(widths, expected_widths)

    def test_is_feasible(self):
        ac = AllConstraints([self.const1, self.const2])
        feasible, details = ac.is_feasible(vector_x=np.array([3, 5]))
        self.assertTrue(feasible)
        feasible, details = ac.is_feasible(vector_x=np.array([3, 3]))
        self.assertFalse(feasible)
        self.assertTrue(details['c1'])
        self.assertFalse(details['c2'])
        feasible, details = ac.is_feasible(vector_x=np.array([15, 5]))
        self.assertFalse(feasible)
        self.assertTrue(details['c2'])
        self.assertFalse(details['c1'])


if __name__ == '__main__':
    unittest.main()
