"""File test_constraints.py

:author: Michel Bierlaire
:date: Mon Mar 25 19:36:51 2024

Tests for the Constraint class
"""

import unittest

from biogeme_optimization.teaching.linear_constraints import (
    Widths,
    ConstraintNamer,
    SignConstraint,
    Term,
    Constraint,
    Variable,
    SignVariable,
    PlusMinusVariables,
)
import numpy as np


class TestWidths(unittest.TestCase):
    def test_total_width_calculation(self):
        widths = Widths(coefficient=5, variable=10)
        expected_total = (
            2 + 5 + 1 + 10
        )  # sign (2) + coefficient (5) + space (1) + variable (10)
        self.assertEqual(widths.total, expected_total)


class TestConstraintNamer(unittest.TestCase):
    def test_slack_names(self):
        namer = ConstraintNamer("constraint1")
        self.assertEqual(namer.slacked_constraint_name, "constraint1_SLACKED")
        self.assertEqual(namer.slack_variable_name, "s_constraint1")

    def test_base_name(self):
        self.assertEqual(
            ConstraintNamer.get_base_name("constraint1_SLACKED"), "constraint1"
        )
        self.assertEqual(ConstraintNamer.get_base_name("s_constraint1"), "constraint1")
        self.assertEqual(ConstraintNamer.get_base_name("other"), "other")


class TestTerm(unittest.TestCase):
    def setUp(self):
        self.variable = Variable(name='x', sign=SignVariable.UNSIGNED)

    def test_term_string_representation(self):
        term = Term(coefficient=1, variable=self.variable)
        self.assertEqual(str(term), "+ x")

        term = Term(
            coefficient=-1,
            variable=self.variable,
            column_width=Widths(coefficient=3, variable=1),
        )
        self.assertIn("- x", str(term))

        term = Term(coefficient=0, variable=self.variable)
        self.assertEqual(str(term), "")

    def test_format_with_width(self):
        term = Term(
            coefficient=2.5,
            variable=self.variable,
            column_width=Widths(coefficient=4, variable=5),
        )
        self.assertIn("2.5", str(term))  # Test formatting with specified width


class TestConstraint(unittest.TestCase):
    def setUp(self):
        self.variable_x = Variable(name='x', sign=SignVariable.UNSIGNED)
        self.variable_y = Variable(name='y', sign=SignVariable.NON_NEGATIVE)
        self.term_x = Term(coefficient=2, variable=self.variable_x)
        self.term_y = Term(coefficient=-3, variable=self.variable_y)

    def test_constraint_creation_and_simplification(self):
        constraint = Constraint(
            name="test",
            left_hand_side=[self.term_x, self.term_y],
            sign=SignConstraint.EQUAL,
            right_hand_side=5,
        )
        simplified_constraint = constraint.simplify()
        self.assertIn("test", str(simplified_constraint))
        self.assertIn("==", str(simplified_constraint))
        self.assertIn("5", str(simplified_constraint))

    def test_add_and_remove_slack_variable(self):
        constraint = Constraint(
            name="test",
            left_hand_side=[self.term_x],
            sign=SignConstraint.LESSER_OR_EQUAL,
            right_hand_side=5,
        )
        slack_added_constraint = constraint.add_slack_variable()
        self.assertIn("s_test", str(slack_added_constraint))

        slack_removed_constraint = slack_added_constraint.remove_slack_variable()
        self.assertNotIn("s_test", str(slack_removed_constraint))

    def test_is_feasible(self):
        constraint = Constraint(
            name="test",
            left_hand_side=[self.term_x, self.term_y],
            sign=SignConstraint.EQUAL,
            right_hand_side=5,
        )
        self.assertTrue(
            constraint.is_feasible(
                vector_x=np.array([2.5, 0]),
                indices={self.variable_x: 0, self.variable_y: 1},
            )
        )
        self.assertFalse(
            constraint.is_feasible(
                vector_x=np.array([2.5, 1]),
                indices={self.variable_x: 0, self.variable_y: 1},
            )
        )


if __name__ == '__main__':
    unittest.main()
