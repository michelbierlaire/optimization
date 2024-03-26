"""File test_variables.py

:author: Michel Bierlaire
:date: Mon Mar 25 17:55:46 2024

Tests for the variable classes
"""

import unittest
from biogeme_optimization.teaching.linear_constraints import (
    SignVariable,
    Variable,
    PlusMinusVariables,
    sort_variables,
)


class TestVariable(unittest.TestCase):

    def test_variable_creation(self):
        var = Variable(name='x', sign=SignVariable.UNSIGNED)
        self.assertEqual(var.name, 'x')
        self.assertEqual(var.sign, SignVariable.UNSIGNED)
        self.assertFalse(var.slack)

    def test_variable_string_representation(self):
        var = Variable(name='x', sign=SignVariable.UNSIGNED)
        self.assertEqual(str(var), 'x')

    def test_variable_comparisons(self):
        var1 = Variable(name='x', sign=SignVariable.UNSIGNED)
        var2 = Variable(name='y', sign=SignVariable.NON_NEGATIVE)
        self.assertTrue(var1 < var2)
        self.assertFalse(var1 == var2)


class TestPlusMinusVariables(unittest.TestCase):

    def setUp(self):
        self.existing = Variable(name='z', sign=SignVariable.UNSIGNED)
        self.plus = Variable(name='x', sign=SignVariable.NON_NEGATIVE)
        self.minus = Variable(name='y', sign=SignVariable.NON_NEGATIVE)

    def test_plus_minus_variables_creation(self):
        pm_vars = PlusMinusVariables(
            existing=self.existing, plus=self.plus, minus=self.minus
        )
        self.assertEqual(pm_vars.existing, self.existing)
        self.assertEqual(pm_vars.plus, self.plus)
        self.assertEqual(pm_vars.minus, self.minus)

    def test_plus_minus_variables_wrong_existing_sign(self):
        wrong_existing = Variable(name='w', sign=SignVariable.NON_POSITIVE)
        with self.assertRaises(ValueError):
            PlusMinusVariables(
                existing=wrong_existing, plus=self.plus, minus=self.minus
            )

    def test_plus_minus_variables_wrong_plus_sign(self):
        wrong_plus = Variable(name='x', sign=SignVariable.NON_POSITIVE)
        with self.assertRaises(ValueError):
            PlusMinusVariables(
                existing=self.existing, plus=wrong_plus, minus=self.minus
            )

    def test_plus_minus_variables_wrong_minus_sign(self):
        wrong_minus = Variable(name='y', sign=SignVariable.UNSIGNED)
        with self.assertRaises(ValueError):
            PlusMinusVariables(
                existing=self.existing, plus=self.plus, minus=wrong_minus
            )


class TestSortVariables(unittest.TestCase):

    def setUp(self):
        self.vars = [
            Variable(name='c', sign=SignVariable.UNSIGNED, slack=True),
            Variable(name='a', sign=SignVariable.NON_NEGATIVE),
            Variable(name='b', sign=SignVariable.UNSIGNED),
        ]

    def test_sort_variables(self):
        sorted_vars = sort_variables(self.vars)
        self.assertEqual([var.name for var in sorted_vars], ['a', 'b', 'c'])

    def test_sort_variables_with_all_slacks(self):
        slack_vars = [
            Variable(name='c', sign=SignVariable.UNSIGNED, slack=True),
            Variable(name='a', sign=SignVariable.UNSIGNED, slack=True),
            Variable(name='b', sign=SignVariable.UNSIGNED, slack=True),
        ]
        sorted_vars = sort_variables(slack_vars)
        self.assertEqual([var.name for var in sorted_vars], ['a', 'b', 'c'])


if __name__ == '__main__':
    unittest.main()
