"""File test_pareto.py

:author: Michel Bierlaire
:date: Tue Jul  4 17:25:58 2023

Tests for the pareto module
"""
import unittest
import os
import tempfile
import numpy as np
from biogeme_optimization.exceptions import (
    OptimizationError,
)  # Import the required exception class
from biogeme_optimization.pareto import SetElement, Pareto

import unittest


class TestSetElement(unittest.TestCase):
    def test_initialization(self):
        element_id = "element1"
        objectives = [1.0, 2.0, 3.0]
        set_element = SetElement(element_id, objectives)
        self.assertEqual(set_element.element_id, element_id)
        self.assertEqual(set_element.objectives, objectives)

    def test_initialization_missing_objectives(self):
        element_id = "element2"
        objectives = [1.0, None, 3.0]
        with self.assertRaises(OptimizationError):
            SetElement(element_id, objectives)

    def test_equality(self):
        element_id = "element3"
        objectives = [1.0, 2.0, 3.0]
        set_element1 = SetElement(element_id, objectives)
        set_element2 = SetElement(element_id, objectives)
        self.assertEqual(set_element1, set_element2)

    def test_equality_different_objectives(self):
        element_id = "element4"
        objectives1 = [1.0, 2.0, 3.0]
        objectives2 = [1.0, 3.0, 2.0]
        set_element1 = SetElement(element_id, objectives1)
        set_element2 = SetElement(element_id, objectives2)
        with self.assertRaises(OptimizationError):
            self.assertEqual(set_element1, set_element2)

    def test_hash(self):
        element_id = "element5"
        objectives = [1.0, 2.0, 3.0]
        set_element = SetElement(element_id, objectives)
        self.assertEqual(hash(set_element), hash((element_id, tuple(objectives))))

    def test_dominance(self):
        element_id1 = "element6"
        objectives1 = [1.0, 2.0, 3.0]
        element_id2 = "element7"
        objectives2 = [2.0, 3.0, 4.0]
        set_element1 = SetElement(element_id1, objectives1)
        set_element2 = SetElement(element_id2, objectives2)
        self.assertTrue(set_element1.dominates(set_element2))
        self.assertFalse(set_element2.dominates(set_element1))
        self.assertFalse(set_element1.dominates(set_element1))
        self.assertFalse(set_element2.dominates(set_element2))

    def test_dominance_incompatible_sizes(self):
        element_id1 = "element8"
        objectives1 = [1.0, 2.0, 3.0]
        element_id2 = "element9"
        objectives2 = [2.0, 3.0]
        set_element1 = SetElement(element_id1, objectives1)
        set_element2 = SetElement(element_id2, objectives2)
        with self.assertRaises(OptimizationError):
            set_element1.dominates(set_element2)


class TestPareto(unittest.TestCase):
    def setUp(self):
        self.filename = tempfile.mktemp()

    def tearDown(self):
        if os.path.exists(self.filename):
            os.remove(self.filename)

    def test_dump_restore(self):
        pareto = Pareto(self.filename)

        # Add some elements to the Pareto set
        pareto.add_invalid(SetElement('1', [1.0, 2.0]))
        pareto.add(SetElement('3', [5.0, 6.0]))
        pareto.add(SetElement('2', [3.0, 4.0]))

        # Check if the restored Pareto set matches the original one
        self.assertEqual(len(pareto.invalid), 1)
        self.assertEqual(len(pareto.pareto), 1)
        self.assertEqual(len(pareto.considered), 2)
        self.assertEqual(len(pareto.removed), 1)

        # Dump the Pareto set to a file
        pareto.dump()

        # Create a new Pareto object and restore from the file
        restored_pareto = Pareto(self.filename)
        restored_pareto.restore()

        # Check if the restored Pareto set matches the original one
        self.assertEqual(len(restored_pareto.invalid), 1)
        self.assertEqual(len(restored_pareto.pareto), 1)
        self.assertEqual(len(restored_pareto.considered), 2)
        self.assertEqual(len(restored_pareto.removed), 1)

        # Check the values of the restored elements
        self.assertSetEqual(pareto.pareto, restored_pareto.pareto)
        self.assertSetEqual(pareto.considered, restored_pareto.considered)
        self.assertSetEqual(pareto.removed, restored_pareto.removed)
        self.assertSetEqual(pareto.invalid, restored_pareto.invalid)

    def test_add(self):
        pareto = Pareto()

        # Add elements to the Pareto set
        pareto.add(SetElement('1', [1.0, 3.0]))
        pareto.add(SetElement('2', [2.0, 2.0]))
        pareto.add(SetElement('3', [3.0, 1.0]))

        # Check the size of the Pareto set
        self.assertEqual(pareto.length(), 3)
        self.assertEqual(pareto.length_of_all_sets(), (3, 3, 0, 0))

        # Add an element that is dominated by an existing element
        pareto.add(SetElement('4', [2.0, 4.0]))

        # Check that the dominated element was not added
        self.assertEqual(pareto.length(), 3)
        self.assertEqual(pareto.length_of_all_sets(), (3, 4, 0, 0))

        # Add an element that dominates an existing element
        pareto.add(SetElement('5', [2.5, 0.0]))

        # Check that the dominated element was removed
        self.assertEqual(pareto.length(), 3)
        self.assertEqual(pareto.length_of_all_sets(), (3, 5, 1, 0))

    def test_get_element_from_id(self):
        pareto = Pareto()

        # Add elements to the Pareto set
        pareto.add(SetElement('1', [1.0, 2.0]))
        pareto.add(SetElement('2', [3.0, 4.0]))
        pareto.add(SetElement('3', [5.0, 6.0]))

        # Get an element by ID
        element = pareto.get_element_from_id('2')

        # Check the properties of the element
        self.assertEqual(element.element_id, '2')
        self.assertEqual(element.objectives, [3.0, 4.0])

    def test_add_invalid(self):
        pareto = Pareto()

        # Add an invalid element to the Pareto set
        pareto.add_invalid(SetElement('1', [1.0, 2.0]))

        # Check the size of the Pareto set
        self.assertEqual(pareto.length(), 0)
        self.assertEqual(pareto.length_of_all_sets(), (0, 0, 0, 1))

        # Try adding the same invalid element again
        pareto.add_invalid(SetElement('1', [1.0, 2.0]))

        # Check that the element was not added again
        self.assertEqual(pareto.length(), 0)
        self.assertEqual(pareto.length_of_all_sets(), (0, 0, 0, 1))


if __name__ == '__main__':
    unittest.main()
