"""File test_vns.py

:author: Michel Bierlaire
:date: Fri Jul  7 15:30:29 2023

Tests for the vns module
"""
import unittest
import os
from collections import defaultdict
import tempfile
from unittest.mock import MagicMock, patch
from random import choice
from knapsack import Knapsack, Sack
from biogeme_optimization.vns import ParetoClass, vns
from biogeme_optimization.pareto import SetElement


class ParetoClassTests(unittest.TestCase):
    def setUp(self):
        self.max_neighborhood = 5
        self.pareto_file = tempfile.mktemp()
        self.pareto_class = ParetoClass(self.max_neighborhood, self.pareto_file)
        self.element1 = SetElement(1, [1, 2])
        self.element2 = SetElement(2, [2, 1])

    def tearDown(self):
        if os.path.exists(self.pareto_file):
            os.remove(self.pareto_file)

    def test_init(self):
        self.assertEqual(self.pareto_class.max_neighborhood, self.max_neighborhood)
        self.assertEqual(self.pareto_class.filename, self.pareto_file)
        self.assertEqual(self.pareto_class.neighborhood_size, defaultdict(int))

    def test_change_neighborhood(self):
        self.pareto_class.change_neighborhood(self.element1)
        self.assertEqual(self.pareto_class.neighborhood_size[self.element1], 1)
        self.pareto_class.change_neighborhood(self.element1)
        self.assertEqual(self.pareto_class.neighborhood_size[self.element1], 2)

    def test_reset_neighborhood(self):
        self.pareto_class.change_neighborhood(self.element1)
        self.pareto_class.reset_neighborhood(self.element1)
        self.assertEqual(self.pareto_class.neighborhood_size[self.element1], 1)

    def test_add_element_added(self):
        self.pareto_class.add(self.element1)
        self.assertEqual(self.pareto_class.neighborhood_size[self.element1], 1)

    def test_add_element_not_added(self):
        self.pareto_class.add(self.element1)
        self.pareto_class.add(self.element1)
        self.assertEqual(self.pareto_class.neighborhood_size[self.element1], 2)

    def test_select(self):
        candidate = (self.element1, 3)
        self.pareto_class.neighborhood_size = {self.element1: 1, self.element2: 3}
        with patch("random.choice", MagicMock(return_value=candidate)):
            selected_element, neighborhood_size = self.pareto_class.select()
            self.assertEqual(selected_element, self.element1)
            self.assertEqual(neighborhood_size, 3)

    def test_select_no_candidates(self):
        self.pareto_class.neighborhood_size = {self.element1: 5, self.element2: 5}
        with patch("random.choice", MagicMock(side_effect=IndexError)):
            selected_element, neighborhood_size = self.pareto_class.select()
            self.assertIsNone(selected_element)
            self.assertIsNone(neighborhood_size)


class VNSTests(unittest.TestCase):
    def setUp(self):
        self.UTILITY = [80, 31, 48, 17, 27, 84, 34, 39, 46, 58, 23, 67]
        self.WEIGHT = [84, 27, 47, 22, 21, 96, 42, 46, 54, 53, 32, 78]
        self.CAPACITY = 300
        self.FILE_NAME = 'knapsack.pareto'
        Sack.utility_data = self.UTILITY
        Sack.weight_data = self.WEIGHT
        self.the_pareto = ParetoClass(max_neighborhood=5, pareto_file=self.FILE_NAME)
        self.the_knapsack = Knapsack(self.UTILITY, self.WEIGHT, self.CAPACITY)
        self.empty_sack = Sack.empty_sack(size=len(self.UTILITY))

    def test_vns(self):
        the_pareto = vns(
            problem=self.the_knapsack,
            first_solutions=[self.empty_sack.get_element()],
            pareto=self.the_pareto,
            number_of_neighbors=5,
        )


if __name__ == "__main__":
    unittest.main()
