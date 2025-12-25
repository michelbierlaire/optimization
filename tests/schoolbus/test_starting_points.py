import random
import unittest
from datetime import datetime

from biogeme_optimization.school_bus import (
    Bus,
    Destination,
    Origin,
    SchoolBusProblem,
    Solution,
)
from biogeme_optimization.school_bus.starting_points import single_bus, taxi_solution


class TestSingleBusOperator(unittest.TestCase):

    def setUp(self):

        self.problem = SchoolBusProblem(
            schools={
                Destination(name="C", arrival_time=datetime(2025, 6, 4, 8, 0)),
                Destination(name="D", arrival_time=datetime(2025, 6, 4, 8, 0)),
                Destination(name="F", arrival_time=datetime(2025, 6, 4, 8, 0)),
            },
            origins={
                Origin("A"),
                Origin("B"),
                Origin("E"),
            },
            travel_times_in_minutes={
                ("A", "C"): 10.0,
                ("A", "D"): 12.0,
                ("A", "F"): 20.0,
                ("B", "C"): 9.0,
                ("B", "D"): 8.0,
                ("B", "F"): 18.0,
                ("E", "C"): 25.0,
                ("E", "D"): 22.0,
                ("E", "F"): 7.0,
                ("A", "B"): 5.0,
                ("B", "A"): 5.0,
                ("C", "D"): 6.0,
                ("D", "C"): 6.0,
                ("C", "F"): 10.0,
                ("F", "C"): 10.0,
            },
            od_table={("A", "C"): 2, ("B", "D"): 2, ("E", "F"): 1},
            buses={Bus("bus1", capacity=50), Bus("bus2", capacity=50)},
        )

    def test_single_bus_assignment(self):
        the_solution = single_bus(the_problem=self.problem)

        # One bus only should be assigned
        assigned_bus = next(iter(the_solution.bus_to_tours.assignment))
        self.assertIn(assigned_bus, {bus.name for bus in self.problem.buses})

        # All tours should be assigned to that single bus
        assigned_tours = the_solution.bus_to_tours.assignment[assigned_bus]
        self.assertSetEqual(
            set(assigned_tours), {tour.the_id for tour in the_solution.set_of_tours}
        )


class TestTaxiSolution(unittest.TestCase):
    def setUp(self):
        self.test_instance = SchoolBusProblem(
            schools={
                Destination('Central_school', datetime(2025, 6, 9, 8, 30)),
                Destination('Northeast_school', datetime(2025, 6, 9, 8, 15)),
                Destination('Southwest_school', datetime(2025, 6, 9, 8, 15)),
                Destination('Southeast_school', datetime(2025, 6, 9, 8, 20)),
            },
            origins={Origin('Village')},
            travel_times_in_minutes={
                ('Village', 'Central_school'): 20,
                ('Village', 'Northeast_school'): 10,
                ('Village', 'Southwest_school'): 5,
                ('Village', 'Southeast_school'): 25,
                ('Central_school', 'Northeast_school'): 10,
                ('Central_school', 'Southwest_school'): 5,
                ('Central_school', 'Southeast_school'): 5,
                ('Northeast_school', 'Southwest_school'): 15,
                ('Northeast_school', 'Southeast_school'): 20,
                ('Southwest_school', 'Southeast_school'): 3,
            },
            od_table={
                ('Village', 'Central_school'): 50,
                ('Village', 'Northeast_school'): 10,
                ('Village', 'Southwest_school'): 12,
                ('Village', 'Southeast_school'): 13,
                ('Central_school', 'Northeast_school'): 6,
                ('Central_school', 'Southwest_school'): 5,
                ('Central_school', 'Southeast_school'): 4,
            },
            buses={Bus('Bus 1', capacity=50), Bus('Bus 2', capacity=50)},
        )
        random.seed(42)  # For deterministic test

    def test_taxi_solution_structure(self):
        solution = taxi_solution(self.test_instance)

        # Check solution is of correct type
        self.assertIsInstance(solution, Solution)

        # Check group IDs
        expected_ids = {
            f'{origin}_to_{dest}' for (origin, dest) in self.test_instance.od_table
        }
        actual_group_ids = {g.the_id for g in solution.set_of_groups}
        actual_tour_ids = {t.the_id for t in solution.set_of_tours}

        # Groups and tours should match OD table
        self.assertEqual(expected_ids, actual_group_ids)
        self.assertEqual(expected_ids, actual_tour_ids)

        # Check group assignment correctness
        self.assertEqual(set(solution.group_to_tour.assignment.keys()), expected_ids)
        self.assertEqual(set(solution.group_to_tour.assignment.values()), expected_ids)

        # Check all buses are included
        assigned_bus_names = set(solution.bus_to_tours.assignment.keys())
        problem_bus_names = {b.name for b in self.test_instance.buses}
        self.assertEqual(assigned_bus_names, problem_bus_names)

        # All tours must be assigned exactly once
        assigned_tours = [
            tid
            for tour_list in solution.bus_to_tours.assignment.values()
            for tid in tour_list
        ]
        self.assertCountEqual(
            assigned_tours, list(expected_ids)
        )  # Same elements, same multiplicity
        self.assertEqual(len(set(assigned_tours)), len(expected_ids))  # Uniqueness

        # Every tour is assigned to exactly one bus
        self.assertEqual(
            sum(len(lst) for lst in solution.bus_to_tours.assignment.values()),
            len(expected_ids),
        )


if __name__ == "__main__":
    unittest.main()
