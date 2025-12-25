import unittest
from datetime import datetime

from icecream import ic

from biogeme_optimization.pareto import SetElement
from biogeme_optimization.school_bus.data_classes import (
    Bus,
    Destination,
    Origin,
    SchoolBusProblem,
)
from biogeme_optimization.school_bus.decision_variables import (
    BusToToursAssignment,
    ChildrenGroup,
    GroupToTourAssignment,
    Solution,
    Tour,
)
from biogeme_optimization.school_bus.element_pareto import ElementSolution
from biogeme_optimization.school_bus.operators import (
    feasible_tours,
    improve_tour_early_arrival,
    improve_tour_late_arrivals,
    improve_tour_maximum_early_arrivals,
    improve_tour_maximum_late_arrivals,
    improve_tour_maximum_travel_time,
    improve_tour_travel_time,
    merge_groups,
    move_tour_to_another_bus,
    split_tour,
)
from biogeme_optimization.school_bus.sanity_check import SanityCheck


class TestFeasibleTours(unittest.TestCase):

    def test_basic_feasibility(self):
        groups = {
            ChildrenGroup(
                group_id="g1",
                origin_name="A",
                destination_name="B",
                size=3,
            ),
            ChildrenGroup(
                group_id="g2",
                origin_name="C",
                destination_name="D",
                size=2,
            ),
        }
        tours = list(feasible_tours(groups))
        # There should be 6 valid permutations: topological sorts of A→B and C→D with no other constraints
        self.assertEqual(len(tours), 6)
        for tour in tours:
            self.assertLess(tour.index("A"), tour.index("B"))
            self.assertLess(tour.index("C"), tour.index("D"))

    def test_conflicting_constraints(self):
        groups = {
            ChildrenGroup(
                group_id="g1",
                origin_name="A",
                destination_name="B",
                size=1,
            ),
            ChildrenGroup(
                group_id="g2",
                origin_name="B",
                destination_name="A",
                size=1,
            ),
        }
        with self.assertRaises(ValueError):
            list(feasible_tours(groups))

    def test_single_group(self):
        groups = {
            ChildrenGroup(
                group_id="g1",
                origin_name="X",
                destination_name="Y",
                size=1,
            ),
        }
        tours = list(feasible_tours(groups))
        self.assertEqual(tours, [["X", "Y"]])

    def test_empty_group_set(self):
        groups = set()
        tours = list(feasible_tours(groups))
        self.assertEqual(tours, [[]])


class TestImproveTour(unittest.TestCase):

    def setUp(self):
        # Initial data: two tours with feasible improvement possible
        groups = {
            ChildrenGroup("g1", "A", "C", 2),
            ChildrenGroup("g2", "B", "D", 2),
            ChildrenGroup("g3", "E", "F", 1),
        }

        tours = {
            Tour("T1", ["A", "B", "C", "D"]),  # Suboptimal ordering
            Tour("T2", ["E", "F"]),  # Already optimal
        }

        group_tour_assignment = {'g1': 'T1', 'g2': 'T1', 'g3': 'T2'}
        bus_tour_assignment = {'bus1': ['T1'], 'bus2': ['T2']}

        self.solution = Solution(
            set_of_groups=groups,
            set_of_tours=tours,
            group_to_tour=GroupToTourAssignment(assignment=group_tour_assignment),
            bus_to_tours=BusToToursAssignment(assignment=bus_tour_assignment),
        )

        self.the_problem = SchoolBusProblem(
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
            buses={Bus(name="bus1", capacity=10), Bus(name="bus2", capacity=10)},
        )
        ElementSolution.the_problem = self.the_problem
        self.element_solution = ElementSolution(the_solution=self.solution)

        self.element = self.element_solution.get_element()

    def test_improvement_detected(self):
        improved_element, changes = improve_tour_travel_time(self.element)
        self.assertIsInstance(improved_element, SetElement)
        self.assertGreaterEqual(changes, 0)

        improved_solution = ElementSolution.from_element(the_element=improved_element)

        self.assertLessEqual(
            improved_solution.timetable.total_travel_time,
            self.element_solution.timetable.total_travel_time,
        )

    def test_no_improvement_possible(self):
        # Create a solution where both tours are already optimal

        groups = {
            ChildrenGroup("g1", "A", "C", 2),
            ChildrenGroup("g2", "B", "D", 2),
            ChildrenGroup("g3", "E", "F", 1),
        }

        group_tour_assignment = {'g1': 'T1', 'g2': 'T1', 'g3': 'T2'}
        bus_tour_assignment = {'bus1': ['T1'], 'bus2': ['T2']}

        tours = {
            Tour("T1", ["A", "C", "B", "D"]),  # Suboptimal ordering
            Tour("T2", ["E", "F"]),  # Already optimal
        }

        solution = Solution(
            set_of_groups=groups,
            set_of_tours=tours,
            group_to_tour=GroupToTourAssignment(assignment=group_tour_assignment),
            bus_to_tours=BusToToursAssignment(assignment=bus_tour_assignment),
        )
        element_solution = ElementSolution(the_solution=solution)
        element = element_solution.get_element()

        improved_code, changes = improve_tour_travel_time(element)
        self.assertEqual(changes, 0)
        self.assertEqual(improved_code.element_id, element.element_id)

    def test_size_limit(self):
        # Use a size=1 to limit to just one tour being improved
        improved_code, changes = improve_tour_travel_time(self.element, size=1)
        self.assertIn(changes, [0, 1])

    def test_improved_maximum_travel_time(self):
        improved_element, changes = improve_tour_maximum_travel_time(self.element)
        new_solution = ElementSolution.from_code(improved_element.element_id)
        self.assertIsInstance(improved_element, SetElement)
        self.assertGreaterEqual(changes, 0)

        self.assertLessEqual(
            new_solution.timetable.maximum_travel_time,
            self.element_solution.timetable.maximum_travel_time,
        )

    def test_improved_early_s(self):
        improved_element, changes = improve_tour_early_arrival(self.element)
        new_solution = ElementSolution.from_code(improved_element.element_id)
        self.assertIsInstance(improved_element, SetElement)
        self.assertGreaterEqual(changes, 0)

        self.assertLessEqual(
            new_solution.timetable.total_early_arrivals,
            self.element_solution.timetable.total_early_arrivals,
        )

    def test_improved_maximum_early_arrivals(self):
        improved_element, changes = improve_tour_maximum_early_arrivals(self.element)
        new_solution = ElementSolution.from_code(improved_element.element_id)
        self.assertIsInstance(improved_element, SetElement)
        self.assertGreaterEqual(changes, 0)

        self.assertLessEqual(
            new_solution.timetable.maximum_early_arrivals,
            self.element_solution.timetable.maximum_early_arrivals,
        )

    def test_improved_late_arrival(self):
        improved_element, changes = improve_tour_late_arrivals(self.element)
        new_solution = ElementSolution.from_code(improved_element.element_id)
        self.assertIsInstance(improved_element, SetElement)
        self.assertGreaterEqual(changes, 0)

        self.assertLessEqual(
            new_solution.timetable.total_late_arrivals,
            self.element_solution.timetable.total_late_arrivals,
        )

    def test_improved_maximum_late_arrivals(self):
        improved_element, changes = improve_tour_maximum_late_arrivals(self.element)
        new_solution = ElementSolution.from_code(improved_element.element_id)
        self.assertIsInstance(improved_element, SetElement)
        self.assertGreaterEqual(changes, 0)

        self.assertLessEqual(
            new_solution.timetable.maximum_late_arrivals,
            self.element_solution.timetable.maximum_late_arrivals,
        )


class TestSplitTourIntegration(unittest.TestCase):
    def setUp(self):
        origins = {Origin("A"), Origin("X")}
        schools = {
            Destination(name="B", arrival_time=datetime(2025, 6, 9, 8, 0)),
            Destination(name="C", arrival_time=datetime(2025, 6, 9, 8, 0)),
            Destination(name="Y", arrival_time=datetime(2025, 6, 9, 8, 0)),
        }

        # Define travel times
        travel_times = {
            ("A", "B"): 5,
            ("A", "C"): 6,
            ("A", "X"): 10,
            ("A", "Y"): 4,
            ("B", "C"): 6,
            ("B", "X"): 10,
            ("B", "Y"): 4,
            ("C", "X"): 10,
            ("C", "Y"): 4,
            ("X", "Y"): 4,
        }

        od_table = {("A", "C"): 10, ("B", "C"): 5, ("A", "B"): 8, ("X", "Y"): 6}
        # Define buses
        buses = {Bus("Bus1", capacity=50)}

        # Build and register the problem
        problem = SchoolBusProblem(
            schools=schools,
            origins=origins,
            travel_times_in_minutes=travel_times,
            od_table=od_table,  # optional for now
            buses=buses,
        )
        ElementSolution.the_problem = problem
        # Groups to be split
        self.group1 = ChildrenGroup("G1", "A", "C", 10)
        self.group2 = ChildrenGroup("G2", "B", "C", 5)
        self.group3 = ChildrenGroup("G3", "A", "B", 8)

        # Irrelevant group
        self.group4 = ChildrenGroup("G4", "X", "Y", 6)

        # Tours
        self.tour1 = Tour(tour_id="T1", list_of_nodes=["A", "B", "C"])
        self.tour0 = Tour(tour_id="T0", list_of_nodes=["X", "Y"])

        group_assignment = {'G1': 'T1', 'G2': 'T1', 'G3': 'T1', 'G4': 'T0'}
        bus_assignment = {"Bus1": ["T0", "T1"]}

        # Solution
        self.solution = Solution(
            set_of_groups={self.group1, self.group2, self.group3, self.group4},
            set_of_tours={self.tour1, self.tour0},
            group_to_tour=GroupToTourAssignment(assignment=group_assignment),
            bus_to_tours=BusToToursAssignment(assignment=bus_assignment),
        )
        self.element = ElementSolution(the_solution=self.solution).get_element()
        sanity_check = SanityCheck(the_problem=problem, the_solution=self.solution)
        ok, msg = sanity_check.all_checks()
        if not ok:
            raise ValueError(msg)

    def test_split_tour_real_functionality(self):
        new_element, changes = split_tour(self.element, size=1)
        self.assertEqual(changes, 1)

        new_solution = ElementSolution.from_code(new_element.element_id).the_solution

        tour_ids = {t.the_id for t in new_solution.set_of_tours}
        self.assertIn("T1_a", tour_ids)
        self.assertIn("T1_b", tour_ids)
        self.assertNotIn("T1", tour_ids)

        new_group_assignment = new_solution.group_to_tour.assignment
        self.assertIn(new_group_assignment['G1'], {'T1_a', 'T1_b'})
        self.assertIn(new_group_assignment['G2'], {'T1_a', 'T1_b'})
        self.assertIn(new_group_assignment['G3'], {'T1_a', 'T1_b'})
        self.assertEqual(new_group_assignment['G4'], 'T0')
        new_bus_assignment = new_solution.bus_to_tours.assignment
        self.assertIn('T1_a', new_bus_assignment['Bus1'])
        self.assertIn('T1_b', new_bus_assignment['Bus1'])
        self.assertNotIn('T1', new_bus_assignment['Bus1'])


class TestMoveTourToAnotherBus(unittest.TestCase):
    def setUp(self):
        groups = {
            ChildrenGroup("g1", "A", "C", 2),
            ChildrenGroup("g2", "B", "D", 2),
            ChildrenGroup("g3", "E", "F", 1),
        }

        tours = {
            Tour("T1", ["A", "B", "C", "D"]),
            Tour("T2", ["E", "F"]),
        }

        group_tour_assignment = {'g1': 'T1', 'g2': 'T1', 'g3': 'T2'}
        bus_tour_assignment = {'bus1': ['T1'], 'bus2': ['T2']}

        self.solution = Solution(
            set_of_groups=groups,
            set_of_tours=tours,
            group_to_tour=GroupToTourAssignment(assignment=group_tour_assignment),
            bus_to_tours=BusToToursAssignment(assignment=bus_tour_assignment),
        )

        self.the_problem = SchoolBusProblem(
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
            buses={Bus(name="bus1", capacity=10), Bus(name="bus2", capacity=10)},
        )

        ElementSolution.the_problem = self.the_problem
        self.element_solution = ElementSolution(the_solution=self.solution)
        self.element = self.element_solution.get_element()

    def test_move_tour_to_another_bus(self):
        new_element, changes = move_tour_to_another_bus(self.element, size=1)
        new_solution = ElementSolution.from_code(new_element.element_id).the_solution

        # Ensure exactly one tour was moved
        self.assertEqual(changes, 1)

        # Collect new bus assignments
        assignments = new_solution.bus_to_tours.assignment
        assigned_tours = {tour for tours in assignments.values() for tour in tours}
        self.assertEqual(assigned_tours, {"T1", "T2"})  # Still the same tours

        # Ensure one of the tours has switched bus
        old_assignments = self.solution.bus_to_tours.assignment
        ic(assignments)
        ic(old_assignments)
        self.assertNotEqual(assignments, old_assignments)


class TestMergeGroups(unittest.TestCase):
    def setUp(self):
        groups = {
            ChildrenGroup("g1", "A", "C", 2),
            ChildrenGroup("g2", "B", "D", 2),
            ChildrenGroup("g3", "E", "F", 1),
        }

        tours = {
            Tour("T1", ["A", "B", "C", "D"]),
            Tour("T2", ["E", "F"]),
        }

        group_tour_assignment = {'g1': 'T1', 'g2': 'T1', 'g3': 'T2'}
        bus_tour_assignment = {'bus1': ['T1'], 'bus2': ['T2']}

        self.solution = Solution(
            set_of_groups=groups,
            set_of_tours=tours,
            group_to_tour=GroupToTourAssignment(assignment=group_tour_assignment),
            bus_to_tours=BusToToursAssignment(assignment=bus_tour_assignment),
        )

        self.the_problem = SchoolBusProblem(
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
            buses={Bus(name="bus1", capacity=10), Bus(name="bus2", capacity=10)},
        )

        ElementSolution.the_problem = self.the_problem
        self.element_solution = ElementSolution(the_solution=self.solution)
        self.element = self.element_solution.get_element()

    def test_merge_groups_improves_solution(self):
        new_element, actual_number_of_changes = merge_groups(self.element)
        new_element_solution = ElementSolution.from_code(new_element.element_id)
        new_solution = new_element_solution.the_solution

        expected_group_assignment = {
            'g1': 'Merged tour 1',
            'g2': 'Merged tour 1',
            'g3': 'Merged tour 1',
        }
        self.assertDictEqual(
            expected_group_assignment, new_solution.group_to_tour.assignment
        )

        expected_bus_assignment = {'bus1': ['Merged tour 1'], 'bus2': ['Merged tour 1']}
        self.assertDictEqual(
            expected_bus_assignment, new_solution.bus_to_tours.assignment
        )

        expected_tours = {'Merged tour 1'}
        actual_tours = {tour.the_id for tour in new_solution.set_of_tours}
        self.assertSetEqual(expected_tours, actual_tours)


if __name__ == "__main__":
    unittest.main()
