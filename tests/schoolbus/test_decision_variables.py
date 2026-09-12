import unittest

from icecream import ic

from biogeme_optimization.school_bus.decision_variables import (
    BusToToursAssignment,
    ChildrenGroup,
    GroupToTourAssignment,
    Solution,
    Tour,
)


class TestDecisionEncoding(unittest.TestCase):

    def test_children_group(self):
        children_group = ChildrenGroup(
            group_id='group_1',
            origin_name='orig',
            destination_name='dest',
            size=4,
        )
        the_code = children_group.generate_code()
        self.assertEqual(the_code, 'ChildrenGroup<group_1>[orig-dest-4]')
        parsed = ChildrenGroup.from_code(the_code)
        self.assertEqual(parsed.the_id, 'group_1')
        self.assertEqual(parsed.origin_name, 'orig')
        self.assertEqual(parsed.destination_name, 'dest')
        self.assertEqual(parsed.group_size, 4)

    def test_tour(self):
        tour = Tour(tour_id='the_tour', list_of_nodes=['A', 'B', 'C'])
        the_code = tour.generate_code()
        self.assertEqual(the_code, 'Tour<the_tour>[A-B-C]')
        parsed = Tour.from_code(the_code)
        self.assertEqual(parsed.the_id, 'the_tour')
        self.assertListEqual(parsed.list_of_nodes, ['A', 'B', 'C'])

    def test_group_tour_assignment(self):
        group_tour_assignment = GroupToTourAssignment(
            {'group_1': 'tour_1', 'group_2': 'tour_1', 'group_3': 'tour_2'}
        )
        the_code = group_tour_assignment.generate_code()
        expected_code = 'GroupToTourAssignment<GroupToTour>[(group_1-tour_1),(group_2-tour_1),(group_3-tour_2)]'
        self.assertEqual(the_code, expected_code)
        parsed = GroupToTourAssignment.from_code(the_code)
        self.assertDictEqual(group_tour_assignment.assignment, parsed.assignment)

    def test_tour_bus_assignment(self):
        tour_bus_assignment = BusToToursAssignment(
            {'bus_1': ['tour_1', 'tour_2'], 'bus_2': ['tour_3']}
        )
        the_code = tour_bus_assignment.generate_code()
        expected_code = 'BusToToursAssignment<BusToTours>[(bus_1-tour_1),(bus_1-tour_2),(bus_2-tour_3)]'
        self.assertEqual(the_code, expected_code)
        parsed = BusToToursAssignment.from_code(the_code)
        self.assertDictEqual(tour_bus_assignment.assignment, parsed.assignment)

    def test_tour_bus_assignment_one_empty(self):
        tour_bus_assignment = BusToToursAssignment(
            {'bus_1': ['tour_1', 'tour_2'], 'bus_2': []}
        )
        the_code = tour_bus_assignment.generate_code()
        ic(the_code)
        expected_code = (
            'BusToToursAssignment<BusToTours>[(bus_1-tour_1),(bus_1-tour_2)]'
        )
        self.assertEqual(the_code, expected_code)
        parsed = BusToToursAssignment.from_code(the_code)
        # The compact code stores only bus/tour pairs. Empty buses are restored
        # by Solution.update_buses when the problem's buses are available.
        self.assertDictEqual(
            {'bus_1': ['tour_1', 'tour_2']}, parsed.assignment
        )

    def test_solution(self):
        children_group_1 = ChildrenGroup(
            group_id='group_1',
            origin_name='orig_1',
            destination_name='dest_1',
            size=4,
        )
        children_group_2 = ChildrenGroup(
            group_id='group_2',
            origin_name='orig_2',
            destination_name='dest_2',
            size=6,
        )
        tour_1 = Tour(tour_id='tour_1', list_of_nodes=['orig_1', 'A', 'dest_1'])
        tour_2 = Tour(tour_id='tour_2', list_of_nodes=['orig_2', 'B', 'dest_2'])
        group_tour_assignment = GroupToTourAssignment(
            {'group_1': 'tour_1', 'group_2': 'tour_2'}
        )
        tour_bus_assignment = BusToToursAssignment(
            {'bus_1': ['tour_1'], 'bus_2': ['tour_2']}
        )
        solution = Solution(
            set_of_groups={children_group_1, children_group_2},
            set_of_tours={tour_1, tour_2},
            group_to_tour=group_tour_assignment,
            bus_to_tours=tour_bus_assignment,
        )
        the_code = solution.generate_code()
        expected_code = (
            'Solution[ChildrenList[ChildrenGroup<group_1>[orig_1-dest_1-4],ChildrenGroup<group_2>[orig_2-dest_2-6]]|'
            'TourList[Tour<tour_1>[orig_1-A-dest_1],Tour<tour_2>[orig_2-B-dest_2]]|'
            'GroupToTourAssignment<GroupToTour>[(group_1-tour_1),(group_2-tour_2)]|'
            'BusToToursAssignment<BusToTours>[(bus_1-tour_1),(bus_2-tour_2)]]'
        )
        self.assertEqual(the_code, expected_code)
        parsed = Solution.from_code(the_code)
        self.assertIn(children_group_1, parsed.set_of_groups)
        self.assertIn(children_group_2, parsed.set_of_groups)
        self.assertIn(tour_1, parsed.set_of_tours)
        self.assertIn(tour_2, parsed.set_of_tours)
        self.assertDictEqual(
            group_tour_assignment.assignment, parsed.group_to_tour.assignment
        )
        self.assertDictEqual(
            tour_bus_assignment.assignment, parsed.bus_to_tours.assignment
        )


if __name__ == '__main__':
    unittest.main()
