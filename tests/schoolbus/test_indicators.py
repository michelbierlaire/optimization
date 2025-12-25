import unittest
from datetime import datetime

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
from biogeme_optimization.school_bus.indicators import TimeTable


class TestTimeTable(unittest.TestCase):
    def setUp(self):
        # Define origins and destinations
        origins = {Origin(name=name) for name in ["1", "2", "3"]}
        destinations = {
            Destination(name="2", arrival_time=datetime(2025, 6, 1, 8, 0)),
            Destination(name="4", arrival_time=datetime(2025, 6, 1, 8, 0)),
        }
        travel_times = {
            ('1', '2'): 5,
            ('2', '3'): 10,
            ('3', '4'): 15,
            ('1', '4'): 30,
        }
        od_table = {
            ('1', '2'): 3,
            ('1', '4'): 5,
            ('2', '4'): 2,
            ('3', '4'): 1,
        }
        bus = Bus(name='the_bus', capacity=10)

        # Define problem and solution
        self.problem = SchoolBusProblem(
            origins=origins,
            schools=destinations,
            travel_times_in_minutes=travel_times,
            od_table=od_table,
            buses={bus},
        )

        tour = Tour(list_of_nodes=["1", "2", "3", "4"], tour_id='the_tour')
        tour_2 = Tour(list_of_nodes=["1", "2"], tour_id='the_second_tour')
        # Define groups
        groups = {
            ChildrenGroup(
                group_id='1_2',
                origin_name="1",
                destination_name="2",
                size=3,
            ),
            ChildrenGroup(
                group_id='1_4',
                origin_name="1",
                destination_name="4",
                size=5,
            ),
            ChildrenGroup(
                group_id='2_4',
                origin_name="2",
                destination_name="4",
                size=2,
            ),
            ChildrenGroup(
                group_id='3_4',
                origin_name="3",
                destination_name="4",
                size=1,
            ),
        }
        group_to_tour_dict = {
            '1_2': 'the_second_tour',
            '1_4': 'the_tour',
            '2_4': 'the_tour',
            '3_4': 'the_tour',
        }
        bus_to_tours_dict = {'the_bus': ['the_tour', 'the_second_tour']}

        self.solution = Solution(
            set_of_groups=groups,
            set_of_tours={tour, tour_2},
            group_to_tour=GroupToTourAssignment(assignment=group_to_tour_dict),
            bus_to_tours=BusToToursAssignment(assignment=bus_to_tours_dict),
        )

        # Scenario for the bus
        # Tour 1: groups 1_4, 2_4, 3_4
        #   Stop 1: 6:55  boarding group 1_4 (5 child.)
        #   Stop 2: 7:00  boarding group 2_4 (2 child.)
        #   Stop 3: 7:10  boarding group 3_4 (1 child)
        #   Stop 4: 7:25  alighting 1_4, 2_4, 3_4 (8 child.)
        # Tour 2: group 1_2
        #   Stop 1: 7:55  boarding group 1_2 (3 child.)
        #   Stop 2: 8:00  alighting group 1_2 (3 child.)
        self.timetable = TimeTable(self.problem, self.solution)

    def test_list_of_stops_for_tour(self):
        arrival_time = datetime(2025, 6, 1, 7, 25)
        stops = self.timetable.dict_of_stops_for_tour('the_tour', arrival_time)

        # There should be 4 stops
        self.assertEqual(len(stops), 4)

        # Stop 4: 8 children alight
        self.assertEqual(stops['the_tour_4'].node_name, '4')
        self.assertEqual(stops['the_tour_4'].time, datetime(2025, 6, 1, 7, 25))
        self.assertEqual(
            sum(g.group_size for g in stops['the_tour_4'].children_alighting), 8
        )  # 5+2+1
        self.assertEqual(
            stops['the_tour_4'].number_of_persons_on_board_before_the_stop, 8
        )

        # Stop 3: 1 child board
        self.assertEqual(stops['the_tour_3'].node_name, '3')
        self.assertEqual(stops['the_tour_3'].time, datetime(2025, 6, 1, 7, 10))
        self.assertEqual(
            sum(g.group_size for g in stops['the_tour_3'].children_boarding), 1
        )  # 1→4
        self.assertEqual(
            sum(g.group_size for g in stops['the_tour_3'].children_alighting), 0
        )
        self.assertEqual(
            stops['the_tour_3'].number_of_persons_on_board_before_the_stop, 7
        )

        # Stop 2: 2 children board
        self.assertEqual(stops['the_tour_2'].node_name, '2')
        self.assertEqual(stops['the_tour_2'].time, datetime(2025, 6, 1, 7, 0))
        self.assertEqual(
            sum(g.group_size for g in stops['the_tour_2'].children_boarding), 2
        )  # 2→4
        self.assertEqual(
            sum(g.group_size for g in stops['the_tour_2'].children_alighting), 0
        )  # 1→2
        self.assertEqual(
            stops['the_tour_2'].number_of_persons_on_board_before_the_stop, 5
        )  # 1→4 and 2→4 and 3→4

        # Stop 1: 8 children board.
        self.assertEqual(stops['the_tour_1'].node_name, '1')
        self.assertEqual(stops['the_tour_1'].time, datetime(2025, 6, 1, 6, 55))
        self.assertEqual(
            sum(g.group_size for g in stops['the_tour_1'].children_boarding), 5
        )  # 3→2, 5→4
        self.assertEqual(
            stops['the_tour_1'].number_of_persons_on_board_before_the_stop, 0
        )

    def test_list_of_stops_for_bus(self):
        stops = self.timetable.dict_of_stops_for_bus(bus_name='the_bus')
        list_of_nodes = [value.node_name for _, value in stops.items()]
        expected_list = ['1', '2', '3', '4', '1', '2']
        self.assertListEqual(list_of_nodes, expected_list)
        list_of_times = [value.time for _, value in stops.items()]
        expected_list_of_times = [
            datetime(2025, 6, 1, 6, 55),
            datetime(2025, 6, 1, 7, 0),
            datetime(2025, 6, 1, 7, 10),
            datetime(2025, 6, 1, 7, 25),
            datetime(2025, 6, 1, 7, 55),
            datetime(2025, 6, 1, 8, 0),
        ]
        self.assertListEqual(list_of_times, expected_list_of_times)

    def test_print_timetable(self):
        printed_time_table = self.timetable.print_buses_timetable()
        self.assertIsInstance(printed_time_table, str)
        self.assertTrue(printed_time_table.strip())

    def test_print_performance(self):
        printed_performance = self.timetable.print_time_performance()
        self.assertIsInstance(printed_performance, str)
        self.assertTrue(printed_performance.strip())

    def test_print_indicators(self):
        printed_indicators = self.timetable.print_main_indicators()
        self.assertIsInstance(printed_indicators, str)
        self.assertTrue(printed_indicators.strip())


if __name__ == '__main__':
    unittest.main()
