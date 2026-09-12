"""Implementation of the school bus routing and scheduling problem inspired by Spada et al. (2005)

Functions to calculate various indicators

Spada M., Bierlaire M., Liebling T. (2005). Decision-aid methodology
for the school bus routing and scheduling problem. Transportation Science 39 (4):477-490
https://dx.doi.org/10.1287/trsc.1040.0096

Michel Bierlaire
Mon May 26 2025, 10:38:44
"""

from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta

from tabulate import tabulate

from .data_classes import Destination, Origin, SchoolBusProblem
from .decision_variables import ChildrenGroup, Solution, Tour


def elapsed_time(start: datetime, end: datetime) -> float:
    return (end - start).total_seconds() / 60.0


@dataclass
class TimePerformance:
    boarding_time: datetime | None
    alighting_time: datetime | None
    arrival_time_deviation: float | None
    school_starting_time: datetime | None


@dataclass
class Stop:
    origin: Origin | None
    school: Destination | None
    time: datetime
    number_of_persons_on_board_before_the_stop: float
    children_boarding: set[ChildrenGroup]
    children_alighting: set[ChildrenGroup]

    @property
    def number_boarding(self) -> float:
        return sum(group.group_size for group in self.children_boarding)

    @property
    def number_alighting(self) -> float:
        return sum(group.group_size for group in self.children_alighting)

    @property
    def node_name(self) -> str:
        if self.origin is not None:
            return self.origin.name
        if self.school is None:
            raise ValueError(
                'The stop has not been initialized properly. It is neither an origin or a destination'
            )
        return self.school.name


class TimeTable:
    def __init__(self, the_problem: SchoolBusProblem, the_solution: Solution):
        self.the_problem = the_problem
        self.the_solution = the_solution
        self.time_performance: dict[ChildrenGroup, TimePerformance] = {
            group: TimePerformance(
                boarding_time=None,
                alighting_time=None,
                arrival_time_deviation=None,
                school_starting_time=None,
            )
            for group in the_solution.set_of_groups
        }
        self.time_performance_calculated: bool = False
        self._total_travel_time: float | None = None
        self._maximum_travel_time: float | None = None
        self._total_late_arrivals: float | None = None
        self._maximum_late_arrivals: float | None = None
        self._total_early_arrivals: float | None = None
        self._maximum_early_arrivals: float | None = None
        ok, msg = self.check_groups()
        if not ok:
            raise ValueError(msg)

    @property
    def total_travel_time(self) -> float:
        if self._total_travel_time is None:
            self._calculate_time_performance()
        return self._total_travel_time

    @property
    def maximum_travel_time(self) -> float:
        if self._maximum_travel_time is None:
            self._calculate_time_performance()
        return self._maximum_travel_time

    @property
    def total_late_arrivals(self) -> float:
        if self._total_late_arrivals is None:
            self._calculate_time_performance()
        return self._total_late_arrivals

    @property
    def maximum_late_arrivals(self) -> float:
        if self._maximum_late_arrivals is None:
            self._calculate_time_performance()
        return self._maximum_late_arrivals

    @property
    def total_early_arrivals(self) -> float:
        if self._total_early_arrivals is None:
            self._calculate_time_performance()
        return self._total_early_arrivals

    @property
    def maximum_early_arrivals(self) -> float:
        if self._maximum_early_arrivals is None:
            self._calculate_time_performance()
        return self._maximum_early_arrivals

    def check_groups(self) -> tuple[bool, str | None]:
        unknown_origins = [
            group.origin_name
            for group in self.the_solution.set_of_groups
            if self.the_problem.get_origin(group.origin_name) is None
        ]
        unknown_destinations = [
            group.destination_name
            for group in self.the_solution.set_of_groups
            if self.the_problem.get_school(group.destination_name) is None
        ]
        if unknown_origins and unknown_destinations:
            return (
                False,
                f'Unknown origins: {unknown_origins}. Unknown destinations: {unknown_destinations}',
            )
        return True, None

    def previous_stop(
        self, current_stop: Stop, previous_node: str, groups: set[ChildrenGroup]
    ) -> Stop:
        """This function calculates the time propagation along a tour, as well as the implications for the groups
        of children"""
        travel_time: float = self.the_problem.get_travel_time(
            previous_node, current_stop.node_name
        )
        previous_node_time = current_stop.time - timedelta(minutes=travel_time)

        children_alighting: set[ChildrenGroup] = {
            group for group in groups if group.destination_name == previous_node
        }
        number_of_children_alighting = sum(
            group.group_size for group in children_alighting
        )

        children_boarding: set[ChildrenGroup] = {
            group for group in groups if group.origin_name == previous_node
        }
        number_of_children_boarding = sum(
            group.group_size for group in children_boarding
        )
        after_the_stop = current_stop.number_of_persons_on_board_before_the_stop
        net_modification = number_of_children_boarding - number_of_children_alighting
        before_the_stop = after_the_stop - net_modification
        return Stop(
            origin=self.the_problem.get_origin(previous_node),
            school=self.the_problem.get_school(previous_node),
            time=previous_node_time,
            children_boarding=children_boarding,
            children_alighting=children_alighting,
            number_of_persons_on_board_before_the_stop=before_the_stop,
        )

    def dict_of_stops_for_tour(
        self, tour_name: str, arrival_time: datetime
    ) -> OrderedDict[str, Stop]:
        """Generate the list of stops of a tour. It works backward starting from the destination,
        where the arrival time is known."""
        the_tour = self.the_solution.get_tour(tour_name=tour_name)
        children_groups = self.the_solution.get_groups_for_tour(tour_name=tour_name)
        last_node: str = the_tour.list_of_nodes[-1]
        # The last_node must be a school
        the_school = self.the_problem.get_school(school_name=last_node)
        if the_school is None:
            error_msg = (
                f'The last node of tour {tour_name} [{last_node}] is not a destination.'
            )
            raise ValueError(error_msg)

        the_current_time: datetime = arrival_time
        children_boarding: set[ChildrenGroup] = {
            group for group in children_groups if group.origin_name == last_node
        }
        if children_boarding:
            raise ValueError(
                f'Node {last_node} is the last stop of tour {tour_name}. Nobody should board.'
            )  # It is the last stop. Nobody boards
        children_alighting: set[ChildrenGroup] = {
            group
            for group in children_groups
            if group.destination_name == the_school.name
        }
        last_stop = Stop(
            origin=self.the_problem.get_origin(origin_name=last_node),
            school=the_school,
            time=the_current_time,
            number_of_persons_on_board_before_the_stop=sum(
                group.group_size for group in children_alighting
            ),
            children_boarding=children_boarding,
            children_alighting=children_alighting,
        )
        reversed_stops = [last_stop]

        # Generate stops in reverse order
        for prev_node, curr_node in reversed(
            list(zip(the_tour.list_of_nodes, the_tour.list_of_nodes[1:]))
        ):
            current_stop = reversed_stops[-1]
            stop = self.previous_stop(
                current_stop=current_stop,
                previous_node=prev_node,
                groups=children_groups,
            )
            reversed_stops.append(stop)

        # Reverse the list to restore chronological order
        stops = list(reversed(reversed_stops))
        return OrderedDict((f'{tour_name}_{stop.node_name}', stop) for stop in stops)

    def dict_of_stops_for_bus(self, bus_name: str) -> OrderedDict[str, Stop]:
        the_bus_tours: list[Tour] = self.the_solution.get_bus_tours(bus_name=bus_name)
        next_tour = None
        list_of_dict_of_stops = []
        for tour in reversed(the_bus_tours):
            current_tour = tour
            node_name = current_tour.list_of_nodes[-1]
            school = self.the_problem.get_school(school_name=node_name)
            if next_tour is None:
                # The last tour of the bus must arrive on time at the school
                arrival_time_for_current_tour = school.arrival_time
                list_of_stops = self.dict_of_stops_for_tour(
                    tour_name=tour.the_id, arrival_time=arrival_time_for_current_tour
                )
                list_of_dict_of_stops.append(list_of_stops)
                first_stop: Stop = next(iter(list_of_stops.values()))
                time_first_stop_next_tour: datetime = first_stop.time
                next_tour = current_tour
            else:
                end_of_current_tour = current_tour.list_of_nodes[-1]
                start_of_next_tour = next_tour.list_of_nodes[0]
                travel_time = self.the_problem.get_travel_time(
                    end_of_current_tour, start_of_next_tour
                )
                arrival_time_for_current_tour: datetime = (
                    time_first_stop_next_tour - timedelta(minutes=travel_time)
                )
                list_of_stops = self.dict_of_stops_for_tour(
                    tour_name=tour.the_id, arrival_time=arrival_time_for_current_tour
                )
                list_of_dict_of_stops.append(list_of_stops)
                first_stop: Stop = next(iter(list_of_stops.values()))
                time_first_stop_next_tour = first_stop.time
                next_tour = current_tour
        return OrderedDict(
            (key, value)
            for the_dict in reversed(list_of_dict_of_stops)
            for key, value in the_dict.items()
        )

    def generate_buses_timetable(self, table_format='simple') -> dict[str, str | None]:
        """Generates the timetable for all buses"""

        result = {}
        headers = [
            'Location',
            'Time',
            'Boarding',
            'Alighting',
            'In the bus (before the stop)',
        ]
        for bus in self.the_problem.buses:
            stops: OrderedDict[str, Stop] = self.dict_of_stops_for_bus(
                bus_name=bus.name
            )
            if not stops:
                result[bus.name] = None
                break
            rows = [
                [
                    stop.node_name,
                    stop.time.strftime("%H:%M"),
                    stop.number_boarding,
                    stop.number_alighting,
                    stop.number_of_persons_on_board_before_the_stop,
                ]
                for stop in stops.values()
            ]
            result[bus.name] = tabulate(rows, headers=headers, tablefmt=table_format)

        if all(value is None for value in result.values()):
            raise ValueError('No bus timetable has been generated')
        return result

    def print_buses_timetable(self, table_format='simple') -> str:
        """Return all generated bus timetables as a printable string.

        This method keeps the historical ``print_buses_timetable`` API while
        delegating timetable generation to :meth:`generate_buses_timetable`.
        """
        schedules = self.generate_buses_timetable(table_format=table_format)
        sections = []
        for bus_name, schedule in schedules.items():
            if schedule is not None:
                sections.extend((bus_name, '~' * len(bus_name), schedule))
        return '\n'.join(sections)

    def print_time_performance(self, table_format='simple') -> str:
        if not self.time_performance_calculated:
            self._calculate_time_performance()
        headers = [
            'Group',
            'School starting time',
            'Boarding time',
            'Alighting time',
            'Travel time',
            'Late (-)/ early (+) arrival',
        ]
        rows = [
            [
                group.the_id,
                performance.school_starting_time.strftime("%H:%M"),
                performance.boarding_time.strftime("%H:%M"),
                performance.alighting_time.strftime("%H:%M"),
                elapsed_time(performance.boarding_time, performance.alighting_time),
                performance.arrival_time_deviation,
            ]
            for group, performance in self.time_performance.items()
        ]
        return tabulate(rows, headers=headers, tablefmt=table_format)

    def _calculate_time_performance(self):
        if self.time_performance_calculated:
            raise RuntimeError('Time performance has already been calculated.')

        for bus in self.the_problem.buses:
            stops: OrderedDict[str, Stop] = self.dict_of_stops_for_bus(
                bus_name=bus.name
            )
            for stop in stops.values():
                # Boarding children
                for group in stop.children_boarding:
                    the_performance = self.time_performance.get(group)
                    if the_performance is None:
                        raise ValueError(
                            f'Group {group.the_id} not properly initialized.'
                        )
                    the_performance.boarding_time = stop.time
                # Alighting children
                for group in stop.children_alighting:
                    the_performance = self.time_performance.get(group)
                    if the_performance is None:
                        raise ValueError(
                            f'Group {group.the_id} not properly initialized.'
                        )
                    the_performance.alighting_time = stop.time
                    the_performance.arrival_time_deviation = elapsed_time(
                        stop.time, stop.school.arrival_time
                    )
                    the_performance.school_starting_time = stop.school.arrival_time

        self._total_travel_time = 0
        self._maximum_travel_time = 0
        self._total_late_arrivals = 0
        self._maximum_late_arrivals = 0
        self._total_early_arrivals = 0
        self._maximum_early_arrivals = 0
        for group, performance in self.time_performance.items():
            travel_time = elapsed_time(
                performance.boarding_time, performance.alighting_time
            )
            self._total_travel_time += travel_time * group.group_size
            self._maximum_travel_time = max(self._maximum_travel_time, travel_time)

            time_early = max(performance.arrival_time_deviation, 0)
            self._total_early_arrivals += time_early * group.group_size
            self._maximum_early_arrivals = max(self._maximum_early_arrivals, time_early)

            time_late = max(-performance.arrival_time_deviation, 0)
            self._total_late_arrivals += time_late * group.group_size
            self._maximum_late_arrivals = max(self._maximum_late_arrivals, time_late)

    def print_main_indicators(self, table_format: str = 'plain'):
        rows = [
            ['Total travel time', self.total_travel_time],
            ['Maximum travel time', self.maximum_travel_time],
            ['Total time late', self.total_late_arrivals],
            ['Maximum time late', self.maximum_late_arrivals],
            ['Total time early', self.total_early_arrivals],
            ['Maximum time early', self.maximum_early_arrivals],
        ]
        return tabulate(rows, tablefmt=table_format)


def describe_solution(the_problem: SchoolBusProblem, the_solution: Solution) -> str:
    timetable = TimeTable(the_problem=the_problem, the_solution=the_solution)
    result = '=' * 70 + '\n'
    result += 'Solution for the school bus problem\n'
    result += '=' * 70 + '\n'
    bus_schedule = timetable.generate_buses_timetable()
    for bus, schedule in bus_schedule.items():
        if schedule is not None:
            result += bus + '\n'
            result += '~' * len(bus) + '\n'
            result += schedule + '\n'
    result += timetable.print_time_performance() + '\n'
    result += timetable.print_main_indicators() + '\n'
    return result
