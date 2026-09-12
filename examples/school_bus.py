"""Run the school-bus variable-neighborhood-search demonstration."""

import logging
from datetime import datetime, timezone

from biogeme_optimization.school_bus import (
    Bus,
    Destination,
    Origin,
    SchoolBusProblem,
    Solution,
)
from biogeme_optimization.school_bus.element_pareto import ElementSolution
from biogeme_optimization.school_bus.indicators import describe_solution
from biogeme_optimization.school_bus.neighborhood import SchoolBus
from biogeme_optimization.school_bus.pareto_report import html_report
from biogeme_optimization.school_bus.starting_points import single_bus, taxi_solution
from biogeme_optimization.vns import ParetoClass, vns


def main() -> None:
    logger = logging.getLogger('biogeme_optimization')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(levelname)s] %(message)s ')
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.info('Example for the school bus problem')

    test_instance = SchoolBusProblem(
        schools={
            Destination(
                name='Central_school',
                arrival_time=datetime(
                    year=2025, month=6, day=9, hour=8, minute=30, tzinfo=timezone.utc
                ),
            ),
            Destination(
                name='Northeast_school',
                arrival_time=datetime(
                    year=2025, month=6, day=9, hour=8, minute=15, tzinfo=timezone.utc
                ),
            ),
            Destination(
                name='Southwest_school',
                arrival_time=datetime(
                    year=2025, month=6, day=9, hour=8, minute=15, tzinfo=timezone.utc
                ),
            ),
            Destination(
                name='Southeast_school',
                arrival_time=datetime(
                    year=2025, month=6, day=9, hour=8, minute=20, tzinfo=timezone.utc
                ),
            ),
        },
        origins={Origin(name='Village')},
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
        buses={Bus(name='Bus 1', capacity=50), Bus(name='Bus 2', capacity=50)},
    )

    one_solution = single_bus(the_problem=test_instance)
    two_solution = taxi_solution(the_problem=test_instance)
    print('Initial solution')
    print(describe_solution(the_solution=one_solution, the_problem=test_instance))

    the_pareto = ParetoClass(max_neighborhood=5, pareto_file='test_bus.pareto')
    the_problem = SchoolBus(the_problem=test_instance)

    ElementSolution.the_problem = test_instance
    first_solution = ElementSolution(one_solution)
    second_solution = ElementSolution(two_solution)
    the_pareto = vns(
        problem=the_problem,
        first_solutions=[first_solution.get_element(), second_solution.get_element()],
        pareto=the_pareto,
        number_of_neighbors=10,
    )

    print(f'Number of pareto solutions: {len(the_pareto.pareto)}')
    print(f'Number of considered solutions: {len(the_pareto.considered)}')
    for a_pareto_solution in the_pareto.pareto:
        a_solution = Solution.from_code(the_code=a_pareto_solution.element_id)
        print('-' * 70)
        print(a_solution)
        print(describe_solution(the_solution=a_solution, the_problem=test_instance))

    html = html_report(the_problem=test_instance, the_pareto=the_pareto)
    with open('report.html', 'w') as report_file:
        report_file.write(html)


if __name__ == '__main__':
    main()
