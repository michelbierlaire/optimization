import dominate
from dominate.tags import div, h1, h2, h3, link, meta, section
from dominate.util import raw
from tabulate import tabulate

from biogeme_optimization.pareto import Pareto
from biogeme_optimization.school_bus import SchoolBusProblem, Solution
from biogeme_optimization.school_bus.indicators import TimeTable


def summary_report(the_problem: SchoolBusProblem, the_pareto: Pareto) -> str:

    headers = [
        '',
        'Total travel time',
        'Maximum travel time',
        'Total time late',
        'Maximum time late',
        'Total time early',
        'Maximum time early',
    ]
    rows = []

    solution_number = 0
    for a_pareto_solution in the_pareto.pareto:
        solution_number += 1
        a_solution = Solution.from_code(the_code=a_pareto_solution.element_id)
        timetable = TimeTable(the_problem=the_problem, the_solution=a_solution)
        rows.append(
            [
                f'Solution {solution_number}',
                timetable.total_travel_time,
                timetable.maximum_travel_time,
                timetable.total_late_arrivals,
                timetable.maximum_late_arrivals,
                timetable.total_early_arrivals,
                timetable.maximum_early_arrivals,
            ]
        )

    return tabulate(rows, headers=headers, tablefmt='html')


def html_report(the_problem: SchoolBusProblem, the_pareto: Pareto) -> str:

    doc = dominate.document(title='Non dominated solutions of the school bus problem')

    # --- HEAD ---
    with doc.head:
        # Bootstrap CSS

        link(
            rel="stylesheet",
            href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css",
        )
        meta(charset="utf-8")

    # --- BODY ---
    with doc:
        with div(cls="container my-5"):
            # Title
            h1(
                "Summary of the objective functions",
                cls="mb-4 text-center bg-light p-3 rounded",
            )
            summary = summary_report(the_problem=the_problem, the_pareto=the_pareto)
            section(cls="mb-5").add(div(raw(summary), cls="my-4"))

            solution_number = 0
            for a_pareto_solution in the_pareto.pareto:
                solution_number += 1
                a_solution = Solution.from_code(the_code=a_pareto_solution.element_id)
                timetable = TimeTable(the_problem=the_problem, the_solution=a_solution)

                with section(cls="mb-5"):
                    h1(
                        f'Solution {solution_number}',
                        cls="mb-4 text-center bg-light p-3 rounded",
                    )

                    h2("Bus schedule", cls="h4 mb-3")
                    schedule_per_bus = timetable.generate_buses_timetable(
                        table_format='html'
                    )
                    for bus, schedule in schedule_per_bus.items():
                        if schedule is not None:
                            h3(bus)
                            # Directly add the schedule table, wrapped in a div with spacing
                            section.add(div(raw(schedule), cls="my-4"))

                    h2("Children schedule", cls="h4 mb-3")
                    children_schedule = timetable.print_time_performance(
                        table_format='html'
                    )
                    section.add(div(raw(children_schedule), cls="my-4"))

                    h2("Objective functions", cls="h4 mb-3")
                    objectives = timetable.print_main_indicators(table_format='html')
                    section.add(div(raw(objectives), cls="my-4"))

    return str(doc)
