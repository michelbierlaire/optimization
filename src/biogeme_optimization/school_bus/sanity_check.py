"""Implementation of the school bus routing and scheduling problem inspired by Spada et al. (2005)

Functions to verify that a solution is consistent with the problem

Spada M., Bierlaire M., Liebling T. (2005). Decision-aid methodology
for the school bus routing and scheduling problem. Transportation Science 39 (4):477-490
https://dx.doi.org/10.1287/trsc.1040.0096

Michel Bierlaire
Wed Jun 04 2025, 10:16:45
"""

from collections import defaultdict

from .data_classes import SchoolBusProblem
from .decision_variables import Solution


class SanityCheck:
    def __init__(self, the_problem: SchoolBusProblem, the_solution: Solution):
        self.the_problem = the_problem
        self.the_solution = the_solution

    def all_checks(self) -> tuple[bool, str]:
        """Performs all checks"""
        all_ok = True
        messages = []

        checks = [
            self.check_demand,
            self.check_buses,
            self.the_solution.check_tours_validity,
        ]

        for check in checks:
            ok, msg = check()
            all_ok = all_ok and ok
            if not ok:
                messages.append(msg)

        return all_ok, '\n'.join(messages)

    def check_demand(self) -> tuple[bool, str | None]:
        """Verify if the whole demand has been assigned to children group"""
        demand_by_od = defaultdict(float)
        for group in self.the_solution.set_of_groups:
            key = (group.origin_name, group.destination_name)
            demand_by_od[key] += group.group_size

        pairs_in_problem_but_not_in_solution = set(
            self.the_problem.od_table.keys()
        ) - set(demand_by_od.keys())
        msg_1 = (
            f'OD defined in problem but not in solution: {pairs_in_problem_but_not_in_solution}'
            if pairs_in_problem_but_not_in_solution
            else ''
        )

        pairs_in_solution_but_not_in_problem = set(demand_by_od.keys()) - set(
            self.the_problem.od_table.keys()
        )
        msg_2 = (
            f'OD defined in solution but not in problem: {pairs_in_solution_but_not_in_problem}'
            if pairs_in_solution_but_not_in_problem
            else ''
        )

        common_keys = set(self.the_problem.od_table) & set(demand_by_od)
        common_keys_with_different_values = [
            key
            for key in common_keys
            if abs(self.the_problem.od_table[key] - demand_by_od[key]) > 1e-6
        ]
        msg_3 = (
            f'OD pairs with mismatched demand: {common_keys_with_different_values}'
            if common_keys_with_different_values
            else ''
        )

        if msg_1 == '' and msg_2 == '' and msg_3 == '':
            return True, None

        msg = ''
        if msg_1 != '':
            msg += f'{msg_1}\n'
        if msg_2 != '':
            msg += f'{msg_2}\n'
        if msg_3 != '':
            msg += f'{msg_3}\n'
        return False, msg

    def check_buses(self) -> tuple[bool, str | None]:
        """Check if the list of buses in the problem definition and on the solution match."""
        buses_in_solution_but_not_in_problem = set(
            self.the_solution.bus_to_tours.assignment.keys()
        ) - set(bus.name for bus in self.the_problem.buses)
        msg = (
            f'Buses defined in solution but not in problem: {buses_in_solution_but_not_in_problem}'
            if buses_in_solution_but_not_in_problem
            else ''
        )

        if msg == '':
            return True, None
        return False, msg
