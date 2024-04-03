"""Illustration of the simplex tableau

See exercises 03-simplex_q06

Michel Bierlaire
Wed Apr 3 13:34:27 2024
"""

import numpy as np
from biogeme_optimization.teaching.tableau import SimplexTableau, RowColumn

# Consider a linear optimization problem in standard form, defined by

matrix_a = np.array([[1, -1, 1, 0], [1, 1, 0, 1]])
right_hand_side = np.array([2, 6])
coefficients = np.array([-2, -1, 0, 0])

# Suppose that variables 0 and 3 are in the basis. The corresponding tableau can be built as follows.
the_basic_indices = [0, 3]
tableau = SimplexTableau.from_basis(
    matrix=matrix_a,
    right_hand_side=right_hand_side,
    objective=coefficients,
    basic_indices=the_basic_indices,
)

# Here is the tableau
print(tableau)

# We check the basic variables
variables_in_the_basis: list[RowColumn] = tableau.identify_basic_variables()
for element in variables_in_the_basis:
    print(f'Basic variable {element.column} corresponds to row {element.row}.')

# The corresponding feasible solution.
print(f'Feasible basic solution: {tableau.feasible_basic_solution}.')

# Value of the objective function
print(f'Objective function: {tableau.value_objective_function}')

# Reduced costs
reduced_costs: dict[int, float] = tableau.reduced_costs
print(f'Reduced costs of non basic variables: {reduced_costs}.')

# Basic directions
basic_directions: dict[int, np.ndarray] = tableau.basic_directions
print(f'Basic directions of non basic variables: {basic_directions}')

# The same information are available from the report.
underline_start = '\033[4m'
underline_end = '\033[0m'
print(f'{underline_start}Report of the tableau{underline_end}')
print(tableau.report())

# Now, we pivot the tableau.
pivot = RowColumn(row=1, column=1)
tableau.pivoting(pivot=pivot)
print(f'{underline_start}Report of the pivoted tableau{underline_end}')
print(tableau.report())
