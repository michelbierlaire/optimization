"""Illustration of the simplex algorithm

Michel Bierlaire
Tue Apr 2 16:22:24 2024
"""

import numpy as np

from biogeme_optimization.teaching.simplex import SimplexAlgorithm
from biogeme_optimization.teaching.linear_optimization import LinearOptimization
from biogeme_optimization.teaching.linear_constraints import (
    draw_polyhedron_standard_form,
    LabeledPoint,
    LabeledDirection,
)

standard_a = np.array([[2, 1, 1, 0], [1, -1, 0, 1]])
standard_b = np.array([6, 2])
standard_c = np.array([-4, 3, 0, 0])

the_problem = LinearOptimization(
    objective=standard_c, constraint_matrix=standard_a, right_hand_side=standard_b
)
initial_basis = [2, 3]

# Initialize the algorithm
algorithm = SimplexAlgorithm(
    optimization_problem=the_problem, initial_basis=initial_basis
)

# Run the algorithm and store the reports of each iteration.
iterations = [iterate_report for iterate_report in algorithm]

# Print the result.
print(f'Optimal solution: {algorithm.solution}')
print(f'Exit status: {algorithm.stopping_cause}')

# Analyze the iterations
for iter in iterations:
    print(iter)

# Plot the iterates on the polyhedron.
points = [LabeledPoint(coordinates=iter.iterate) for iter in iterations]
directions = [
    LabeledDirection(
        start=iter.iterate,
        direction=iter.direction,
        label=f'It. {iter.iteration_number}',
    )
    for iter in iterations
    if iter.direction is not None
]

draw_polyhedron_standard_form(
    matrix_a=standard_a,
    vector_b=standard_b,
    objective_coefficients=standard_c,
    points=points,
    directions=directions,
    level_lines=[-26.0 / 3.0, 0, 26.0 / 3.0],
)
