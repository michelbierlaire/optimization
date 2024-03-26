"""Identification of vertices and plotting of the polyhedron: illustration of usage
********************************************************************************

Michel Bierlaire
Mon Mar 11 13:19:43 2024
"""

import numpy as np
from matplotlib import pyplot as plt

from biogeme_optimization.teaching.polyhedron_2d import intersection, Polyhedron2d
from biogeme_optimization.teaching.linear_constraints import (
    AllConstraints,
    StandardCanonicalForms,
)

# We first test the function calculating the intersection.

# We expect (1, -4)
A = np.array([[1, 0], [0, 1]])
b = np.array([1, -4])
the_intersection = intersection(rows_a=A, rows_b=b)
print(f'{the_intersection=}')

# We expect (5, -4)
A = np.array([[1, 1], [0, 1]])
b = np.array([1, -4])
the_intersection = intersection(rows_a=A, rows_b=b)
print(f'{the_intersection=}')

# We expect None
A = np.array([[1, 1], [2, 2]])
b = np.array([1, -4])
the_intersection = intersection(rows_a=A, rows_b=b)
print(f'{the_intersection=}')

# We now test the calculation of the vertices.
# Example 3.39 from Bierlaire (2015), Figure 3.16
A = np.array([[1, 1], [1, -1], [-1, 0], [0, -1]])
b = np.array([1, 1, 0, 0])
polyhedron = AllConstraints.from_canonical_form(matrix=A, vector=b)
the_plot_polyhedron = Polyhedron2d(
    polyhedron=StandardCanonicalForms(constraints=polyhedron),
    x_min=-1.1,
    x_max=1.1,
    y_min=-1.1,
    y_max=1.1,
)
the_vertices = the_plot_polyhedron.vertices()
print(the_vertices)

ax = the_plot_polyhedron.plot_polyhedron()
plt.show()
