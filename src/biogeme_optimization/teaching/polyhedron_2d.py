"""Class in charge of plotting a polygon, thst is a polyhedron in dimension 2.

Michel Bierlaire
Tue Mar 12 18:49:20 2024
"""

import itertools

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axis import Axis
from scipy.spatial import ConvexHull

from biogeme_optimization.teaching.linear_constraints import StandardCanonicalForms


def intersection(
    rows_a: np.ndarray,
    rows_b: np.ndarray,
) -> np.ndarray | None:
    """
    Calculates the intersection of two lines, thst is the solution of A x = b.

    :param rows_a: 2x2 matrix with the two rows corresponding to the two constraints.
    :param rows_b: vector of dimension 2 with the right hand side of the constraints.

    """
    try:
        return np.linalg.solve(rows_a, rows_b)
    except np.linalg.LinAlgError:
        return None


# %%
class Polyhedron2d:
    """Class in charge of plotting a polyhedron defining a polyhedron"""

    def __init__(
        self,
        polyhedron: StandardCanonicalForms,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
    ):
        if polyhedron.canonical_form.n_variables != 2:
            error_msg = (
                f'Plotting a polyhedron is valid only in dimension 2. '
                f'The polyhedron is in dimension {polyhedron.canonical_form.n_variables}'
            )
            raise ValueError(error_msg)

        self.polyhedron: StandardCanonicalForms = polyhedron
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

    def vertices(self) -> set[tuple[float, float]]:
        """Identifies the vertices of the polyhedron by calculating the intersections between each pair of
        constraints"""

        the_vertices = set()
        for i, j in itertools.combinations(
            range(len(self.polyhedron.canonical_matrix)), 2
        ):
            rows_from_a = np.array(
                [
                    self.polyhedron.canonical_matrix[i],
                    self.polyhedron.canonical_matrix[j],
                ]
            )
            elements_from_b = np.array(
                [
                    self.polyhedron.canonical_vector[i],
                    self.polyhedron.canonical_vector[j],
                ]
            )
            the_intersection = intersection(rows_a=rows_from_a, rows_b=elements_from_b)
            if (
                the_intersection is not None
                and self.polyhedron.canonical_form.is_feasible(
                    vector_x=the_intersection
                )
            ):
                the_vertices.add((the_intersection[0], the_intersection[1]))
        return the_vertices

    def plot_one_constraint(
        self,
        row_a: np.ndarray,
        elem_b: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculates the elements to plot one constraint.

        :param row_a: row of the matrix corresponding to the constraint.
        :param elem_b: right hand side of the constraint.

        """

        if np.abs(row_a[1]) > np.abs(row_a[0]):
            x_0 = np.linspace(self.x_min, self.x_max, 400)
            x_1 = (elem_b - row_a[0] * x_0) / row_a[1]
            mask = (x_1 >= self.y_min) & (x_1 <= self.y_max)
            x_0, x_1 = x_0[mask], x_1[mask]
        else:
            x_1 = np.linspace(self.y_min, self.y_max, 400)
            x_0 = (elem_b - row_a[1] * x_1) / row_a[0]
            mask = (x_0 >= self.x_min) & (x_0 <= self.x_max)
            x_0, x_1 = x_0[mask], x_1[mask]

        return x_0, x_1

    def plot_polyhedron(self, ax: Axis | None = None) -> Axis:
        """Plots the polyhedron Ax <= b in dimension 2. It assumes that it is bounded.

        :param ax: matplotlib axis object, in case the plot must be included in an existing plot.

        """
        if ax is None:
            fig, ax = plt.subplots()

        # Plot each line
        for row_a, elem_b in zip(
            self.polyhedron.canonical_matrix, self.polyhedron.canonical_vector
        ):
            x_0, x_1 = self.plot_one_constraint(
                row_a=row_a,
                elem_b=elem_b,
            )
            ax.plot(x_0, x_1, color='orange')

        # Plot the polyhedron itself (a polygon here)
        the_vertices = list(self.vertices())
        if the_vertices:
            all_vertices = np.array(the_vertices)
            hull = ConvexHull(all_vertices)
            for simplex in hull.simplices:
                ax.plot(all_vertices[simplex, 0], all_vertices[simplex, 1], 'orange')

            # Fill the polyhedron
            ax.fill(
                all_vertices[hull.vertices, 0],
                all_vertices[hull.vertices, 1],
                'lightblue',
                edgecolor='blue',
                alpha=0.5,
            )

            # Set equal scaling and labels
            ax.axis('equal')
            ax.set_xlabel('x_1')
            ax.set_ylabel('x_2')

            # Display the coordinate axes with a specific range
            ax.axhline(0, color='black', linewidth=0.5)
            ax.axvline(0, color='black', linewidth=0.5)

            return ax
