"""Class in charge of plotting a polygon, thst is a polyhedron in dimension 2.

Michel Bierlaire
Tue Mar 12 18:49:20 2024
"""

import itertools
import logging
from typing import NamedTuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axis import Axis
from scipy.spatial import ConvexHull, QhullError

from biogeme_optimization.teaching.linear_constraints import StandardCanonicalForms

logger = logging.getLogger(__name__)

RELATIVE_MARGIN = 1.01


class BoundingBox(NamedTuple):
    """Bounds on each coordinate."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float


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
        objective_coefficients: np.ndarray | None = None,
        bounding_box: BoundingBox | None = None,
    ) -> None:
        if polyhedron.canonical_form.n_variables != 2:
            error_msg = (
                f'Plotting a polyhedron is valid only in dimension 2. '
                f'The polyhedron is in dimension {polyhedron.canonical_form.n_variables}'
            )
            raise ValueError(error_msg)

        self.polyhedron: StandardCanonicalForms = polyhedron
        self.objective_coefficients: np.ndarray | None = objective_coefficients
        if self.objective_coefficients is not None:
            assert (
                len(self.objective_coefficients) == 2,
                f'Requires a vector of length 2, and not {len(self.objective_coefficients)}',
            )
        self.bounding_box: BoundingBox | None = bounding_box
        self.specify_ranges()

    def objective_function(self, x, y):
        return x * self.objective_coefficients[0] + y * self.objective_coefficients[1]

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
                )[0]
            ):
                the_vertices.add((the_intersection[0], the_intersection[1]))

        return the_vertices

    def specify_ranges(self) -> None:
        """
        Specifies the ranges, if not done yet.
        """
        the_vertices = np.array(list(self.vertices()))

        # Find the lowest values of each coordinate
        min_values = np.min(the_vertices, axis=0)

        # Find the largest values of each coordinate
        max_values = np.max(the_vertices, axis=0)

        x_min: float = min_values[0]
        x_max: float = max_values[0]
        range_x: float = x_max - x_min
        y_min: float = min_values[1]
        y_max: float = max_values[1]
        range_y: float = y_max - x_min

        absolute_margin = np.maximum(range_x * 0.01, range_y * 0.01)
        x_margins: float = np.maximum(absolute_margin, range_x * RELATIVE_MARGIN)
        y_margins = np.maximum(absolute_margin, range_y * RELATIVE_MARGIN)

        if self.bounding_box is None:
            self.bounding_box = BoundingBox(
                x_min=x_min - x_margins,
                x_max=x_max + x_margins,
                y_min=y_min - y_margins,
                y_max=y_max + y_margins,
            )
            return

        if self.bounding_box.x_min > x_min:
            logger.warning(
                f'The specified x_min [{self.bounding_box.x_min}] may exclude some part of the polyhedron. '
                f'Suggested value: {x_min - x_margins}.'
            )
        if self.bounding_box.x_max < x_max:
            logger.warning(
                f'The specified x_max [{self.bounding_box.x_max}] may exclude some part of the polyhedron. '
                f'Suggested value: {x_max - x_margins}.'
            )
        if self.bounding_box.y_min > y_min:
            logger.warning(
                f'The specified y_min [{self.bounding_box.y_min}] may exclude some part of the polyhedron. '
                f'Suggested value: {y_min - y_margins}.'
            )
        if self.bounding_box.y_max < y_max:
            logger.warning(
                f'The specified y_max [{self.bounding_box.y_max}] may exclude some part of the polyhedron. '
                f'Suggested value: {y_max - y_margins}.'
            )

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
            x_0 = np.linspace(self.bounding_box.x_min, self.bounding_box.x_max, 400)
            x_1 = (elem_b - row_a[0] * x_0) / row_a[1]
            mask = (x_1 >= self.bounding_box.y_min) & (x_1 <= self.bounding_box.y_max)
            x_0, x_1 = x_0[mask], x_1[mask]
        else:
            x_1 = np.linspace(self.bounding_box.y_min, self.bounding_box.y_max, 400)
            x_0 = (elem_b - row_a[1] * x_1) / row_a[0]
            mask = (x_0 >= self.bounding_box.x_min) & (x_0 <= self.bounding_box.x_max)
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
            try:
                hull = ConvexHull(all_vertices)
            except QhullError as e:
                error_msg = (
                    f'Drawing of polyhedron with vertices {the_vertices} failed: {e}'
                )
                logger.error(error_msg)
                return ax
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

    def plot_level_lines(self, ax: plt.Axes, levels: list[float], resolution=100):
        """
        Plots level lines for a given function f at specified levels.

        :param ax: matplotlib Axes object to plot on.
        :param levels: List of scalar levels to draw contours for.
        :param resolution: The resolution of the grid for computing function values.
        """
        if self.objective_coefficients is None:
            logger.warning('No objective function has been defined.')
            return

        # Generate a grid over the specified bounding box.
        x = np.linspace(self.bounding_box.x_min, self.bounding_box.x_max, resolution)
        y = np.linspace(self.bounding_box.y_min, self.bounding_box.y_max, resolution)
        X, Y = np.meshgrid(x, y)

        # Compute the function values on the grid.
        Z = self.objective_function(X, Y)  # Use the function f defined earlier.

        # Levels are expected sorted.
        levels.sort()
        # Plot the level lines.
        contour = ax.contour(X, Y, Z, levels=levels, colors='red', linestyles='--')
        ax.clabel(contour, inline=True, fontsize=8, fmt='%1.1f')

        return ax
