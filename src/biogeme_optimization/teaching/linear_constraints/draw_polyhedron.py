"""Function to draw polyhedrons

Michel Bierlaire
Fri Mar 29 15:50:49 2024
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from matplotlib import pyplot as plt

from biogeme_optimization.teaching.linear_constraints import (
    AllConstraints,
    StandardCanonicalForms,
    Polyhedron2d,
    BoundingBox,
)


@dataclass
class LabeledPoint:
    """Labeled points for drawing"""

    coordinates: np.ndarray
    label: str | None = None
    shift: float = 0.0
    color: str = 'blue'
    marker: str = 'o'

    def from_standard_to_canonical(self, forms: StandardCanonicalForms) -> LabeledPoint:
        """Transforms the coordinates of the point from standard to canonical form."""
        transformed_coordinates = forms.from_standard_to_canonical(
            standard_x=self.coordinates
        )

        # Create a dictionary of current attributes, update the coordinates
        new_attributes = asdict(self)
        new_attributes['coordinates'] = transformed_coordinates

        # Unpack the dictionary to create a new instance
        return LabeledPoint(**new_attributes)


@dataclass
class LabeledDirection:
    """Labeled directions for drawing"""

    start: np.ndarray
    direction: np.ndarray
    length: float = 1.0
    head_fraction: float = 0.15
    label: str | None = None
    width: float = 0.01
    shift: float = 0.0
    color: str = 'blue'
    marker: str = 'o'

    @classmethod
    def from_two_points(cls, start: np.ndarray, end: np.ndarray, **kwargs):
        direction = end - start
        return cls(start, direction, **kwargs)

    @property
    def normalized_direction(self) -> np.ndarray:
        # Normalize the length of the direction.
        norm_direction = np.linalg.norm(self.direction)
        return np.array(
            [
                self.length * self.direction[0] / norm_direction,
                self.length * self.direction[1] / norm_direction,
            ]
        )

    @property
    def head_width(self) -> float:
        return self.length * self.head_fraction

    @property
    def head_length(self) -> float:
        return self.length * self.head_fraction


def draw_polyhedron(
    drawing_polyhedron: Polyhedron2d,
    level_lines: list[float] | None,
    points: list[LabeledPoint] | None,
    directions: list[LabeledDirection] | None,
) -> None:
    """
    Plot a 2D polyhedron with a list of points.
    :param drawing_polyhedron: the polyhedron.
    :param level_lines: level lines to display.
    :param points: list of points to display.
    :param directions: list of directions to display.
    """

    fig, ax = plt.subplots()
    drawing_polyhedron.plot_polyhedron(ax=ax)

    if level_lines is not None:
        drawing_polyhedron.plot_level_lines(ax=ax, levels=level_lines)

    if points is not None:
        for point in points:
            ax.scatter(
                x=point.coordinates[0],
                y=point.coordinates[1],
                s=50,
                color=point.color,
                marker=point.marker,
            )
            ax.text(
                point.coordinates[0] + point.shift,
                point.coordinates[1] + point.shift,
                point.label,
                color=point.color,
            )
    if directions is not None:
        for direction in directions:
            ax.arrow(
                direction.start[0],
                direction.start[1],
                direction.normalized_direction[0],
                direction.normalized_direction[1],
                width=direction.width,
                head_width=direction.head_width,
                head_length=direction.head_length,
                fc=direction.color,
                ec=direction.color,
            )

            # Add the label.
            if direction.label is not None:
                ax.text(
                    direction.start[0] + direction.normalized_direction[0] / 2,
                    direction.start[1] + direction.normalized_direction[1] / 2,
                    direction.label,
                    color=direction.color,
                )

    plt.show()


def draw_polyhedron_standard_form(
    matrix_a: np.ndarray,
    vector_b: np.ndarray,
    objective_coefficients: np.ndarray | None = None,
    level_lines: list[float] | None = None,
    bounding_box: BoundingBox | None = None,
    points: list[LabeledPoint] | None = None,
    directions: list[LabeledDirection] | None = None,
) -> None:
    """
    Draw a polyhedron in standard form characterized by matrix $A$ and vector $b$.
    If requested, points are also displayed.
    :param matrix_a: $A$
    :param vector_b: $b$
    :param bounding_box: limits of each coordinate.
    :param points: points to add on the figure.
    :param directions: list of directions to display.

    """
    drawing_polyhedron = AllConstraints.from_standard_form(
        matrix=matrix_a, vector=vector_b
    )
    the_forms = StandardCanonicalForms(constraints=drawing_polyhedron)
    the_points = (
        None
        if points is None
        else [point.from_standard_to_canonical(the_forms) for point in points]
    )
    drawing_polyhedron = Polyhedron2d(
        polyhedron=the_forms,
        bounding_box=bounding_box,
        objective_coefficients=objective_coefficients,
    )
    draw_polyhedron(
        drawing_polyhedron=drawing_polyhedron,
        level_lines=level_lines,
        points=the_points,
        directions=directions,
    )


def draw_polyhedron_canonical_form(
    matrix_a: np.ndarray,
    vector_b: np.ndarray,
    objective_coefficients: np.ndarray | None = None,
    level_lines: list[float] | None = None,
    bounding_box: BoundingBox | None = None,
    points: list[LabeledPoint] | None = None,
    directions: list[LabeledDirection] | None = None,
) -> None:
    """
    Draw a polyhedron in canonical form characterized by matrix $A$ and vector $b$.
    If requested, points are also displayed.
    :param matrix_a: $A$
    :param vector_b: $b$
    :param bounding_box: limits of each coordinate.
    :param points: points to add on the figure.
    :param directions: list of directions to display.

    """
    drawing_polyhedron = AllConstraints.from_canonical_form(
        matrix=matrix_a, vector=vector_b
    )
    the_forms = StandardCanonicalForms(constraints=drawing_polyhedron)
    drawing_polyhedron = Polyhedron2d(
        polyhedron=the_forms,
        bounding_box=bounding_box,
        objective_coefficients=objective_coefficients,
    )
    draw_polyhedron(
        drawing_polyhedron=drawing_polyhedron,
        level_lines=level_lines,
        points=points,
        directions=directions,
    )
