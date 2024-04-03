import unittest
import numpy as np

from biogeme_optimization.teaching.linear_constraints import (
    AllConstraints,
    StandardCanonicalForms,
)
from biogeme_optimization.teaching.linear_constraints.polyhedron_2d import (
    Polyhedron2d,
    BoundingBox,
)


# from biogeme_optimization.teaching.linear_constraints import StandardCanonicalForms


class TestPolyhedron2d(unittest.TestCase):

    def setUp(self):
        """Setup test fixtures for the polyhedron tests."""
        canonical_a = np.array([[-1, 1], [1, 1], [-1, 0], [0, -1]])
        canonical_b = np.array([2, 4, 0, 0])

        # Create a polyhedron object for plotting
        polyhedron = AllConstraints.from_canonical_form(
            matrix=canonical_a, vector=canonical_b
        )

        # Create an object managing the standard and canonical forms.
        self.polyhedron_forms = StandardCanonicalForms(constraints=polyhedron)
        # non-2D polyhedron (3D in this case).
        canonical_a_3d = np.array([[-1, 1, 1], [1, 1, -1]])
        canonical_b_3d = np.array([2, 4])
        polyhedron_3d = AllConstraints.from_canonical_form(
            matrix=canonical_a_3d, vector=canonical_b_3d
        )
        self.polyhedron_forms_3d = StandardCanonicalForms(constraints=polyhedron_3d)

    def test_non_two_dimensional_input(self):
        """Test initialization rejects non-2D polyhedrons."""
        bounding_box = BoundingBox(x_min=0, x_max=1, y_min=0, y_max=1)
        with self.assertRaises(ValueError):
            Polyhedron2d(polyhedron=self.polyhedron_forms_3d, bounding_box=bounding_box)

    def test_two_dimensional_input(self):
        """Test initialization accepts valid 2D polyhedrons."""
        bounding_box = BoundingBox(x_min=0, x_max=1, y_min=0, y_max=1)

        try:
            Polyhedron2d(polyhedron=self.polyhedron_forms, bounding_box=bounding_box)
        except ValueError:
            self.fail("Initialization failed for a 2-dimensional polyhedron.")

    def test_vertices_calculation(self):
        """Test vertices are correctly calculated."""
        bounding_box = BoundingBox(x_min=-1, x_max=1, y_min=-1, y_max=1)

        polyhedron_2d = Polyhedron2d(
            polyhedron=self.polyhedron_forms, bounding_box=bounding_box
        )
        expected_vertices = {(0.0, 2.0), (4.0, 0.0), (1.0, 3.0), (0.0, 0.0)}
        self.assertEqual(polyhedron_2d.vertices(), expected_vertices)


if __name__ == '__main__':
    unittest.main()
