"""File test_bounds.py

:author: Michel Bierlaire
:date: Wed Jun 21 10:06:46 2023

Tests for the bounds module
"""
import unittest
import numpy as np
from biogeme_optimization.exceptions import (
    OptimizationError,
)  # Import the required exception class
from biogeme_optimization.bounds import (
    Bounds,
    inner_product_subspace,
    matrix_vector_mult_subspace,
)
from biogeme_optimization.diagnostics import ConjugateGradientDiagnostic


class TestCode(unittest.TestCase):
    def setUp(self):
        the_bounds = [(1, 10), (2, 20), (3, 30)]
        self.bounds = Bounds(the_bounds)

    def test_constructor(self):
        same_bounds = Bounds.from_bounds(
            self.bounds.lower_bounds, self.bounds.upper_bounds
        )
        np.testing.assert_array_equal(
            self.bounds.lower_bounds, same_bounds.lower_bounds
        )
        np.testing.assert_array_equal(
            self.bounds.upper_bounds, same_bounds.upper_bounds
        )

    def test_inner_product_subspace(self):
        a = np.array([1, 2, 3, 4, 5])
        b = np.array([2, 4, 6, 8, 10])
        subspace_indices = [1, 3, 4]

        result = inner_product_subspace(a, b, subspace_indices)
        expected = 2 * 4 + 4 * 8 + 5 * 10

        self.assertEqual(result, expected)

    def test_inner_product_subspace_empty_indices(self):
        a = np.array([1, 2, 3, 4, 5])
        b = np.array([2, 4, 6, 8, 10])
        subspace_indices = []

        result = inner_product_subspace(a, b, subspace_indices)
        expected = 0  # Empty selection, inner product is 0

        self.assertEqual(result, expected)

    def test_inner_product_subspace_invalid_indices(self):
        a = np.array([1, 2, 3, 4, 5])
        b = np.array([2, 4, 6, 8, 10])
        subspace_indices = [1, 5]  # Out of bounds index

        with self.assertRaises(IndexError):
            inner_product_subspace(a, b, subspace_indices)

    def test_matrix_vector_mult_subspace(self):
        # Test case 1
        matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        vector = np.array([1, 2, 3])
        subspace_indices = {0, 2}
        expected_result = np.array([1 * 1 + 3 * 3, 4 * 1 + 6 * 3, 7 * 1 + 9 * 3])
        result = matrix_vector_mult_subspace(matrix, vector, subspace_indices)
        np.testing.assert_array_equal(result, expected_result)

        # Test case 2
        matrix = np.array([[2, 4, 6], [1, 3, 5], [0, 7, 9]])
        vector = np.array([-1, 2, 0])
        subspace_indices = {1, 2}
        expected_result = np.array([4 * 2 + 6 * 0, 3 * 2 + 5 * 0, 7 * 2 + 9 * 0])
        result = matrix_vector_mult_subspace(matrix, vector, subspace_indices)
        np.testing.assert_array_equal(result, expected_result)

        # Test case 3 (empty subspace_indices)
        matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        vector = np.array([1, 2, 3])
        subspace_indices = set()
        expected_result = np.array(
            [
                1 * 1 + 2 * 2 + 3 * 3,
                4 * 1 + 5 * 2 + 6 * 3,
                7 * 1 + 8 * 2 + 9 * 3,
            ]
        )
        result = matrix_vector_mult_subspace(matrix, vector, subspace_indices)
        np.testing.assert_array_equal(result, expected_result)

    def test_invalid_dimensions(self):
        # Test case with incompatible dimensions
        matrix = np.array([[1, 2], [4, 5]])
        vector = np.array([1, 2, 3])
        subspace_indices = [0, 2]
        with self.assertRaises(ValueError):
            matrix_vector_mult_subspace(matrix, vector, subspace_indices)

    def test_invalid_indices(self):
        # Test case with out-of-range indices
        matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        vector = np.array([1, 2, 3])
        subspace_indices = [0, 3]  # Index 3 is out of range
        with self.assertRaises(IndexError):
            matrix_vector_mult_subspace(matrix, vector, subspace_indices)

    def test_project_infinite(self):
        the_bounds = Bounds([(0, 1), (0, None), (None, 1), (None, None)])
        point_1 = np.array([-1, -1, -1, -1])
        output_1 = the_bounds.project(point_1)
        expected_1 = np.array([0, 0, -1, -1])
        np.testing.assert_array_equal(output_1, expected_1)

        point_2 = np.array([2, 2, 2, 2])
        output_2 = the_bounds.project(point_2)
        expected_2 = np.array([1, 2, 1, 2])
        np.testing.assert_array_equal(output_2, expected_2)

    def test_project(self):
        example_input = [5, 15, 25]
        expected_output = np.array(
            [
                5,
                15,
                25,
            ]
        )  # Expected output when all values are within bounds
        np.testing.assert_array_equal(
            self.bounds.project(example_input), expected_output
        )

    def test_lower_out_of_bounds(self):
        input_with_lower_out_of_bounds = [0, 15, 25]
        expected_output_with_lower_out_of_bounds = np.array([1, 15, 25])
        np.testing.assert_array_almost_equal(
            self.bounds.project(input_with_lower_out_of_bounds),
            expected_output_with_lower_out_of_bounds,
        )

    def test_upper_out_of_bounds(self):
        input_with_upper_out_of_bounds = [5, 25, 40]
        expected_output_with_upper_out_of_bounds = np.array([5, 20, 30])
        np.testing.assert_array_almost_equal(
            self.bounds.project(input_with_upper_out_of_bounds),
            expected_output_with_upper_out_of_bounds,
        )

    def test_both_out_of_bounds(self):
        input_with_both_bounds_out_of_bounds = [0, 25, 40]
        expected_output_with_both_bounds_out_of_bounds = np.array([1, 20, 30])
        np.testing.assert_array_almost_equal(
            self.bounds.project(input_with_both_bounds_out_of_bounds),
            expected_output_with_both_bounds_out_of_bounds,
        )

    def test_incompatible(self):
        input_with_incompatible_size = [1, 2, 3, 4]
        with self.assertRaises(OptimizationError):
            self.bounds.project(input_with_incompatible_size)

    def test_intersect(self):
        """
        Test the intersect method of the Bounds class.
        """
        # Create Bounds objects for testing
        bounds1 = Bounds([(1, 5), (2, 6), (3, 7)])
        bounds2 = Bounds([(3, 8), (4, 9), (5, 10)])

        # Test intersect with compatible dimensions
        expected_output = Bounds([(3, 5), (4, 6), (5, 7)])
        self.assertEqual(bounds1.intersect(bounds2).bounds, expected_output.bounds)

        # Test intersect with incompatible dimensions
        bounds3 = Bounds([(1, 5), (2, 6), (3, 7), (4, 8)])
        with self.assertRaises(OptimizationError):
            bounds1.intersect(bounds3)

    def test_intersection_with_trust_region(self):
        """
        Test the intersection_with_trust_region method of the Bounds class.
        """
        # Create Bounds object for testing
        bounds = Bounds([(0, 5), (-2, 2), (1, 10), (-6, -5)])

        # Test intersection with compatible dimensions
        center = np.array([2.5, 1.5, 1, -5.5])
        delta = 1.5
        expected_output = Bounds([(1, 4), (0, 2), (1, 2.5), (-6, -5)])
        self.assertEqual(
            bounds.intersection_with_trust_region(center, delta).bounds,
            expected_output.bounds,
        )

        # Test intersection with incompatible dimensions
        center = np.array([2, 0])
        delta = 1.5
        with self.assertRaises(OptimizationError):
            bounds.intersection_with_trust_region(center, delta)

    def test_bounds_for_trust_region_subproblem(self):
        """ """
        # Create Bounds object for testing
        bounds = Bounds([(0, 5), (-2, 2), (1, 10), (-6, -5)])

        # Test intersection with compatible dimensions
        center = np.array([2.5, 1.5, 1, -5.5])
        delta = 1.5
        expected_output = Bounds([(-1.5, 1.5), (-1.5, 0.5), (0, 1.5), (-0.5, 0.5)])
        self.assertEqual(
            bounds.get_bounds_for_trust_region_subproblem(center, delta).bounds,
            expected_output.bounds,
        )

        # Test intersection with incompatible dimensions
        center = np.array([2, 0])
        delta = 1.5
        with self.assertRaises(OptimizationError):
            bounds.intersection_with_trust_region(center, delta)

    def test_subspace(self):
        """
        Test the subspace method of the Bounds class.
        """
        # Create Bounds object for testing
        bounds = Bounds([(0, 5), (-2, 2), (1, 10)])

        # Test subspace with selected variables
        selected_variables = [True, False, True]
        expected_output = Bounds([(0, 5), (1, 10)])
        self.assertEqual(
            bounds.subspace(selected_variables).bounds, expected_output.bounds
        )

        # Test subspace with incompatible dimensions
        selected_variables = [True, False]
        with self.assertRaises(OptimizationError):
            bounds.subspace(selected_variables)

    def test_feasible(self):
        """
        Test the feasible method of the Bounds class.
        """
        # Create Bounds object for testing
        bounds = Bounds([(0, 5), (-2, 2), (1, 10)])

        # Test feasible with a feasible point
        point = np.array([2, 0, 5])
        self.assertTrue(bounds.feasible(point))

        # Test feasible with a point violating the lower bound
        point = np.array([-1, 0, 5])
        self.assertFalse(bounds.feasible(point))

        # Test feasible with a point violating the upper bound
        point = np.array([2, 0, 11])
        self.assertFalse(bounds.feasible(point))

        # Test feasible with a point violating both lower and upper bounds
        point = np.array([-1, 0, 11])
        self.assertFalse(bounds.feasible(point))

        # Test feasible with incompatible dimensions
        point = np.array([2, 0])
        with self.assertRaises(OptimizationError):
            bounds.feasible(point)

    def test_maximum_step_feasible_point(self):
        bounds = Bounds([(1, 2), (1, 2)])
        x = np.array([1.5, 1.5])
        d = np.array([1, 1])
        expected_alpha = 0.5
        expected_indices = np.array([0, 1])

        alpha, indices = bounds.maximum_step(x, d)

        self.assertAlmostEqual(alpha, expected_alpha)
        np.testing.assert_array_equal(indices, expected_indices)

    def test_maximum_step_infeasible_point(self):
        bounds = Bounds([(1, 2), (1, 2)])
        x = np.array([0.5, 0.5])
        d = np.array([1, 1])

        with self.assertRaises(OptimizationError):
            bounds.maximum_step(x, d)

    def test_maximum_step_positive_direction(self):
        bounds = Bounds([(1, 2), (1, 2)])
        x = np.array([1.5, 1.5])
        d = np.array([1, 1])
        expected_alpha = 0.5
        expected_indices = np.array([0, 1])

        alpha, indices = bounds.maximum_step(x, d)

        self.assertAlmostEqual(alpha, expected_alpha)
        np.testing.assert_array_equal(indices, expected_indices)

    def test_maximum_step_negative_direction(self):
        bounds = Bounds([(1, 2), (1, 2)])
        x = np.array([1.5, 1.5])
        d = np.array([-1, -1])
        expected_alpha = 0.5
        expected_indices = np.array([0, 1])

        alpha, indices = bounds.maximum_step(x, d)

        self.assertAlmostEqual(alpha, expected_alpha)
        np.testing.assert_array_equal(indices, expected_indices)

    def test_maximum_step_mixed_direction(self):
        bounds = Bounds([(1, 2), (1, 2)])
        x = np.array([1.5, 1.5])
        d = np.array([1, -1])
        expected_alpha = 0.5
        expected_indices = np.array([0, 1])

        alpha, indices = bounds.maximum_step(x, d)

        self.assertAlmostEqual(alpha, expected_alpha)
        np.testing.assert_array_equal(indices, expected_indices)

    def test_activity_inactive_bounds(self):
        # Test when no bounds are active
        bounds = Bounds([(-1, 3), (0, 1), (2, 4)])
        point = np.array([0, 0.5, 3])
        activity = bounds.activity(point)
        expected_activity = np.array([0, 0, 0])
        np.testing.assert_array_equal(activity, expected_activity)

    def test_activity_lower_bound_active(self):
        # Test when lower bounds are active
        bounds = Bounds([(-1, 3), (0, 1), (2, 4)])
        point = np.array([-1, 0.5, 2])
        activity = bounds.activity(point)
        expected_activity = np.array([-1, 0, -1])
        np.testing.assert_array_equal(activity, expected_activity)

    def test_activity_upper_bound_active(self):
        # Test when upper bounds are active
        bounds = Bounds([(-1, 3), (0, 1), (2, 4)])
        point = np.array([2, 0.5, 4])
        activity = bounds.activity(point)
        expected_activity = np.array([0, 0, 1])
        np.testing.assert_array_equal(activity, expected_activity)

    def test_activity_both_bounds_active(self):
        # Test when both lower and upper bounds are active
        bounds = Bounds([(-1, 3), (0, 1), (2, 4)])
        point = np.array([-1, 0, 4])
        activity = bounds.activity(point)
        expected_activity = np.array([-1, -1, 1])
        np.testing.assert_array_equal(activity, expected_activity)

    def test_activity_feasible_point(self):
        # Test with a feasible point
        bounds = Bounds([(-1, 3), (0, 1), (2, 4)])
        point = np.array([1, 0.5, 3])
        activity = bounds.activity(point)
        expected_activity = np.array([0, 0, 0])
        np.testing.assert_array_equal(activity, expected_activity)

    def test_activity_invalid_point_size(self):
        # Test with an invalid point size
        bounds = Bounds([(-1, 3), (0, 1), (2, 4)])
        point = np.array([1, 2])  # Size does not match the dimension of bounds
        with self.assertRaises(OptimizationError):
            bounds.activity(point)

    def test_activity_infeasible_point(self):
        # Test with an infeasible point
        bounds = Bounds([(-1, 3), (0, 1), (2, 4)])
        point = np.array([4, 0.5, 3])
        with self.assertRaises(OptimizationError):
            bounds.activity(point)

    def test_active_constraints_all_active(self):
        # Test case: All variables are inactive
        bounds = Bounds([(0, 1), (2, 3), (-1, 1)])
        point = np.array([0.5, 2.5, 0])
        expected_status = set()
        active_constraints = bounds.active_constraints(point)
        self.assertSetEqual(active_constraints, expected_status)

    def test_active_constraints_lower_bound_active(self):
        # Test case: Variable 1 is lower bound active
        bounds = Bounds([(0, 1), (2, 3), (-1, 1)])
        point = np.array([0, 2.5, 0])
        expected_status = {0}
        active_constraints = bounds.active_constraints(point)
        self.assertSetEqual(active_constraints, expected_status)

    def test_active_constraints_upper_bound_active(self):
        # Test case: Variable 2 is upper bound active
        bounds = Bounds([(0, 1), (2, 3), (-1, 1)])
        point = np.array([0.5, 3, 0])
        expected_status = {1}
        active_constraints = bounds.active_constraints(point)
        self.assertSetEqual(active_constraints, expected_status)

    def test_active_constraints_both_bounds_active(self):
        # Test case: Variable 1 is lower bound active and Variable 3 is upper bound active
        bounds = Bounds([(0, 1), (2, 3), (-1, 1)])
        point = np.array([0, 2.5, 1])
        expected_status = {0, 2}
        active_constraints = bounds.active_constraints(point)
        self.assertSetEqual(active_constraints, expected_status)

    def test_active_constraints_no_active(self):
        # Test case: All variables are active
        bounds = Bounds([(0, 1), (2, 3), (-1, 1)])
        point = np.array([0, 3, -1])
        expected_status = {0, 1, 2}
        active_constraints = bounds.active_constraints(point)
        self.assertSetEqual(active_constraints, expected_status)

    def test_inactive_constraints_all_inactive(self):
        # Test case: All variables are inactive
        bounds = Bounds([(0, 1), (2, 3), (-1, 1)])
        point = np.array([0.5, 2.5, 0])
        expected_status = {0, 1, 2}
        active_constraints = bounds.inactive_constraints(point)
        self.assertSetEqual(active_constraints, expected_status)

    def test_inactive_constraints_lower_bound_inactive(self):
        # Test case: Variable 1 is lower bound active
        bounds = Bounds([(0, 1), (2, 3), (-1, 1)])
        point = np.array([0, 2.5, 0])
        expected_status = {1, 2}
        active_constraints = bounds.inactive_constraints(point)
        self.assertSetEqual(active_constraints, expected_status)

    def test_inactive_constraints_upper_bound_inactive(self):
        # Test case: Variable 2 is upper bound active
        bounds = Bounds([(0, 1), (2, 3), (-1, 1)])
        point = np.array([0.5, 3, 0])
        expected_status = {0, 2}
        active_constraints = bounds.inactive_constraints(point)
        self.assertSetEqual(active_constraints, expected_status)

    def test_inactive_constraints_both_bounds_inactive(self):
        # Test case: Variable 1 is lower bound active and Variable 3 is upper bound active
        bounds = Bounds([(0, 1), (2, 3), (-1, 1)])
        point = np.array([0, 2.5, 1])
        expected_status = {1}
        active_constraints = bounds.inactive_constraints(point)
        self.assertSetEqual(active_constraints, expected_status)

    def test_inactive_constraints_no_inactive(self):
        # Test case: All variables are active
        bounds = Bounds([(0, 1), (2, 3), (-1, 1)])
        point = np.array([0, 3, -1])
        expected_status = set()
        active_constraints = bounds.inactive_constraints(point)
        self.assertSetEqual(active_constraints, expected_status)

    def test_breakpoints(self):
        # Test with a feasible point and direction
        bounds_list = [(0, 1), (-1, 2), (1, 3)]
        bounds = Bounds(bounds_list)
        point = np.array([0.5, 0, 2])
        direction = np.array([1, 0.5, -1])
        result = bounds.breakpoints(point, direction)
        expected_result = [(0, 0.5), (2, 1), (1, 4)]
        self.assertEqual(result, expected_result)

    def test_breakpoints_invalid_direction(self):
        # Test with an invalid direction size
        bounds_list = [(0, 1), (-1, 2), (1, 3)]
        bounds = Bounds(bounds_list)
        point = np.array([0, 0, 0])
        direction = np.array([1, 0.5])  # Size does not match the dimension of bounds
        with self.assertRaises(OptimizationError):
            bounds.breakpoints(point, direction)

    def test_breakpoints_infeasible_point(self):
        # Test with an infeasible point
        bounds_list = [(0, 1), (-1, 2), (1, 3)]
        bounds = Bounds(bounds_list)
        point = np.array([0, 2, 4])
        direction = np.array([1, 0.5, -1])
        with self.assertRaises(OptimizationError):
            bounds.breakpoints(point, direction)

    def test_projected_direction(self):
        # Define the bounds for the variables
        bounds = Bounds([(-1, 1), (-1, 1), (-1, 1)])

        # Test case 1: x_current is inactive
        x_current = np.array([0.0, 0.0, 0.0])
        direction = np.array([-1.0, 2.0, 0.0])
        expected_direction = direction
        projected_direction = bounds.projected_direction(x_current, direction)

        np.testing.assert_array_almost_equal(projected_direction, expected_direction)

        # Test case 2: x_current is active
        x_current = np.array([-1.0, 0.0, 0.0])
        direction = np.array([-0.5, -1.0, 0.0])
        expected_direction = direction
        expected_direction[0] = 0
        projected_direction = bounds.projected_direction(x_current, direction)

        np.testing.assert_array_almost_equal(projected_direction, expected_direction)

        # Test case 3: x_current is partially active
        x_current = np.array([1.0, 1.0, 0.0])
        g_current = np.array([-1.0, 1.0, 5.0])
        expected_direction = direction
        expected_direction[1] = 0

        projected_direction = bounds.projected_direction(x_current, direction)

        np.testing.assert_array_almost_equal(projected_direction, expected_direction)


import unittest
import numpy as np


class TestBounds(unittest.TestCase):
    def test_generalized_cauchy_point_convergence(self):
        # Test case inputs
        bounds = Bounds([(-100, 100), (-100, 100)])

        iterate = np.array([0, 0])
        gradient = np.array([1, 1])
        hessian = np.array([[1, 0], [0, 1]])

        # Call the generalized_cauchy_point function
        result = bounds.generalized_cauchy_point(iterate, gradient, hessian)

        # Define the expected result
        expected_result = np.array([-1, -1])

        # Perform the assertion
        np.testing.assert_array_almost_equal(result, expected_result)

    def test_generalized_cauchy_point_active_bounds(self):
        # Test case inputs
        bounds = Bounds([(-0.5, 100), (-100, 100)])

        iterate = np.array([0, 0])
        gradient = np.array([1, 1])
        hessian = np.array([[1, 0], [0, 1]])

        # Call the generalized_cauchy_point function
        result = bounds.generalized_cauchy_point(iterate, gradient, hessian)

        # Define the expected result
        expected_result = np.array([-0.5, -1])

        # Perform the assertion
        np.testing.assert_array_almost_equal(result, expected_result)

    def test_generalized_cauchy_point_all_active_bounds(self):
        # Test case inputs
        bounds = Bounds([(-0.5, 100), (-0.4, 100)])

        iterate = np.array([0, 0])
        gradient = np.array([1, 1])
        hessian = np.array([[1, 0], [0, 1]])

        # Call the generalized_cauchy_point function
        result = bounds.generalized_cauchy_point(iterate, gradient, hessian)

        # Define the expected result
        expected_result = np.array([-0.5, -0.4])

        # Perform the assertion
        np.testing.assert_array_almost_equal(result, expected_result)

    def test_conjugate_gradient_convergence(self):

        bounds = Bounds([(None, None), (None, None)])

        gradient = np.array([1, 1])
        hessian = np.array([[1, 0], [0, 1]])
        expected_solution = np.array([-1, -1])
        expected_diagnostic = ConjugateGradientDiagnostic.CONVERGENCE
        solution, diagnostic = bounds.truncated_conjugate_gradient(
            gradient=gradient,
            hessian=hessian,
        )
        np.testing.assert_array_almost_equal(solution, expected_solution)
        self.assertEqual(diagnostic, expected_diagnostic)

        gradient = np.array([1, 1])
        hessian = np.array([[1, 0], [0, 10]])
        expected_solution = np.array([-1, -0.1])
        expected_diagnostic = ConjugateGradientDiagnostic.CONVERGENCE
        solution, diagnostic = bounds.truncated_conjugate_gradient(
            gradient=gradient,
            hessian=hessian,
        )
        np.testing.assert_array_almost_equal(solution, expected_solution)
        self.assertEqual(diagnostic, expected_diagnostic)

    def test_conjugate_gradient_convergence_negative_curvature(self):

        bounds = Bounds([(-100, 100), (-100, 100)])

        gradient = np.array([1, 1])
        hessian = np.array([[1, 0], [0, -1]])
        expected_solution = np.array([-100, -100])
        expected_diagnostic = ConjugateGradientDiagnostic.NEGATIVE_CURVATURE
        solution, diagnostic = bounds.truncated_conjugate_gradient(
            gradient=gradient,
            hessian=hessian,
        )
        np.testing.assert_array_almost_equal(solution, expected_solution)
        self.assertEqual(diagnostic, expected_diagnostic)

    def test_conjugate_gradient_convergence_out_of_trust_region(self):

        bounds = Bounds([(None, None), (-0.05, None)])

        gradient = np.array([1, 1])
        hessian = np.array([[1, 0], [0, 1]])
        expected_solution = np.array([-0.05, -0.05])
        expected_diagnostic = ConjugateGradientDiagnostic.OUT_OF_TRUST_REGION
        solution, diagnostic = bounds.truncated_conjugate_gradient(
            gradient=gradient,
            hessian=hessian,
        )
        np.testing.assert_array_almost_equal(solution, expected_solution)
        self.assertEqual(diagnostic, expected_diagnostic)

        gradient = np.array([1, 1])
        hessian = np.array([[1, 0], [0, 10]])
        expected_solution = np.array([-0.05, -0.05])
        expected_diagnostic = ConjugateGradientDiagnostic.OUT_OF_TRUST_REGION
        solution, diagnostic = bounds.truncated_conjugate_gradient(
            gradient=gradient,
            hessian=hessian,
        )
        np.testing.assert_array_almost_equal(solution, expected_solution)
        self.assertEqual(diagnostic, expected_diagnostic)

    def test_conjugate_gradient_subspace_convergence(self):

        bounds = Bounds([(None, None), (None, None)])

        iterate = np.array([0, 0])
        gradient = np.array([1, 1])
        hessian = np.array([[1, 0], [0, 1]])
        radius = 100
        expected_solution = np.array([-1, -1])
        expected_diagnostic = ConjugateGradientDiagnostic.CONVERGENCE
        solution, diagnostic = bounds.truncated_conjugate_gradient_subspace(
            iterate=iterate,
            gradient=gradient,
            hessian=hessian,
            radius=radius,
        )
        np.testing.assert_array_almost_equal(solution, expected_solution)
        self.assertEqual(diagnostic, expected_diagnostic)

        iterate = np.array([1, 1])
        gradient = np.array([1, 1])
        hessian = np.array([[1, 0], [0, 1]])
        radius = 100
        expected_solution = np.array([0, 0])
        expected_diagnostic = ConjugateGradientDiagnostic.CONVERGENCE
        solution, diagnostic = bounds.truncated_conjugate_gradient_subspace(
            iterate=iterate,
            gradient=gradient,
            hessian=hessian,
            radius=radius,
        )
        np.testing.assert_array_almost_equal(solution, expected_solution)
        self.assertEqual(diagnostic, expected_diagnostic)

    def test_conjugate_gradient_subspace_convergence(self):

        bounds = Bounds([(None, None), (None, None)])

        iterate = np.array([0, 0])
        gradient = np.array([1, 1])
        hessian = np.array([[1, 0], [0, 1]])
        radius = 100
        expected_solution = np.array([-1, -1])
        expected_diagnostic = ConjugateGradientDiagnostic.CONVERGENCE
        solution, diagnostic = bounds.truncated_conjugate_gradient_subspace(
            iterate=iterate,
            gradient=gradient,
            hessian=hessian,
            radius=radius,
        )
        np.testing.assert_array_almost_equal(solution, expected_solution)
        self.assertEqual(diagnostic, expected_diagnostic)

        iterate = np.array([1, 1])
        gradient = np.array([1, 1])
        hessian = np.array([[1, 0], [0, 1]])
        radius = 100
        expected_solution = np.array([0, 0])
        expected_diagnostic = ConjugateGradientDiagnostic.CONVERGENCE
        solution, diagnostic = bounds.truncated_conjugate_gradient_subspace(
            iterate=iterate,
            gradient=gradient,
            hessian=hessian,
            radius=radius,
        )
        np.testing.assert_array_almost_equal(solution, expected_solution)
        self.assertEqual(diagnostic, expected_diagnostic)

    def test_conjugate_gradient_subspace_active(self):

        bounds = Bounds([(-0.5, None), (None, None)])

        iterate = np.array([0, 0])
        gradient = np.array([1, 1])
        hessian = np.array([[1, 0], [0, 1]])
        radius = 100
        expected_solution = np.array([-0.5, -1])
        expected_diagnostic = ConjugateGradientDiagnostic.CONVERGENCE
        solution, diagnostic = bounds.truncated_conjugate_gradient_subspace(
            iterate=iterate,
            gradient=gradient,
            hessian=hessian,
            radius=radius,
        )
        np.testing.assert_array_almost_equal(solution, expected_solution)
        self.assertEqual(diagnostic, expected_diagnostic)

        bounds = Bounds([(0.5, None), (None, None)])

        iterate = np.array([1, 1])
        gradient = np.array([1, 1])
        hessian = np.array([[1, 0], [0, 1]])
        radius = 100
        expected_solution = np.array([0.5, 0])
        expected_diagnostic = ConjugateGradientDiagnostic.CONVERGENCE
        solution, diagnostic = bounds.truncated_conjugate_gradient_subspace(
            iterate=iterate,
            gradient=gradient,
            hessian=hessian,
            radius=radius,
        )
        np.testing.assert_array_almost_equal(solution, expected_solution)
        self.assertEqual(diagnostic, expected_diagnostic)


if __name__ == '__main__':
    unittest.main()
