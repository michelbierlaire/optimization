"""File test_trust_region.py

:author: Michel Bierlaire
:date: Fri Jun 23 09:32:05 2023

Tests for the trust_region module
"""
import unittest
import numpy as np
import scipy.linalg as la
from biogeme_optimization.exceptions import (
    OptimizationError,
)  # Import the required exception class
from biogeme_optimization.trust_region import (
    trust_region_intersection,
    cauchy_newton_dogleg,
    dogleg,
    truncated_conjugate_gradient,
    newton_trust_region,
    bfgs_trust_region,
    DoglegDiagnostic,
    ConjugateGradientDiagnostic,
    QuadraticModel,
    NewtonModel,
    BfgsModel,
)
from examples import MyFunctionToMinimize, Example58, Rosenbrock


class TrustRegionIntersectionTest(unittest.TestCase):
    def test_intersection_1(self):
        """See example p. 295 in Bierlaire (2015)"""
        xhat = np.array([9, 1])
        xc = np.array([9, 1])
        dc = xc - xhat
        xd = np.array([7.2, -0.8])
        dd = xd - xhat
        radius = 1
        inside_direction = xc - xhat
        crossing_direction = xd - xc

        expected_intersection = np.array([8.29289, 0.29289])
        expected_lambda = la.norm(expected_intersection - xc) / la.norm(xd - xc)
        result = trust_region_intersection(inside_direction, crossing_direction, radius)

        self.assertAlmostEqual(result, expected_lambda, places=3)

    def test_intersection_2(self):
        """See example p. 295 in Bierlaire (2015)"""
        xhat = np.array([9, 1])
        xc = np.array([7.2, -0.8])
        dc = xc - xhat
        xd = np.array([4.608, 0.512])
        dd = xd - xhat
        radius = 4
        inside_direction = xc - xhat
        crossing_direction = xd - xc

        expected_intersection = np.array([5.06523, 0.28056])
        expected_lambda = la.norm(expected_intersection - xc) / la.norm(xd - xc)
        result = trust_region_intersection(inside_direction, crossing_direction, radius)

        self.assertAlmostEqual(result, expected_lambda, places=3)

    def test_intersection_3(self):
        """See example p. 295 in Bierlaire (2015)"""
        xhat = np.array([9, 1])
        xc = np.array([9, 1])
        dc = xc - xhat
        xd = np.array([0, 0])
        dd = xd - xhat
        radius = 8
        inside_direction = xc - xhat
        crossing_direction = xd - xc

        expected_intersection = np.array([1.04893, 0.11655])
        expected_lambda = la.norm(expected_intersection - xc) / la.norm(xd - xc)
        result = trust_region_intersection(inside_direction, crossing_direction, radius)

        self.assertAlmostEqual(result, expected_lambda, places=3)

    def no_test_intersection(self):
        """See example p. 295 in Bierlaire (2015)"""
        xhat = np.array([9, 1])
        xc = np.array([9, 1])
        dc = xc - xhat
        xd = np.array([0, 0])
        dd = xd - xhat
        radius = 18
        inside_direction = xc - xhat
        crossing_direction = xd - xc

        with self.assertRaises(OptimizationError):
            _ = trust_region_intersection(inside_direction, crossing_direction, radius)

    def test_intersection_xc_outside(self):
        xhat = np.array([9, 1])
        xc = np.array([7.2, -0.8])
        dc = xc - xhat
        xd = np.array([4.608, 0.512])
        dd = xd - xhat
        radius = 0.1
        inside_direction = xc - xhat
        crossing_direction = xd - xc

        with self.assertRaises(OptimizationError):
            result = trust_region_intersection(
                inside_direction, crossing_direction, radius
            )

    def test_intersection_xd_inside(self):
        xhat = np.array([9, 1])
        xc = np.array([7.2, -0.8])
        dc = xc - xhat
        xd = np.array([4.608, 0.512])
        dd = xd - xhat
        radius = 18
        inside_direction = xc - xhat
        crossing_direction = xd - xc

        with self.assertRaises(OptimizationError):
            result = trust_region_intersection(
                inside_direction, crossing_direction, radius
            )

    def test_cauchy_newton_dogleg(self):
        g = np.array([2, -1])  # Example gradient
        H = np.array([[4, -1], [-1, 2]])  # Example Hessian

        # Expected results
        d_cauchy_expected = np.array([-0.45454545, 0.22727273])
        d_newton_expected = np.array([-0.42857143, 0.28571429])
        d_dogleg_expected = np.array([-0.42662338, 0.28441558])

        # Calculate actual results
        d_cauchy, d_newton, d_dogleg = cauchy_newton_dogleg(g, H)
        normalized_cauchy = d_cauchy / np.linalg.norm(d_cauchy)
        normalized_gradient = -g / np.linalg.norm(g)
        np.testing.assert_allclose(normalized_gradient, normalized_cauchy)
        normalized_newton = d_newton / np.linalg.norm(d_newton)
        normalized_dogleg = d_dogleg / np.linalg.norm(d_dogleg)
        np.testing.assert_allclose(normalized_gradient, normalized_cauchy)
        # Compare actual and expected results
        np.testing.assert_allclose(d_cauchy, d_cauchy_expected)
        np.testing.assert_allclose(d_newton, d_newton_expected)
        np.testing.assert_allclose(d_dogleg, d_dogleg_expected)

    def test_cauchy_newton_dogleg_simple(self):
        g = np.array([1, 1])  # Example gradient
        H = np.array([[1, 0], [0, 1]])  # Example Hessian

        # Expected results
        d_cauchy_expected = np.array([-1, -1])
        d_newton_expected = np.array([-1, -1])
        d_dogleg_expected = np.array([-1, -1])

        # Calculate actual results
        d_cauchy, d_newton, d_dogleg = cauchy_newton_dogleg(g, H)
        normalized_cauchy = d_cauchy / np.linalg.norm(d_cauchy)
        normalized_gradient = -g / np.linalg.norm(g)
        np.testing.assert_allclose(normalized_gradient, normalized_cauchy)
        normalized_newton = d_newton / np.linalg.norm(d_newton)
        normalized_dogleg = d_dogleg / np.linalg.norm(d_dogleg)
        np.testing.assert_allclose(normalized_gradient, normalized_cauchy)
        # Compare actual and expected results
        np.testing.assert_allclose(d_cauchy, d_cauchy_expected)
        np.testing.assert_allclose(d_newton, d_newton_expected)
        np.testing.assert_allclose(d_dogleg, d_dogleg_expected)

    def test_cauchy_newton_dogleg_non_convex(self):
        g = np.array([1, 1])  # Example gradient
        H = np.array([[1, 0], [0, -1]])  # Example Hessian
        d_cauchy, d_newton, d_dogleg = cauchy_newton_dogleg(g, H)
        self.assertIsNone(d_newton)
        self.assertIsNone(d_dogleg)
        self.assertIsNone(d_cauchy)

    def test_cauchy_newton_dogleg_non_convex_2(self):
        g = np.array([1, 0])  # Example gradient
        H = np.array([[1, 0], [0, -1]])  # Example Hessian
        d_cauchy, d_newton, d_dogleg = cauchy_newton_dogleg(g, H)
        self.assertIsNone(d_newton)
        self.assertIsNone(d_dogleg)
        d_cauchy_expected = np.array([-1, 0])
        np.testing.assert_allclose(d_cauchy, d_cauchy_expected)

    def test_dogleg(self):
        # Test case 1: Negative curvature along Cauchy direction
        gradient = np.array([1, 0])
        hessian = np.array([[-1, 0], [0, -1]])
        radius = 1.0
        expected_solution = np.array([-1, 0])
        expected_diagnostic = DoglegDiagnostic.NEGATIVE_CURVATURE
        solution, diagnostic = dogleg(gradient, hessian, radius)
        np.testing.assert_array_almost_equal(solution, expected_solution)
        self.assertEqual(diagnostic, expected_diagnostic)

        # Test case 2: Negative curvature along Newton direction
        gradient = np.array([2, -1])
        hessian = np.array([[4, -1], [-1, -3]])
        radius = 3
        expected_solution = np.array([-0.58823529, 0.29411765])
        expected_diagnostic = DoglegDiagnostic.CAUCHY
        solution, diagnostic = dogleg(gradient, hessian, radius)
        np.testing.assert_array_almost_equal(solution, expected_solution)
        self.assertEqual(diagnostic, expected_diagnostic)

        # Test case 3: Partial Cauchy step
        xhat = np.array([9, 1])
        gradient = np.array([9, 9])
        hessian = np.array([[1, 0], [0, 9]])
        radius = 1
        xc = np.array([7.2, -0.8])
        xsol = np.array([8.29289, 0.29289])
        expected_solution = xsol - xhat
        expected_diagnostic = DoglegDiagnostic.PARTIAL_CAUCHY
        solution, diagnostic = dogleg(gradient, hessian, radius)
        np.testing.assert_array_almost_equal(solution, expected_solution, decimal=3)
        self.assertEqual(diagnostic, expected_diagnostic)

        # Test case 4: Dogleg
        xhat = np.array([9, 1])
        gradient = np.array([9, 9])
        hessian = np.array([[1, 0], [0, 9]])
        radius = 4
        xc = np.array([7.2, -0.8])
        xsol = np.array([5.06523, 0.28056])
        expected_solution = xsol - xhat
        expected_diagnostic = DoglegDiagnostic.DOGLEG
        solution, diagnostic = dogleg(gradient, hessian, radius)
        np.testing.assert_array_almost_equal(solution, expected_solution, decimal=3)
        self.assertEqual(diagnostic, expected_diagnostic)

        # Test case 5: partial Newton
        xhat = np.array([9, 1])
        gradient = np.array([9, 9])
        hessian = np.array([[1, 0], [0, 9]])
        radius = 8
        xc = np.array([7.2, -0.8])
        xsol = np.array([1.04893, 0.11655])
        expected_solution = xsol - xhat
        expected_diagnostic = DoglegDiagnostic.PARTIAL_NEWTON
        solution, diagnostic = dogleg(gradient, hessian, radius)
        np.testing.assert_array_almost_equal(solution, expected_solution, decimal=3)
        self.assertEqual(diagnostic, expected_diagnostic)

        # Test case 10: Newton
        xhat = np.array([9, 1])
        gradient = np.array([9, 9])
        hessian = np.array([[1, 0], [0, 9]])
        radius = 10
        xc = np.array([7.2, -0.8])
        xsol = np.array([0, 0])
        expected_solution = xsol - xhat
        expected_diagnostic = DoglegDiagnostic.NEWTON
        solution, diagnostic = dogleg(gradient, hessian, radius)
        np.testing.assert_array_almost_equal(solution, expected_solution, decimal=3)
        self.assertEqual(diagnostic, expected_diagnostic)

    def test_dogleg_invalid_input(self):
        # Test invalid input: gradient and hessian are not NumPy arrays
        with self.assertRaises(OptimizationError):
            dogleg([1, 2, 3], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], 1.0)

        # Test invalid input: gradient is not 1-dimensional or hessian is not 2-dimensional
        with self.assertRaises(OptimizationError):
            dogleg(np.array([1, 2, 3]), np.array([[1, 0], [0, 1], [0, 0]]), 1.0)

        # Test invalid input: gradient and hessian dimensions are incompatible
        with self.assertRaises(OptimizationError):
            dogleg(np.array([1, 2, 3]), np.array([[1, 0, 0], [0, 1, 0]]), 1.0)

    def test_conjugate_gradient_convergence(self):
        gradient = np.array([1, 2, 3])
        hessian = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        radius = 10.0
        expected_step = np.array([-1, -2, -3])
        expected_diagnostic = ConjugateGradientDiagnostic.CONVERGENCE

        step, diagnostic = truncated_conjugate_gradient(gradient, hessian, radius)
        np.testing.assert_array_almost_equal(step, expected_step)
        self.assertEqual(diagnostic, expected_diagnostic)

    def test_conjugate_gradient_out_of_trust_region(self):
        gradient = np.array([1, 2, 3])
        hessian = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        radius = 1.0
        expected_step = np.array([-0.26726124, -0.53452248, -0.80178373])
        expected_diagnostic = ConjugateGradientDiagnostic.OUT_OF_TRUST_REGION

        step, diagnostic = truncated_conjugate_gradient(gradient, hessian, radius)
        np.testing.assert_array_almost_equal(step, expected_step)
        self.assertEqual(diagnostic, expected_diagnostic)

    def test_conjugate_gradient_negative_curvature(self):
        gradient = np.array([1, 2, 3])
        hessian = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
        radius = 10.0
        expected_step = np.array([-2.56752763, -5.83763813, -7.70258288])
        expected_diagnostic = ConjugateGradientDiagnostic.NEGATIVE_CURVATURE

        step, diagnostic = truncated_conjugate_gradient(gradient, hessian, radius)
        np.testing.assert_array_almost_equal(step, expected_step)
        self.assertEqual(diagnostic, expected_diagnostic)

    def test_conjugate_gradient_numerical_problem(self):
        gradient = np.array([1, 2, 3])
        hessian = np.array([[1, 0, 0], [0, 10e12, 0], [0, 0, -10e12]])
        radius = 10.0
        expected_step = np.array([-2.67261242, -5.34522484, -8.01783726])
        expected_diagnostic = ConjugateGradientDiagnostic.NEGATIVE_CURVATURE
        step, diagnostic = truncated_conjugate_gradient(gradient, hessian, radius)
        np.testing.assert_array_almost_equal(step, expected_step)
        self.assertEqual(diagnostic, expected_diagnostic)


class TestQuadraticModel(unittest.TestCase):
    def test_f_g_h(self):
        function = MyFunctionToMinimize(dimension=2)
        with self.assertRaises(TypeError):
            _ = QuadraticModel(function)


class TestNewtonModel(unittest.TestCase):
    def test_f_g_h(self):
        function = MyFunctionToMinimize(dimension=2)
        model = NewtonModel(function)
        iterate = np.array([1, 1])
        f, g, h = model.get_f_g_h(iterate)
        expected_g = np.array([1, 1])
        np.testing.assert_array_almost_equal(g, expected_g)

        expected_h = np.array([[1.0, 0.0], [0.0, 1.0]])
        np.testing.assert_array_almost_equal(h, expected_h)

        optimal_solution = np.array([0, 0])
        f, g, h = model.get_f_g_h(optimal_solution)
        self.assertIsNone(g)
        self.assertIsNone(h)


class TestBfgsModel(unittest.TestCase):
    def test_f_g_h(self):
        function = MyFunctionToMinimize(dimension=2)
        model = BfgsModel(function)
        iterate = np.array([1, 1])
        f, g, h = model.get_f_g_h(iterate)

        expected_g = np.array([1, 1])
        np.testing.assert_array_almost_equal(g, expected_g)

        expected_h = np.array([[1.0, 0.0], [0.0, 1.0]])
        np.testing.assert_array_almost_equal(h, expected_h)

        optimal_solution = np.array([0, 0])
        f, g, h = model.get_f_g_h(optimal_solution)
        self.assertIsNone(g)
        self.assertIsNone(h)


class TestAlgorithms(unittest.TestCase):
    def test_newton_trust_region(self):
        fct = MyFunctionToMinimize(dimension=2)

        # Set the starting point
        starting_point = np.array([2.0, 2.0])

        # Set the expected solution
        expected_solution = np.array([0.0, 0.0])

        # Call the newton_linesearch function
        solution, messages = newton_trust_region(
            the_function=fct, starting_point=starting_point, use_dogleg=True
        )

        # Check the solution
        np.testing.assert_allclose(solution, expected_solution, atol=1e-6)

        # Check the termination message
        self.assertEqual(
            messages['Cause of termination'], 'Relative gradient = 0 <= 6.1e-06'
        )

        # Call the newton_linesearch function
        solution, messages = newton_trust_region(
            the_function=fct, starting_point=starting_point, use_dogleg=False
        )

        # Check the solution
        np.testing.assert_allclose(solution, expected_solution, atol=1e-6)

        # Check the termination message
        self.assertEqual(
            messages['Cause of termination'], 'Relative gradient = 0 <= 6.1e-06'
        )

    def test_bfgs_trust_region(self):
        fct = MyFunctionToMinimize(dimension=2)

        # Set the starting point
        starting_point = np.array([2.0, 2.0])

        # Set the expected solution
        expected_solution = np.array([0.0, 0.0])

        # Call the newton_linesearch function
        solution, messages = bfgs_trust_region(
            the_function=fct, starting_point=starting_point, use_dogleg=True
        )

        # Check the solution
        np.testing.assert_allclose(solution, expected_solution, atol=1e-6)

        # Check the termination message
        self.assertEqual(
            messages['Cause of termination'], 'Relative gradient = 0 <= 6.1e-06'
        )

        # Call the newton_linesearch function
        solution, messages = bfgs_trust_region(
            the_function=fct, starting_point=starting_point, use_dogleg=False
        )

        # Check the solution
        np.testing.assert_allclose(solution, expected_solution, atol=1e-6)

        # Check the termination message
        self.assertEqual(
            messages['Cause of termination'], 'Relative gradient = 0 <= 6.1e-06'
        )

    def test_newton_trust_region_ex58(self):
        fct = Example58()

        # Set the starting point
        starting_point = np.array([1.0, 1.0])

        # Set the expected solution
        expected_solution = np.array([-1.0, 0])

        # Call the newton_linesearch function
        solution, messages = newton_trust_region(
            the_function=fct, starting_point=starting_point, use_dogleg=True
        )

        # Check the solution
        np.testing.assert_allclose(solution, expected_solution, atol=1e-3)

        # Check the termination message
        self.assertEqual(
            messages['Cause of termination'], 'Relative gradient = 7.1e-07 <= 6.1e-06'
        )

        # Call the newton_linesearch function
        solution, messages = newton_trust_region(
            the_function=fct, starting_point=starting_point, use_dogleg=False
        )

        # Check the solution
        np.testing.assert_allclose(solution, expected_solution, atol=1e-6)

        # Check the termination message
        self.assertEqual(
            messages['Cause of termination'], 'Relative gradient = 1.5e-09 <= 6.1e-06'
        )

    def test_bfgs_trust_region_ex58(self):
        fct = Example58()

        # Set the starting point
        starting_point = np.array([1.0, 1.0])

        # Set the expected solution
        expected_solution = np.array([-1, 0])

        # Call the newton_linesearch function
        solution, messages = bfgs_trust_region(
            the_function=fct, starting_point=starting_point, use_dogleg=True
        )

        # Check the solution
        np.testing.assert_allclose(solution, expected_solution, atol=1e-3)

        # Check the termination message
        termination_mesage = messages['Cause of termination']
        expected_termination_message = 'Relative gradient = ...'
        self.assertEqual(termination_mesage[:19], expected_termination_message[:19])

        # Call the newton_linesearch function
        solution, messages = bfgs_trust_region(
            the_function=fct, starting_point=starting_point, use_dogleg=False
        )

        # Check the solution
        np.testing.assert_allclose(solution, expected_solution, atol=1e-3)

        # Check the termination message
        termination_mesage = messages['Cause of termination']
        expected_termination_message = 'Relative gradient = ...'
        self.assertEqual(termination_mesage[:19], expected_termination_message[:19])

    def test_newton_trust_region_rosenbrock(self):
        fct = Rosenbrock()

        # Set the starting point
        starting_point = np.array([-1.5, 1.5])

        # Set the expected solution
        expected_solution = np.array([1.0, 1.0])

        # Call the newton_linesearch function
        solution, messages = newton_trust_region(
            the_function=fct, starting_point=starting_point, use_dogleg=True
        )

        # Check the solution
        np.testing.assert_allclose(solution, expected_solution, atol=1e-3)

        # Check the termination message
        self.assertEqual(
            messages['Cause of termination'], 'Relative gradient = 2.6e-06 <= 6.1e-06'
        )

        # Call the newton_linesearch function
        solution, messages = newton_trust_region(
            the_function=fct, starting_point=starting_point, use_dogleg=False
        )

        # Check the solution
        np.testing.assert_allclose(solution, expected_solution, atol=1e-6)

        # Check the termination message
        self.assertEqual(
            messages['Cause of termination'], 'Relative gradient = 9.8e-10 <= 6.1e-06'
        )

    def test_bfgs_trust_region_rosenbrock(self):
        fct = Rosenbrock()

        # Set the starting point
        starting_point = np.array([-1.5, 1.5])

        # Set the expected solution
        expected_solution = np.array([1.0, 1.0])

        # Call the newton_linesearch function
        solution, messages = bfgs_trust_region(
            the_function=fct,
            starting_point=starting_point,
            use_dogleg=True,
            maxiter=20000,
        )

        # Check the solution
        np.testing.assert_allclose(solution, expected_solution, atol=1e-3)

        # Check the termination message
        termination_mesage = messages['Cause of termination']
        expected_termination_message = 'Relative gradient = ...'
        self.assertEqual(termination_mesage[:19], expected_termination_message[:19])

        # Call the newton_linesearch function
        solution, messages = bfgs_trust_region(
            the_function=fct,
            starting_point=starting_point,
            use_dogleg=False,
            maxiter=20000,
        )

        # Check the solution
        np.testing.assert_allclose(solution, expected_solution, atol=1e-3)

        # Check the termination message
        termination_mesage = messages['Cause of termination']
        expected_termination_message = 'Relative gradient = ...'
        self.assertEqual(termination_mesage[:19], expected_termination_message[:19])


if __name__ == '__main__':
    unittest.main()
