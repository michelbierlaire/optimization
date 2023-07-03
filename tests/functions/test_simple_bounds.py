"""File test_simple_bounds.py

:author: Michel Bierlaire
:date: Wed Jun 28 11:58:37 2023

Tests for the simple_bounds module
"""
import unittest
import itertools
import numpy as np
from scipy.optimize import minimize
from biogeme_optimization.bounds import Bounds
from biogeme_optimization.simple_bounds import simple_bounds_newton_algorithm

from examples import MyFunctionToMinimize, Rosenbrock


def my_function_for_scipy(x):
    the_function = MyFunctionToMinimize(dimension=3)
    the_function.set_variables(x)
    return the_function.f()


def rosenbrock_for_scipy(x):
    the_function = Rosenbrock()
    the_function.set_variables(x)
    return the_function.f()


class TestSimpleBoundsSimpleQuadratic(unittest.TestCase):
    def test_1(self):
        the_function = MyFunctionToMinimize(dimension=3)
        the_bounds = [(None, None), (None, None), (None, None)]
        bounds = Bounds(the_bounds)
        starting_point = np.array([1, 2, 3])
        values_proportion_analytical_hessian = [0.0, 0.1, 0.5, 1.0]
        values_first_radius = [0.1, 1.0, 10]
        scipy_result = minimize(
            my_function_for_scipy, x0=starting_point, bounds=the_bounds
        )
        expected_solution = scipy_result.x
        for proportion_analytical_hessian, first_radius in itertools.product(
            values_proportion_analytical_hessian, values_first_radius
        ):

            solution, messages = simple_bounds_newton_algorithm(
                the_function=the_function,
                bounds=bounds,
                starting_point=starting_point,
                proportion_analytical_hessian=proportion_analytical_hessian,
                first_radius=first_radius,
            )
            np.testing.assert_array_almost_equal(solution, expected_solution)

    def test_2(self):
        the_function = MyFunctionToMinimize(dimension=3)
        the_bounds = [(1, None), (2, None), (3, None)]
        bounds = Bounds(the_bounds)
        starting_point = np.array([10, 20, 30])
        values_proportion_analytical_hessian = [0.0, 0.1, 0.5, 1.0]
        values_first_radius = [0.1, 1.0, 10]
        scipy_result = minimize(
            my_function_for_scipy, x0=starting_point, bounds=the_bounds
        )
        expected_solution = scipy_result.x

        for proportion_analytical_hessian, first_radius in itertools.product(
            values_proportion_analytical_hessian, values_first_radius
        ):

            solution, messages = simple_bounds_newton_algorithm(
                the_function=the_function,
                bounds=bounds,
                starting_point=starting_point,
                proportion_analytical_hessian=proportion_analytical_hessian,
                first_radius=first_radius,
            )
            np.testing.assert_array_almost_equal(solution, expected_solution)

    def test_3(self):
        the_function = MyFunctionToMinimize(dimension=3)
        the_bounds = [(-1, 1), (-2, 2), (-3, 3)]
        bounds = Bounds(the_bounds)
        starting_point = np.array([-1, -2, -3])
        values_proportion_analytical_hessian = [0.0, 0.1, 0.5, 1.0]
        values_first_radius = [0.1, 1.0, 10]
        scipy_result = minimize(
            my_function_for_scipy, x0=starting_point, bounds=the_bounds
        )
        expected_solution = scipy_result.x

        for proportion_analytical_hessian, first_radius in itertools.product(
            values_proportion_analytical_hessian, values_first_radius
        ):
            solution, messages = simple_bounds_newton_algorithm(
                the_function=the_function,
                bounds=bounds,
                starting_point=starting_point,
                proportion_analytical_hessian=proportion_analytical_hessian,
                first_radius=first_radius,
            )
            np.testing.assert_array_almost_equal(solution, expected_solution)

    def test_4(self):
        the_function = MyFunctionToMinimize(dimension=3)
        the_bounds = [(1, None), (2, None), (3, None)]
        bounds = Bounds(the_bounds)
        starting_point = np.array([10, 2, 30])
        values_proportion_analytical_hessian = [0.0, 0.1, 0.5, 1.0]
        values_first_radius = [0.1, 1.0, 10]
        scipy_result = minimize(
            my_function_for_scipy, x0=starting_point, bounds=the_bounds
        )
        expected_solution = scipy_result.x

        for proportion_analytical_hessian, first_radius in itertools.product(
            values_proportion_analytical_hessian, values_first_radius
        ):

            solution, messages = simple_bounds_newton_algorithm(
                the_function=the_function,
                bounds=bounds,
                starting_point=starting_point,
                proportion_analytical_hessian=proportion_analytical_hessian,
                first_radius=first_radius,
            )
            np.testing.assert_array_almost_equal(solution, expected_solution)


class TestSimpleBoundsRosenbrock(unittest.TestCase):
    def test_1(self):
        the_function = Rosenbrock()
        the_bounds = [(None, None), (None, None)]
        bounds = Bounds(the_bounds)
        starting_point = np.array([-1.5, 1.5])
        values_proportion_analytical_hessian = [0.0, 0.1, 0.5, 1.0]
        values_first_radius = [0.1, 1.0, 10]
        scipy_result = minimize(
            rosenbrock_for_scipy, x0=starting_point, bounds=the_bounds
        )
        expected_solution = scipy_result.x
        print(f'{expected_solution=}')
        for proportion_analytical_hessian, first_radius in itertools.product(
            values_proportion_analytical_hessian, values_first_radius
        ):

            solution, messages = simple_bounds_newton_algorithm(
                the_function=the_function,
                bounds=bounds,
                starting_point=starting_point,
                proportion_analytical_hessian=proportion_analytical_hessian,
                first_radius=first_radius,
            )
            np.testing.assert_array_almost_equal(solution, expected_solution, decimal=3)

    def test_2(self):
        the_function = Rosenbrock()
        the_bounds = [(None, None), (None, 0)]
        bounds = Bounds(the_bounds)
        starting_point = np.array([-1.5, 1.5])
        values_proportion_analytical_hessian = [0.0, 0.1, 0.5, 1.0]
        values_first_radius = [0.1, 1.0, 10]
        scipy_result = minimize(
            rosenbrock_for_scipy, x0=starting_point, bounds=the_bounds
        )
        expected_solution = scipy_result.x

        for proportion_analytical_hessian, first_radius in itertools.product(
            values_proportion_analytical_hessian, values_first_radius
        ):

            solution, messages = simple_bounds_newton_algorithm(
                the_function=the_function,
                bounds=bounds,
                starting_point=starting_point,
                proportion_analytical_hessian=proportion_analytical_hessian,
                first_radius=first_radius,
            )
            np.testing.assert_array_almost_equal(solution, expected_solution, decimal=3)

    def test_3(self):
        the_function = Rosenbrock()
        the_bounds = [(1, None), (None, 0)]
        bounds = Bounds(the_bounds)
        starting_point = np.array([1, 1.5])
        values_proportion_analytical_hessian = [0.0, 0.1, 0.5, 1.0]
        values_first_radius = [0.1, 1.0, 10]
        scipy_result = minimize(
            rosenbrock_for_scipy, x0=starting_point, bounds=the_bounds
        )
        expected_solution = scipy_result.x

        for proportion_analytical_hessian, first_radius in itertools.product(
            values_proportion_analytical_hessian, values_first_radius
        ):

            solution, messages = simple_bounds_newton_algorithm(
                the_function=the_function,
                bounds=bounds,
                starting_point=starting_point,
                proportion_analytical_hessian=proportion_analytical_hessian,
                first_radius=first_radius,
            )
            np.testing.assert_array_almost_equal(solution, expected_solution, decimal=3)

    def test_4(self):
        the_function = Rosenbrock()
        the_bounds = [(-12, 1), (-1, 1)]
        bounds = Bounds(the_bounds)
        starting_point = np.array([1, 1.5])
        values_proportion_analytical_hessian = [0.0, 0.1, 0.5, 1.0]
        values_first_radius = [0.1, 1.0, 10]
        scipy_result = minimize(
            rosenbrock_for_scipy, x0=starting_point, bounds=the_bounds
        )
        expected_solution = scipy_result.x

        for proportion_analytical_hessian, first_radius in itertools.product(
            values_proportion_analytical_hessian, values_first_radius
        ):

            solution, messages = simple_bounds_newton_algorithm(
                the_function=the_function,
                bounds=bounds,
                starting_point=starting_point,
                proportion_analytical_hessian=proportion_analytical_hessian,
                first_radius=first_radius,
            )
            np.testing.assert_array_almost_equal(solution, expected_solution, decimal=3)


if __name__ == '__main__':
    unittest.main()
