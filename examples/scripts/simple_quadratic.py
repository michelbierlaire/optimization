"""File simple_quadratic.py

:author: Michel Bierlaire
:date: Sun Jul  2 17:40:19 2023

Minimization of a simple quadratic function
"""

import logging
import numpy as np
from biogeme_optimization.bounds import Bounds
from biogeme_optimization.simple_bounds import simple_bounds_newton_algorithm

from test_functions import MyFunctionToMinimize, Example58, Rosenbrock

from logger import logger

logger.info('*** Simple quadratic example ***')


the_function = MyFunctionToMinimize(dimension=3)
bounds = Bounds([(-2, -1), (-3, -2), (-4, -3)])
starting_point = np.array([-1, -2.5, -3])
proportion_analytical_hessian = 0.0
first_radius = 0.1
variable_names = ['x_1', 'x_2']

solution, messages = simple_bounds_newton_algorithm(
    the_function=the_function,
    bounds=bounds,
    starting_point=starting_point,
    proportion_analytical_hessian=proportion_analytical_hessian,
    first_radius=first_radius,
    variable_names=variable_names,
)
print(f'{solution=}')
print(f'{messages=}')
