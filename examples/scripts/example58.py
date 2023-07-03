"""File example58.py

:author: Michel Bierlaire
:date: Mon Jul  3 08:52:57 2023

Minimization of the function from Example 5.8 in the book
"""

import logging
import numpy as np
from biogeme_optimization.bounds import Bounds
from biogeme_optimization.simple_bounds import simple_bounds_newton_algorithm

from test_functions import Example58
from logger import logger

logger.info('***** Example 5.8 from Bierlaire (2015) *****')


the_function = Example58()
bounds = Bounds([(None, None), (None, None)])
starting_point = np.array([-1.5, 1.5])

variable_names = ['x_1', 'x_2']
print('=====  With analytical hessian =====')
solution, messages = simple_bounds_newton_algorithm(
    the_function=the_function,
    bounds=bounds,
    starting_point=starting_point,
    proportion_analytical_hessian=1.0,
    first_radius=1,
    variable_names=variable_names,
)
print(f'{solution=}')
print(f'{messages=}')

print('=====  With BFGS =====')

solution, messages = simple_bounds_newton_algorithm(
    the_function=the_function,
    bounds=bounds,
    starting_point=starting_point,
    proportion_analytical_hessian=0.0,
    first_radius=1,
)
print(f'{solution=}')
print(f'{messages=}')

print('=====  With hybrid =====')

solution, messages = simple_bounds_newton_algorithm(
    the_function=the_function,
    bounds=bounds,
    starting_point=starting_point,
    proportion_analytical_hessian=0.5,
    first_radius=1,
    variable_names=variable_names,
)
print(f'{solution=}')
print(f'{messages=}')

print('=====  Max iterations =====')

solution, messages = simple_bounds_newton_algorithm(
    the_function=the_function,
    bounds=bounds,
    starting_point=starting_point,
    proportion_analytical_hessian=0.5,
    first_radius=1,
    maxiter=2,
    variable_names=variable_names,
)

print(f'{solution=}')
print(f'{messages=}')
