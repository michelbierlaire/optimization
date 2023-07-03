"""File rosenbrock.py

:author: Michel Bierlaire
:date: Sun Jul  2 18:10:06 2023

Minimization of the Rosenbrock function
"""

import logging
import numpy as np
from biogeme_optimization.bounds import Bounds
from biogeme_optimization.simple_bounds import simple_bounds_newton_algorithm

from test_functions import Rosenbrock
from logger import logger

logger.info('***** Rosenbrock example *****')


the_function = Rosenbrock()
bounds = Bounds([(None, None), (None, None)])
starting_point = np.array([-1.5, 1.5])

print('=====  With analytical hessian =====')
the_function.reset()
solution, messages = simple_bounds_newton_algorithm(
    the_function=the_function,
    bounds=bounds,
    starting_point=starting_point,
    proportion_analytical_hessian=1.0,
    first_radius=1,
)
print(f'{solution=}')
print(f'{messages=}')

print('=====  With BFGS =====')

the_function.reset()
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

the_function.reset()
solution, messages = simple_bounds_newton_algorithm(
    the_function=the_function,
    bounds=bounds,
    starting_point=starting_point,
    proportion_analytical_hessian=0.5,
    first_radius=1,
)
print(f'{solution=}')
print(f'{messages=}')

print('=====  Max iterations =====')

the_function.reset()
solution, messages = simple_bounds_newton_algorithm(
    the_function=the_function,
    bounds=bounds,
    starting_point=starting_point,
    proportion_analytical_hessian=0.5,
    first_radius=1,
    maxiter=10,
)

print(f'{solution=}')
print(f'{messages=}')
