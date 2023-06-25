"""File linesearch.py

:author: Michel Bierlaire
:date: Thu Jun 22 15:12:21 2023

Functions for line search algorithms
"""
import logging
import numpy as np
from abc import abstractmethod
from biogeme_optimization.exceptions import OptimizationError
from biogeme_optimization.function import relative_gradient
from biogeme_optimization.bfgs import inverse_bfgs
from biogeme_optimization.algebra import schnabel_eskow_direction

logger = logging.getLogger(__name__)

class DirectionManagement:
    """ Abstract class for the generation of a descent direction.
    """

    @abstractmethod
    def get_direction(self, iterate, delta_x=None, delta_g=None):
        """Obtain a descent direction
        
        :param iterate: current iterate
        :type iterate: numpy.array (n x 1)

        :param delta_x: difference of the two consecutive iterates
        :type delta_x: numpy.array (n x 1)

        :param delta_g: difference of the two consecutive gradients
        :type delta_g: numpy.array (n x 1)
        """

    @abstractmethod
    def get_value_function(self):
        """Obtain the vsalue of the function (for reporting) """
        pass
    
    @abstractmethod
    def number_of_function_evaluations(self):
        """Obtain the total number of function evaluations """
        pass
    
    @abstractmethod
    def number_of_gradient_evaluations(self):
        """Obtain the total number of gradient evaluations """
        pass
    
    @abstractmethod
    def number_of_hessian_evaluations(self):
        """Obtain the total number of hessian evaluations """
        pass
    
    @abstractmethod
    def update(self, iterate, delta_x, delta_g):
        """ Update the data necessary to calculate the direction.

        """
        pass

    GIVE THE RESPONSIBILITY OF VERIFYING THE RELATIVE GRADIENT TO THE DIRECTION MANAGEMENT OBJECT, SO THAT IT IS THE ONLY ONBJECT THAT CALLS THE FUNCTION
    
class NewtonDirection(DirectionManagement):
    """ Class for the generation of the Newton direction.
    """
    def __init(self, the_function):

        self.the_function = the_function
        self.number_function = 0
        self.number_gradient = 0
        self.number_hessian = 0
        self.value_function = None
        self.direction = None

    def number_of_function_evaluations(self):
        """Obtain the total number of function evaluations """
        return self.number_function
        
    def number_of_gradient_evaluations(self):
        """Obtain the total number of gradient evaluations """
        return self.number_gradient
    
    def number_of_hessian_evaluations(self):
        """Obtain the total number of hessian evaluations """
        return self.number_hessian
        
    def get_direction(self, iterate):
        """ Update the data necessary to calculate the direction.

        :param iterate: current iterate
        :type iterate: numpy.array (n x 1)

        :param delta_x: difference of the two consecutive iterates
        :type delta_x: numpy.array (n x 1)

        :param delta_g: difference of the two consecutive gradients
        :type delta_g: numpy.array (n x 1)
        """
        self.the_function.set_variables(iterate)
        self.value_function, gradient, hessian = self.the_function.f_g_h()
        self.number_function += 1
        self.number_gradient += 1
        self.number_hessian += 1
        direction = schnabel_eskow_direction(gradient, hessian)
        return direction
        
class InverseBfgsDirection(DirectionManagement):
    """ Class for the generation of the inverse BFGS direction.
    """
    def __init__(self, dimension, first_inverse_approximation=None):
        """Constructor

        :param first_inverse_approximation: first inverse approximation of the Hessian
        :type first_inverse_approximation: numpy.array (n x n)
        """
        
        if first_inverse_approximation is None:
            self.inverse_hessian = np.identity(dimension)
        else:
            if first_inverse_approximation.ndim != dimension:
                error_msg = (
                    f'Incompatible dimensions: {dimension} '
                    f'and {first_inverse_approximation.ndim}'
                )
                raise OptimizationError(error_msg)
            self.inverse_hessian_approx = first_inverse_approximation
        self.last_iterate = None
        self.last_gradient = None

    def get_direction(self, iterate):
        """Obtain a descent direction

        :param iterate: current iterate
        :type iterate: numpy.array (n x 1)

        :param delta_x: difference of the two consecutive iterates
        :type delta_x: numpy.array (n x 1)

        :param delta_g: difference of the two consecutive gradients
        :type delta_g: numpy.array (n x 1)
        """
        self.the_function.set_variables(iterate)
        self.value_function, gradient = self.the_function.f_g()
        self.number_function += 1
        self.number_gradient += 1
        # The first time, there is nothinh to update
        if self.last_iterate is not None:
            delta_x = iterate - self.last_iterate
            delta_g = gradient - self.last_gradient
            # Update the approximation, if possible
            try:
                self.inverse_hessian_approx = inverse_bfgs(
                    self.inverse_hessian_approx,
                    delta_x,
                    delta_g)
            except OptimizationError as e:
                logger.warning(e)
        self.last_iterate = iterate
        self.last_gradient = gradient
        
        # Calculate and return the direction
        direction = - self.inverse_hessian_approx @ gradient
        return direction
    
def linesearch(
    fct, iterate, descent_direction, alpha0=1.0, beta1=1.0e-4, beta2=0.99, lbd=2.0
):
    """
    Calculate a step along a direction that satisfies both Wolfe conditions

    :param fct: object to calculate the objective function and its derivatives.
    :type fct: function_to_minimize

    :param iterate: current iterate.
    :type iterate: numpy.array

    :param descent_direction: descent direction.
    :type descent_direction: numpy.array

    :param alpha0: first step to test.
    :type alpha0: float

    :param beta1: parameter of the first Wolfe condition.
    :type beta1: float

    :param beta2: parameter of the second Wolfe condition.
    :type beta2: float

    :param lbd: expansion factor for a short step.
    :type lbd: float

    :return: a step verifing both Wolfe conditions
    :rtype: float

    :raises OptimizationError: if ``lbd`` :math:`\\leq` 1
    :raises OptimizationError: if ``alpha0`` :math:`\\leq` 0
    :raises OptimizationError: if ``beta1`` :math:`\\geq` beta2
    :raises OptimizationError: if ``descent_direction`` is not a descent
                                             direction

    """
    MAX_ITERATIONS = 1000

    if lbd <= 1:
        raise OptimizationError(f'lambda is {lbd} and must be > 1')
    if alpha0 <= 0:
        raise OptimizationError(f'alpha0 is {alpha0} and must be > 0')
    if beta1 >= beta2:
        error_msg = (
            f'Incompatible Wolfe cond. parameters: '
            f'beta1= {beta1} is greater than '
            f'beta2={beta2}'
        )
        raise OptimizationError(error_msg)

    fct.set_variables(iterate)
    function, gradient = fct.f_g()

    nbr_function_evaluations = 1
    deriv = np.inner(gradient, descent_direction)

    if deriv >= 0:
        raise OptimizationError(
            f'descent_direction is not a descent direction: {deriv} >= 0'
        )

    alpha = alpha0
    alphal = 0
    alphar = np.inf
    for _ in range(MAX_ITERATIONS):
        candidate = iterate + alpha * descent_direction
        fct.set_variables(candidate)
        value_candidate, gradient_candidate = fct.f_g()
        nbr_function_evaluations += 1
        # First Wolfe condition violated?
        if value_candidate > function + alpha * beta1 * deriv:
            alphar = alpha
            alpha = (alphal + alphar) / 2.0
        elif np.inner(gradient_candidate, descent_direction) < beta2 * deriv:
            alphal = alpha
            if alphar == np.inf:
                alpha = lbd * alpha
            else:
                alpha = (alphal + alphar) / 2.0
        else:
            break
    else:
        raise OptimizationError(
            f'Line search algorithm could not find a step verifying both Wolfe '
            f'conditions after {MAX_ITERATIONS} iterations.'
        )
    return alpha, nbr_function_evaluations

def minimization_with_linesearch(
    fct,
    starting_point,
    descent_direction,
    eps=np.finfo(np.float64).eps ** 0.3333,
    maxiter=1000,
):
    """Minimization algorithm with inexact line search (Wolfe conditions)

    :param fct: object to calculate the objective function and its derivatives.
    :type fct: optimization.functionToMinimize

    :param starting_point: starting point
    :type starting_point: numpy.array (n x 1)

    :param starting_value: value of the objective function at the starting point
    :type starting_value: float
    
    :param starting_gradient: gradient at the starting point
    :type starting_point: numpy.array (n x 1)

    :param descent_direction: object in charge of calculating the descent directions
    :type descent_direction: DirectionManagement

    :param eps: the algorithm stops when this precision is reached.
                 Default: :math:`\\varepsilon^{\\frac{1}{3}}`
    :type eps: float

    :param maxiter: the algorithm stops if this number of iterations
                    is reached. Default: 1000
    :type maxiter: int

    :return: tuple x, messages, where

            - x is the solution found,

            - messages is a dictionary reporting various aspects
              related to the run of the algorithm.

    :rtype: numpy.array, dict(str:object)

    :raises OptimizationError: if the dimensions of the
             matrix initBfgs do not match the length of starting_point.

    """
    n = len(starting_point)
    xk = starting_point
    typx = np.ones(np.asarray(xk).shape)
    typf = max(np.abs(f), 1.0)
    relgrad = relative_gradient(xk, f, g, typx, typf)
    if relgrad <= eps:
        message = f'Relative gradient = {relgrad:.2g} <= {eps:.2g}'
        messages = {
            'Algorithm': 'Inverse BFGS with line search',
            'Relative gradient': relgrad,
            'Cause of termination': message,
            'Number of iterations': 0,
            'Number of function evaluations': nfev,
            'Number of gradient evaluations': ngev,
        }
        return xk, messages
    nfev = 0
    cont = True
    for k in range(maxiter):
        d = -Hinv @ g
        alpha, nfls = line_search(fct, xk, f, g, d)
        nfev += nfls
        delta = alpha * d
        xk = xk + delta
        gprev = g
        fct.set_variables(xk)
        f, g = fct.f_g()
        nfev += 1
        ngev += 1
        Hinv = inverse_bfgs(Hinv, delta, g - gprev)
        nfev += 1
        relgrad = relative_gradient(xk, f, g, typx, typf)
        if relgrad <= eps:
            message = f'Relative gradient = {relgrad:.2g} <= {eps:.2g}'
            cont = False
        if k == maxiter:
            message = f'Maximum number of iterations reached: {maxiter}'
            cont = False
        #logger.debug(f'{k} f={f:10.7g} relgrad={relgrad:6.2g}' f' alpha={alpha:6.2g}')
    messages = {
        'Algorithm': 'Inverse BFGS with line search',
        'Relative gradient': relgrad,
        'Cause of termination': message,
        'Number of iterations': k,
        'Number of function evaluations': nfev,
        'Number of gradient evaluations': ngev,
    }

    return xk, messages


def newton_linesearch(
    fct, starting_point, eps=np.finfo(np.float64).eps ** 0.3333, maxiter=100
):
    """
    Newton method with inexact line search (Wolfe conditions)

    :param fct: object to calculate the objective function and its derivatives.
    :type fct: optimization.functionToMinimize

    :param starting_point: starting point
    :type starting_point: numpy.array

    :param eps: the algorithm stops when this precision is reached.
                 Default: :math:`\\varepsilon^{\\frac{1}{3}}`
    :type eps: float

    :param maxiter: the algorithm stops if this number of iterations
                    is reached. Defaut: 100
    :type maxiter: int

    :return: x, messages

        - x is the solution generated by the algorithm,
        - messages is a dictionary describing information about the lagorithm

    :rtype: numpay.array, dict(str:object)

    """

    xk = starting_point
    fct.set_variables(xk)
    f, g, h = fct.f_g_h()
    nfev = 1
    ngev = 1
    nhev = 1
    typx = np.ones_like(xk)
    typf = max(np.abs(f), 1.0)
    relgrad = relative_gradient(xk, f, g, typx, typf)
    if relgrad <= eps:
        message = f'Relative gradient = {relgrad:.3g} <= {eps:.2g}'
        messages = {
            'Algorithm': 'Unconstrained Newton with line search',
            'Relative gradient': relgrad,
            'Number of iterations': 0,
            'Number of function evaluations': nfev,
            'Number of gradient evaluations': ngev,
            'Number of hessian evaluations': nhev,
            'Cause of termination': message,
        }
        return xk, messages

    for k in range(maxiter):
        direction = schnabel_eskow_direction(g, h)
        alpha, nfls = linesearch(fct, xk, direction)
        nfev += nfls
        ngev += nfls
        xk += alpha * direction
        fct.set_variables(xk)
        f, g, h = fct.f_g_h()
        nfev += 1
        ngev += 1
        nhev += 1
        typf = max(np.abs(f), 1.0)
        relgrad = relative_gradient(xk, f, g, typx, typf)
        if relgrad <= eps:
            message = f'Relative gradient = {relgrad:.2g} <= {eps:.2g}'
            break
        logger.debug(f'{k+1} f={f:10.7g} relgrad={relgrad:6.2g}' f' alpha={alpha:6.2g}')
    else:
        message = f'Maximum number of iterations reached: {maxiter}'

    messages = {
        'Algorithm': 'Unconstrained Newton with line search',
        'Relative gradient': relgrad,
        'Number of iterations': k + 1,
        'Number of function evaluations': nfev,
        'Number of gradient evaluations': ngev,
        'Number of hessian evaluations': nhev,
        'Cause of termination': message,
    }

    return xk, messages

def bfgs_line_search(
    fct,
    starting_point,
    init_bfgs=None,
    eps=np.finfo(np.float64).eps ** 0.3333,
    maxiter=1000,
):
    """BFGS method with inexact line search (Wolfe conditions)

    :param fct: object to calculate the objective function and its derivatives.
    :type fct: optimization.functionToMinimize

    :param starting_point: starting point
    :type starting_point: numpy.array

    :param initBfgs: matrix used to initialize BFGS. If None, the
                     identity matrix is used. Default: None.
    :type initBfgs: numpy.array

    :param eps: the algorithm stops when this precision is reached.
                 Default: :math:`\\varepsilon^{\\frac{1}{3}}`
    :type eps: float

    :param maxiter: the algorithm stops if this number of iterations
                    is reached. Default: 1000
    :type maxiter: int

    :return: tuple x, messages, where

            - x is the solution found,

            - messages is a dictionary reporting various aspects
              related to the run of the algorithm.

    :rtype: numpy.array, dict(str:object)

    :raises biogeme.exceptions.BiogemeError: if the dimensions of the
             matrix initBfgs do not match the length of starting_point.

    """

    n = len(starting_point)
    xk = starting_point
    fct.setVariables(xk)
    f, g = fct.f_g()
    nfev = 1
    ngev = 1
    if init_bfgs is None:
        Hinv = np.identity(n)
    else:
        if init_bfgs.shape != (n, n):
            errorMsg = (
                f'BFGS must be initialized with a {n}x{n} '
                f'matrix and not a {init_bfgs.shape[0]}'
                'x{initBfgs.shape[1]} matrix.'
            )
            raise OptimizationError(errorMsg)
        Hinv = init_bfgs
    typx = np.ones(np.asarray(xk).shape)
    typf = max(np.abs(f), 1.0)
    relgrad = relative_gradient(xk, f, g, typx, typf)
    if relgrad <= eps:
        message = f'Relative gradient = {relgrad:.2g} <= {eps:.2g}'
        messages = {
            'Algorithm': 'Inverse BFGS with line search',
            'Relative gradient': relgrad,
            'Cause of termination': message,
            'Number of iterations': 0,
            'Number of function evaluations': nfev,
            'Number of gradient evaluations': ngev,
        }
        return xk, messages
    nfev = 0
    cont = True
    for k in range(maxiter):
        d = -Hinv @ g
        alpha, nfls = line_search(fct, xk, f, g, d)
        nfev += nfls
        delta = alpha * d
        xk = xk + delta
        gprev = g
        fct.set_variables(xk)
        f, g = fct.f_g()
        nfev += 1
        ngev += 1
        Hinv = inverse_bfgs(Hinv, delta, g - gprev)
        nfev += 1
        relgrad = relative_gradient(xk, f, g, typx, typf)
        if relgrad <= eps:
            message = f'Relative gradient = {relgrad:.2g} <= {eps:.2g}'
            cont = False
        if k == maxiter:
            message = f'Maximum number of iterations reached: {maxiter}'
            cont = False
        #logger.debug(f'{k} f={f:10.7g} relgrad={relgrad:6.2g}' f' alpha={alpha:6.2g}')
    messages = {
        'Algorithm': 'Inverse BFGS with line search',
        'Relative gradient': relgrad,
        'Cause of termination': message,
        'Number of iterations': k,
        'Number of function evaluations': nfev,
        'Number of gradient evaluations': ngev,
    }

    return xk, messages
