"""File trust_region.py

:author: Michel Bierlaire
:date: Fri Jun 23 09:27:17 2023

Functions for trust region algorithms
"""
import logging
from enum import Enum, auto
import numpy as np
import scipy.linalg as la
from biogeme_optimization.exceptions import OptimizationError
from biogeme_optimization.function import relative_gradient
from biogeme_optimization.algebra import schnabel_eskow_direction

logger = logging.getLogger(__name__)


class DoglegDiagnostic(Enum):
    """Possible aoutcomes of the dogleg method"""

    NEGATIVE_CURVATURE = auto()
    CAUCHY = auto()
    PARTIAL_CAUCHY = auto()
    NEWTON = auto()
    PARTIAL_NEWTON = auto()
    DOGLEG = auto()


class ConjugateGradientDiagnostic(Enum):
    """Possible aoutcomes of the dogleg method"""

    CONVERGENCE = auto()
    OUT_OF_TRUST_REGION = auto()
    NEGATIVE_CURVATURE = auto()
    NUMERICAL_PROBLEM = auto()


def trust_region_intersection(
    inside_direction, crossing_direction, radius, check_step=True
):
    """Calculates the intersection with the boundary of the trust region.

    Consider a trust region of radius :math:`\\delta`, centered at
    :math:`\\hat{x}`. Let :math:`x_c` be in the trust region, and
    :math:`d_c = x_c - \\hat{x}`, so that :math:`\\|d_c\\| \\leq
    \\delta`. Let :math:`x_d` be out of the trust region, and
    :math:`d_d = x_d - \\hat{x}`, so that :math:`\\|d_d\\| \\geq
    \\delta`.  We calculate :math:`\\lambda` such that

    .. math:: \\| d_c + \\lambda (d_d - d_c)\\| = \\delta

    :param inside_direction: xc - xhat.
    :type inside_direction: numpy.array
    :param crossing_direction: dd - dc
    :type crossing_direction: numpy.array
    :param radius: radius of the trust region.
    :type radius: float

    :return: :math:`\\lambda` such that :math:`\\| d_c +
              \\lambda (d_d - d_c)\\| = \\delta`

    :rtype: float

    """
    a = np.inner(crossing_direction, crossing_direction)
    b = 2 * np.inner(inside_direction, crossing_direction)
    c = np.inner(inside_direction, inside_direction) - radius**2
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0:
        raise OptimizationError('No intersection')
    step = (-b + np.sqrt(discriminant)) / (2 * a)
    if check_step:
        if step < 0:
            raise OptimizationError('Starting point outside trust region')
        if step > 1:
            raise OptimizationError('Target point inside trust region')
    return step


def cauchy_newton_dogleg(gradient, hessian):
    """Calculate the Cauchy, the Newton and the dogleg points.

    The Cauchy point is defined as

    .. math:: d_c = - \\frac{\\nabla f(x)^T \\nabla f(x)}
                    {\\nabla f(x)^T \\nabla^2 f(x)
                    \\nabla f(x)} \\nabla f(x)

    The Newton point :math:`d_n` verifies Newton equation:

    .. math:: H_s d_n = - \\nabla f(x)

    where :math:`H_s` is a positive definite matrix generated with the
    method by `Schnabel and Eskow (1999)`_.

    The Dogleg point is

    .. math:: d_d = \\eta d_n

    where

    .. math:: \\eta = 0.2 + 0.8 \\frac{\\alpha^2}{\\beta |\\nabla f(x)^T d_n|}

    and :math:`\\alpha= \\nabla f(x)^T \\nabla f(x)`,
    :math:`\\beta=\\nabla f(x)^T \\nabla^2 f(x)\\nabla f(x).`

    :param gradient: gradient :math:`\\nabla f(x)`

    :type gradient: numpy.array

    :param hessian: hessian :math:`\\nabla^2 f(x)`

    :type hessian: numpy.array

    :return: tuple with Cauchy point, Newton point, Dogleg point. The
        Cauchy point in None if the curvature is negative along the
        gradient direction. The Newton and the Dogleg directions are
        None if the curvature is negative along the Newton direction.
    :rtype: numpy.array, numpy.array, numpy.array

    """
    alpha = np.inner(gradient, gradient)
    beta = np.inner(gradient, hessian @ gradient)
    if beta > 0:
        d_cauchy = -(alpha / beta) * gradient
    else:
        d_cauchy = None
    try:
        d_newton = schnabel_eskow_direction(
            gradient=gradient, hessian=hessian, check_convexity=True
        )
    except OptimizationError:
        return d_cauchy, None, None
    eta = 0.2 + (0.8 * alpha * alpha / (beta * abs(np.inner(gradient, d_newton))))
    return d_cauchy, d_newton, eta * d_newton


def dogleg(gradient, hessian, radius):
    """
    Find an approximation of the trust region subproblem using
    the dogleg method

    :param gradient: gradient of the quadratic model.
    :type gradient: numpy.array
    :param hessian: hessian of the quadratic model.
    :type hessian: numpy.array
    :param radius: radius of the trust region.
    :type radius: float

    :return: d, diagnostic where

          - d is an approximate solution of the trust region subproblem
          - diagnostic is the nature of the solution:

             * NEGATIVE_CURVATURE if negative curvature along Cauchy direction
               (i.e. along the gradient)
             * CAUCHY if Cauchy step
             * PARTIAL_CAUCHY partial Cauchy step
             * NEWTON if Newton step
             * PARTIAL_NEWTON if partial Newton step
             * DOGLEG if Dogleg

    :rtype: numpy.array, int
    """

    # Check if the inputs are valid NumPy arrays
    if not isinstance(gradient, np.ndarray) or not isinstance(hessian, np.ndarray):
        raise OptimizationError('gradient and hessian must be NumPy arrays')
    if gradient.ndim != 1 or hessian.ndim != 2:
        raise OptimizationError(
            'gradient must be a 1-dimensional array and '
            'hessian must be a 2-dimensional array'
        )
    if gradient.shape[0] != hessian.shape[0] or hessian.shape[0] != hessian.shape[1]:
        raise OptimizationError("gradient and hessian must have compatible dimensions")

    cauchy_dir, newton_dir, dogleg_dir = cauchy_newton_dogleg(gradient, hessian)

    # Check if the model is convex along the gradient direction
    if cauchy_dir is None:
        dstar = -radius * gradient / la.norm(gradient)
        return dstar, DoglegDiagnostic.NEGATIVE_CURVATURE

    alpha = np.inner(gradient, gradient)
    beta = np.inner(gradient, hessian @ gradient)
    normdc = alpha * np.sqrt(alpha) / beta
    if normdc >= radius:
        # The Cauchy point is outside the trust
        # region. We move along the Cauchy
        # direction until the border of the trust
        # region.
        dstar = (radius / normdc) * cauchy_dir
        return dstar, DoglegDiagnostic.PARTIAL_CAUCHY

    # Check the convexity of the model along Newton direction
    if newton_dir is None:
        # Return the Cauchy point
        return cauchy_dir, DoglegDiagnostic.CAUCHY

    # Compute Newton point
    normdn = la.norm(newton_dir)

    if normdn <= radius:
        # Newton point is inside the trust region
        return newton_dir, DoglegDiagnostic.NEWTON

    # Compute the dogleg point

    eta = 0.2 + (0.8 * alpha * alpha / (beta * abs(np.inner(gradient, newton_dir))))

    partial_newton = eta * normdn

    if partial_newton <= radius:
        # Dogleg point is inside the trust region
        dstar = (radius / normdn) * newton_dir
        return dstar, DoglegDiagnostic.PARTIAL_NEWTON

    # Between Cauchy and dogleg
    crossing_direction = dogleg_dir - cauchy_dir
    lbd = trust_region_intersection(cauchy_dir, crossing_direction, radius)
    dstar = cauchy_dir + lbd * crossing_direction
    return dstar, DoglegDiagnostic.DOGLEG


def truncated_conjugate_gradient(gradient, hessian, radius, tol=1.0e-6):
    """Find an approximation of the trust region subproblem using the
    truncated conjugate gradient method

    :param gradient: gradient of the quadratic model.
    :type gradient: numpy.array

    :param hessian: hessian of the quadrartic model.
    :type hessian: numpy.array

    :param radius: radius of the trust region.
    :type radius: float

    :param tol: tolerance on the norm of the gradient, used as a stopping criterion
    :type tol: float

    :return: d, diagnostic, where

          - d is the approximate solution of the trust region subproblem,
          - diagnostic is the nature of the solution:

            * CONVERGENCE for convergence,
            * OUT_OF_TRUST_REGION if out of the trust region,
            * NEGATIVE_CURVATURE if negative curvature detected.
            * NUMERICAL_PROBLEM if a numerical problem has been encountered

    :rtype: numpy.array, ConjugateGradientDiagnostic

    """
    dimension = len(gradient)
    xk = np.zeros(dimension)
    gk = gradient
    dk = -gk
    for _ in range(dimension):
        try:
            curv = np.inner(dk, hessian @ dk)
            if curv <= 0:
                # Negative curvature has been detected
                diagnostic = ConjugateGradientDiagnostic.NEGATIVE_CURVATURE
                step = trust_region_intersection(xk, dk, radius, check_step=False)
                solution = xk + step * dk
                return solution, diagnostic
            alphak = -np.inner(dk, gk) / curv
            xkp1 = xk + alphak * dk
            if np.isnan(xkp1).any() or la.norm(xkp1) > radius:
                # Out of the trust region
                diagnostic = ConjugateGradientDiagnostic.OUT_OF_TRUST_REGION
                step = trust_region_intersection(xk, dk, radius, check_step=False)
                solution = xk + step * dk
                return solution, diagnostic
            xk = xkp1
            gkp1 = hessian @ xk + gradient
            betak = np.inner(gkp1, gkp1) / np.inner(gk, gk)
            dk = -gkp1 + betak * dk
            gk = gkp1
            if la.norm(gkp1) <= tol:
                diagnostic = ConjugateGradientDiagnostic.CONVERGENCE
                step = xk
                return step, diagnostic
        except ValueError:
            # Numerical problem. We follow the last direction until the border of the trust region
            diagnostic = ConjugateGradientDiagnostic.NUMERICAL_PROBLEM
            step = trust_region_intersection(xk, dk, radius, check_step=False)
            solution = xk + step * dk
            return solution, diagnostic
    diagnostic = ConjugateGradientDiagnostic.CONVERGENCE
    step = xk
    return step, diagnostic


def newton_trust_region(
    fct,
    starting_point,
    delta0=1.0,
    eps=np.finfo(np.float64).eps ** 0.3333,
    use_dogleg=False,
    maxiter=1000,
    eta1=0.01,
    eta2=0.9,
):
    """Newton method with trust region

    :param fct: object to calculate the objective function and its derivatives.
    :type fct: optimization.functionToMinimize

    :param starting_point: starting point
    :type starting_point: numpy.array

    :param delta0: initial radius of the trust region. Default: 100.
    :type delta0: float

    :param eps: the algorithm stops when this precision is reached.
              Default: :math:`\\varepsilon^{\\frac{1}{3}}`
    :type eps: float

    :param use_dogleg: If True, the Dogleg method is used to solve the
       trut region subproblem. If False, the truncated conjugate
       gradient is used. Default: False.
    :type use_dogleg: bool

    :param maxiter: the algorithm stops if this number of iterations
                    is reached. Default: 1000.

    :type maxiter: int

    :param eta1: threshold for failed iterations. Default: 0.01.
    :type eta1: float

    :param eta2: threshold for very successful iterations. Default 0.9.
    :type eta2: float

    :return: tuple x, messages, where

            - x is the solution found,
            - messages is a dictionary reporting various aspects
              related to the run of the algorithm.

    :rtype: numpy.array, dict(str:object)

    """

    current_iterate = starting_point
    fct.set_variables(current_iterate)
    value_function, gradient, hessian = fct.f_g_h()
    nfev = 1
    ngev = 1
    nhev = 1
    typx = np.ones(np.asarray(current_iterate).shape)
    typf = max(np.abs(value_function), 1.0)
    relgrad = relative_gradient(current_iterate, value_function, gradient, typx, typf)
    if relgrad <= eps:
        message = f'Relative gradient = {relgrad:.2g} <= {eps:.2g}'
        messages = {
            'Algorithm': 'Unconstrained Newton with trust region',
            'Relative gradient': relgrad,
            'Cause of termination': message,
            'Number of iterations': 0,
            'Number of function evaluations': nfev,
            'Number of gradient evaluations': ngev,
            'Number of hessian evaluations': nhev,
        }
        return current_iterate, messages
    delta = delta0
    nfev = 0
    max_delta = np.finfo(float).max
    min_delta = np.finfo(float).eps
    rho = 0.0
    for k in range(maxiter):
        if use_dogleg:
            step, _ = dogleg(gradient, hessian, delta)
        else:
            step, _ = truncated_conjugate_gradient(gradient, hessian, delta)
        candidate = current_iterate + step
        fct.set_variables(candidate)
        # Calculate the value of the function
        value_function_candidate = fct.f()
        nfev += 1
        num = value_function - value_function_candidate
        denom = -np.inner(step, gradient) - 0.5 * np.inner(step, hessian @ step)
        rho = num / denom
        if rho < eta1:
            # Failure: reduce the trust region
            delta = la.norm(step) / 2.0
            status = '-'
        else:
            # Candidate accepted
            (
                value_function_candidate,
                gradient_candidate,
                hessian_candidate,
            ) = fct.f_g_h()
            nfev += 1
            ngev += 1
            nhev += 1
            current_iterate = candidate
            value_function = value_function_candidate
            gradient = gradient_candidate
            hessian = hessian_candidate
            if rho >= eta2:
                # Enlarge the trust region
                delta = min(2 * delta, max_delta)
                status = '++'
            else:
                status = '+'
            relgrad = relative_gradient(
                current_iterate, value_function, gradient, typx, typf
            )
            if relgrad <= eps:
                message = f'Relative gradient = {relgrad:.2g} <= {eps:.2g}'
                break
        if delta <= min_delta:
            message = f'Trust region is too small: {delta}'
            break
        logger.debug(
            f'{k+1} f={value_function:10.7g} relgrad={relgrad:6.2g} '
            f'delta={delta:6.2g} '
            f'rho={rho:6.2g} {status}'
        )
    else:
        message = f'Maximum number of iterations reached: {maxiter}'

    messages = {
        'Algorithm': 'Unconstrained Newton with trust region',
        'Relative gradient': relgrad,
        'Cause of termination': message,
        'Number of iterations': k + 1,
        'Number of function evaluations': nfev,
        'Number of gradient evaluations': ngev,
        'Number of hessian evaluations': nhev,
    }

    return current_iterate, messages
