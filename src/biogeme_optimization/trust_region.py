"""File trust_region.py

:author: Michel Bierlaire
:date: Fri Jun 23 09:27:17 2023

Functions for trust region algorithms
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping

import numpy as np

from biogeme_optimization.algebra import schnabel_eskow_direction
from biogeme_optimization.bfgs import bfgs
from biogeme_optimization.diagnostics import (
    ConjugateGradientDiagnostic,
    Diagnostic,
    DoglegDiagnostic,
    OptimizationResults,
)
from biogeme_optimization.exceptions import OptimizationError
from biogeme_optimization.function import FunctionData, FunctionToMinimize
from biogeme_optimization.state import (
    TRUST_REGION_BFGS_ALGORITHM,
    CheckpointError,
    TrustRegionBFGSState,
)

logger = logging.getLogger(__name__)


class QuadraticModel(ABC):
    """Abstract class for the generation of the quadratic model."""

    def __init__(self, function: FunctionToMinimize):
        """Constructor

        :param function: the function to minimize
        :type function: FunctionToMinimize

        """
        self.the_function: FunctionToMinimize = function

    @abstractmethod
    def get_f_g_h(self, iterate: np.ndarray) -> FunctionData:
        """Obtain the vector g and the matrix h characterizing the
            model, as well as the canonical_value of the function

        :param iterate: current iterate
        :type iterate: numpy.array (n x 1)

        :return: function, gradient and hessian
        :rtype: tuple(float, numpy.array (nx1), numpy.array (nxn)
        """


class NewtonModel(QuadraticModel):
    """Call implementing the quadratic model based on the analytical hessian"""

    def get_f_g_h(self, iterate: np.ndarray) -> FunctionData:
        """Obtain the vector g and the matrix h characterizing the model

        :param iterate: current iterate
        :type iterate: numpy.array (n x 1)

        :return: function, gradient and hessian
        :rtype: tuple(float, numpy.array (nx1), numpy.array (nxn)
        """
        self.the_function.set_variables(iterate)
        evaluation = self.the_function.f_g_h()
        optimal = self.the_function.check_optimality()
        if optimal:
            return FunctionData(function=None, gradient=None, hessian=None)
        return evaluation


class BfgsModel(QuadraticModel):
    """Call implementing the quadratic model based on the BFGS approximation"""

    def __init__(
        self,
        function: FunctionToMinimize,
        first_approximation: np.ndarray | None = None,
    ):
        """Constructor

        :param first_approximation: first approximation of the Hessian
        :type first_approximation: numpy.array (n x n)

        :return: function, gradient and hessian
        :rtype: tuple(float, numpy.array (nx1), numpy.array (nxn)
        """
        super().__init__(function)

        if first_approximation is None:
            self.hessian_approx: np.ndarray = np.identity(self.the_function.dimension())
        else:
            dimension = self.the_function.dimension()
            if first_approximation.shape != (dimension, dimension):
                error_msg = (
                    f'Incompatible dimensions: expected {(dimension, dimension)} '
                    f'and got {first_approximation.shape}'
                )
                raise OptimizationError(error_msg)
            self.hessian_approx = np.array(first_approximation, copy=True)
        self.last_iterate: np.ndarray | None = None
        self.last_gradient: np.ndarray | None = None
        self.last_function: float | None = None
        self.last_evaluation_was_optimal = False

    def get_f_g_h(self, iterate: np.ndarray) -> FunctionData:
        """Obtain the vector g and the matrix h characterizing the model

        :param iterate: current iterate
        :type iterate: numpy.array (n x 1)

        """
        self.the_function.set_variables(iterate)
        evaluation = self.the_function.f_g()
        self.last_function = evaluation.function
        self.last_gradient = (
            None if evaluation.gradient is None else evaluation.gradient.copy()
        )
        optimal = self.the_function.check_optimality()
        if optimal:
            self.last_iterate = iterate.copy()
            self.last_evaluation_was_optimal = True
            return FunctionData(function=None, gradient=None, hessian=None)
        self.last_evaluation_was_optimal = False
        # The first time, there is nothing to update
        if self.last_iterate is not None:
            delta_x = iterate - self.last_iterate
            delta_g = evaluation.gradient - self.last_gradient
            # Update the approximation, if possible
            try:
                self.hessian_approx = bfgs(self.hessian_approx, delta_x, delta_g)
            except OptimizationError as err:
                logger.warning(err)
        self.last_iterate = iterate.copy()
        self.last_gradient = evaluation.gradient.copy()
        return FunctionData(
            function=evaluation.function,
            gradient=evaluation.gradient,
            hessian=self.hessian_approx,
        )


def trust_region_intersection(
    inside_direction: np.ndarray,
    crossing_direction: np.ndarray,
    radius: float,
    check_step: bool = True,
) -> float:
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

    :param check_step: if True, an exception is raised if the canonical_value of
        the step is out of the interval [0,1]
    :type check_step: bool

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


def cauchy_newton_dogleg(
    gradient: np.ndarray, hessian: np.ndarray
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
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


def dogleg(
    gradient: np.ndarray, hessian: np.ndarray, radius: float
) -> tuple[np.ndarray, DoglegDiagnostic]:
    """
    Find an approximation of the trust region sub-problem using
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
        d_star = -radius * gradient / np.linalg.norm(gradient)
        return d_star, DoglegDiagnostic.NEGATIVE_CURVATURE

    alpha = np.inner(gradient, gradient)
    beta = np.inner(gradient, hessian @ gradient)
    norm_dc = alpha * np.sqrt(alpha) / beta
    if norm_dc >= radius:
        # The Cauchy point is outside the trust
        # region. We move along the Cauchy
        # direction until the border of the trust
        # region.
        d_star = (radius / norm_dc) * cauchy_dir
        return d_star, DoglegDiagnostic.PARTIAL_CAUCHY

    # Check the convexity of the model along Newton direction
    if newton_dir is None:
        # Return the Cauchy point
        return cauchy_dir, DoglegDiagnostic.CAUCHY

    # Compute Newton point
    norm_dn = np.linalg.norm(newton_dir)

    if norm_dn <= radius:
        # Newton point is inside the trust region
        return newton_dir, DoglegDiagnostic.NEWTON

    # Compute the dogleg point

    eta = 0.2 + (0.8 * alpha * alpha / (beta * abs(np.inner(gradient, newton_dir))))

    partial_newton = eta * norm_dn

    if partial_newton <= radius:
        # Dogleg point is inside the trust region
        d_star = (radius / norm_dn) * newton_dir
        return d_star, DoglegDiagnostic.PARTIAL_NEWTON

    # Between Cauchy and dogleg
    crossing_direction = dogleg_dir - cauchy_dir
    lbd = trust_region_intersection(cauchy_dir, crossing_direction, radius)
    d_star = cauchy_dir + lbd * crossing_direction
    return d_star, DoglegDiagnostic.DOGLEG


def truncated_conjugate_gradient(
    gradient: np.ndarray, hessian: np.ndarray, radius: float, tol: float = 1.0e-6
) -> tuple[np.ndarray, ConjugateGradientDiagnostic]:
    """Find an approximation of the trust region sub-problem using the
    truncated conjugate gradient method

    :param gradient: gradient of the quadratic model.
    :type gradient: numpy.array

    :param hessian: hessian of the quadratic model.
    :type hessian: numpy.array

    :param radius: radius of the trust region.
    :type radius: float

    :param tol: tolerance on the norm of the gradient, used as a stopping criterion
    :type tol: float

    :return: d, diagnostic, where

          - d is the approximate solution of the trust region sub-problem,
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
            curvature = np.inner(dk, hessian @ dk)
            if curvature <= 0:
                # Negative curvature has been detected
                diagnostic = ConjugateGradientDiagnostic.NEGATIVE_CURVATURE
                step = trust_region_intersection(xk, dk, radius, check_step=False)
                solution = xk + step * dk
                return solution, diagnostic
            alphak = -np.inner(dk, gk) / curvature
            xkp1 = xk + alphak * dk
            if np.isnan(xkp1).any() or np.linalg.norm(xkp1) > radius:
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
            if np.linalg.norm(gkp1) <= tol:
                diagnostic = ConjugateGradientDiagnostic.CONVERGENCE
                step = xk
                return step, diagnostic
        except ValueError:
            # Numerical problem. We follow the last direction until
            # the border of the trust region
            diagnostic = ConjugateGradientDiagnostic.NUMERICAL_PROBLEM
            step = trust_region_intersection(xk, dk, radius, check_step=False)
            solution = xk + step * dk
            return solution, diagnostic
    diagnostic = ConjugateGradientDiagnostic.CONVERGENCE
    step = xk
    return step, diagnostic


def minimization_with_trust_region(
    the_function: FunctionToMinimize,
    starting_point: np.ndarray,
    quadratic_model: QuadraticModel,
    solving_trust_region_subproblem: Callable[
        [np.ndarray, np.ndarray, float], tuple[np.ndarray, Diagnostic]
    ],
    maxiter: int = 1000,
    initial_radius: float = 1.0,
    eta1: float = 0.01,
    eta2: float = 0.9,
):
    """Minimization method with trust region

    :param the_function: object to calculate the objective function and its derivatives.
    :type the_function: optimization.functionToMinimize

    :param starting_point: starting point
    :type starting_point: numpy.array

    :param quadratic_model: object in charge of generating the quadratic model
    :type quadratic_model: QuadraticModel

    :param solving_trust_region_subproblem: function in charge of
        solving the trust region sub-problem


    :param maxiter: the algorithm stops if this number of iterations
                    is reached. Default: 1000.
    :type maxiter: int

    :param initial_radius: initial radius of the trust region.
    :type initial_radius: float

    :param eta1: threshold for failed iterations. Default: 0.01.
    :type eta1: float

    :param eta2: threshold for very successful iterations. Default 0.9.
    :type eta2: float

    :return: named tuple:

            - solution is the solution found,
            - message is a dictionary reporting various aspects
              related to the run of the algorithm.
            - convergence is a bool which is True if the algorithm has converged

    :rtype: OptimizationResults

    """
    xk = starting_point
    the_output: FunctionData = quadratic_model.get_f_g_h(xk)
    value_iterate = the_output.function
    g = the_output.gradient
    h = the_output.hessian
    radius = initial_radius
    max_delta = np.finfo(float).max
    min_delta = np.finfo(float).eps
    for k in range(maxiter):
        # If the iterate is optimal, the returned canonical_value is None
        if value_iterate is None:
            messages = the_function.messages
            messages['Number of iterations'] = k
            optimization_results = OptimizationResults(
                solution=xk, messages=messages, convergence=True
            )
            break
        step, _ = solving_trust_region_subproblem(g, h, radius)
        candidate = xk + step
        the_function.set_variables(candidate)
        # Calculate the canonical_value of the function
        value_candidate = the_function.f()
        if value_candidate >= value_iterate:
            radius = np.linalg.norm(step) / 2.0
            continue

        num = value_iterate - value_candidate
        denominator = -np.inner(step, g) - 0.5 * np.inner(step, h @ step)
        rho = num / denominator
        if rho < eta1:
            # Failure: reduce the trust region
            radius = np.linalg.norm(step) / 2.0
            continue

        # Candidate accepted
        xk = candidate
        the_output: FunctionData = quadratic_model.get_f_g_h(xk)
        value_iterate = the_output.function
        g = the_output.gradient
        h = the_output.hessian
        if rho >= eta2:
            # Enlarge the trust region
            radius = min(2 * radius, max_delta)
        if radius <= min_delta:
            messages = the_function.messages
            messages['Cause of termination'] = f'Trust region is too small: {radius}'
            optimization_results = OptimizationResults(
                solution=xk, messages=messages, convergence=False
            )
            break
    else:
        messages = the_function.messages
        messages['Cause of termination'] = (
            f'Maximum number of iterations reached: {maxiter}'
        )
        optimization_results = OptimizationResults(
            solution=xk, messages=messages, convergence=False
        )

    return optimization_results


def _counter_value(the_function: FunctionToMinimize, name: str) -> int:
    """Read an evaluation counter from the objective protocol."""
    method = getattr(the_function, name, None)
    if not callable(method):
        return 0
    try:
        value = int(method())
    except (TypeError, ValueError):
        return 0
    return max(0, value)


def _restore_counter(the_function: FunctionToMinimize, name: str, value: int) -> None:
    """Restore counters for the bundled objective implementation when possible."""
    attribute = {
        "nbr_function_evaluations": "number_of_functions",
        "nbr_gradient_evaluations": "number_of_gradients",
        "nbr_hessian_evaluations": "number_of_hessians",
    }[name]
    try:
        setattr(the_function, attribute, int(value))
    except (AttributeError, TypeError):
        # Third-party objectives may expose read-only counters.  They remain
        # usable; their own protocol determines how counters are maintained.
        pass


def _objective_snapshot(the_function: FunctionToMinimize) -> object | None:
    hook = getattr(the_function, "snapshot_state", None)
    if not callable(hook):
        return None
    try:
        return hook()
    except Exception as error:  # pragma: no cover - objective-specific failure
        raise CheckpointError("objective snapshot_state() failed") from error


def _make_bfgs_state(
    *,
    the_function: FunctionToMinimize,
    model: BfgsModel,
    iterate: np.ndarray,
    value: float | None,
    gradient: np.ndarray | None,
    radius: float,
    iteration: int,
    accepted_iterations: int,
    algorithm_options: Mapping[str, object],
    convergence: bool = False,
    termination_reason: str | None = None,
    optimality_checked: bool | None = None,
) -> TrustRegionBFGSState:
    """Capture all optimizer state without evaluating the objective."""
    effective_value = model.last_function if value is None else value
    effective_gradient = model.last_gradient if gradient is None else gradient
    if effective_value is not None and effective_gradient is None:
        raise CheckpointError("cannot checkpoint a BFGS state without its gradient")
    try:
        return TrustRegionBFGSState(
            schema_version=1,
            algorithm=TRUST_REGION_BFGS_ALGORITHM,
            x=np.array(iterate, copy=True),
            objective=effective_value,
            gradient=(
                None
                if effective_gradient is None
                else np.array(effective_gradient, copy=True)
            ),
            hessian_approximation=np.array(model.hessian_approx, copy=True),
            last_iterate=(
                None
                if model.last_iterate is None
                else np.array(model.last_iterate, copy=True)
            ),
            last_gradient=(
                None
                if model.last_gradient is None
                else np.array(model.last_gradient, copy=True)
            ),
            trust_region_radius=radius,
            iteration=iteration,
            accepted_iterations=accepted_iterations,
            function_evaluations=_counter_value(
                the_function, "nbr_function_evaluations"
            ),
            gradient_evaluations=_counter_value(
                the_function, "nbr_gradient_evaluations"
            ),
            hessian_evaluations=_counter_value(
                the_function, "nbr_hessian_evaluations"
            ),
            algorithm_options=dict(algorithm_options),
            convergence=convergence,
            termination_reason=termination_reason,
            dtype=str(iterate.dtype),
            dimension=iterate.size,
            objective_state=_objective_snapshot(the_function),
            optimality_checked=(
                model.last_evaluation_was_optimal
                if optimality_checked is None
                else optimality_checked
            ),
            typical_parameter_scales=(
                None
                if getattr(the_function, "typx", None) is None
                else np.array(the_function.typx, copy=True)
            ),
            typical_objective_scale=getattr(the_function, "typf", None),
            relative_gradient_norm=getattr(
                the_function, "relative_gradient_norm", None
            ),
        )
    except CheckpointError:
        raise
    except (TypeError, ValueError, OptimizationError) as error:
        raise CheckpointError("cannot create a valid BFGS checkpoint") from error


def _emit_bfgs_checkpoint(
    state: TrustRegionBFGSState,
    checkpoint_callback: Callable[[TrustRegionBFGSState], object] | None,
) -> TrustRegionBFGSState:
    """Deliver an immutable state snapshot and return the durable state."""
    if checkpoint_callback is None:
        return state
    snapshot = state.immutable_copy()
    try:
        checkpoint_callback(snapshot)
    except Exception as error:
        if isinstance(error, CheckpointError):
            raise
        raise CheckpointError("trust-region BFGS checkpoint callback failed") from error
    return state


def _validate_bfgs_resume(
    state: TrustRegionBFGSState,
    starting_point: np.ndarray,
    algorithm_options: Mapping[str, object],
) -> None:
    """Validate a checkpoint against the current call before any evaluation."""
    if state.x.shape != starting_point.shape:
        raise ValueError(
            "trust-region BFGS checkpoint dimension does not match initial_values."
        )
    if state.x.dtype != starting_point.dtype:
        raise ValueError(
            "trust-region BFGS checkpoint dtype does not match initial_values: "
            f"{state.x.dtype} != {starting_point.dtype}."
        )
    if state.iteration == 0 and not np.array_equal(state.x, starting_point):
        raise ValueError(
            "initial_values do not match the parameter vector in the checkpoint."
        )
    expected = dict(algorithm_options)
    stored = dict(state.algorithm_options)
    if stored != expected:
        raise ValueError(
            "trust-region BFGS checkpoint algorithm options do not match the current call."
        )
    if (
        (state.objective is None or state.gradient is None)
        and not state.convergence
    ):
        raise ValueError(
            "a resumable trust-region BFGS checkpoint must contain objective and gradient."
        )
    if (state.last_iterate is None) != (state.last_gradient is None):
        raise ValueError(
            "checkpoint last_iterate and last_gradient must be present together."
        )


def _resumable_bfgs_trust_region(
    *,
    the_function: FunctionToMinimize,
    starting_point: np.ndarray,
    init_bfgs: np.ndarray | None,
    use_dogleg: bool,
    maxiter: int,
    initial_radius: float,
    eta1: float,
    eta2: float,
    solving_trust_region_subproblem: Callable[
        [np.ndarray, np.ndarray, float], tuple[np.ndarray, Diagnostic]
    ],
    state: TrustRegionBFGSState | Mapping[str, object] | None,
    checkpoint_callback: Callable[[TrustRegionBFGSState], object] | None,
    stop_requested: Callable[[], bool] | None,
    algorithm_options: Mapping[str, object],
) -> OptimizationResults:
    """Trust-region BFGS loop with stable, resumable checkpoint boundaries."""
    if state is not None and not isinstance(state, TrustRegionBFGSState):
        state = TrustRegionBFGSState.from_dict(state)

    if state is None:
        xk = np.array(starting_point, copy=True)
        model = BfgsModel(the_function, init_bfgs)
        the_output = model.get_f_g_h(xk)
        value_iterate = the_output.function
        g = the_output.gradient
        h = model.hessian_approx if the_output.hessian is None else the_output.hessian
        radius = initial_radius
        iteration = 0
        accepted_iterations = 0
    else:
        _validate_bfgs_resume(state, starting_point, algorithm_options)
        xk = np.array(state.x, copy=True)
        model = BfgsModel(the_function, np.array(state.hessian_approximation, copy=True))
        model.hessian_approx = np.array(state.hessian_approximation, copy=True)
        model.last_iterate = (
            None if state.last_iterate is None else np.array(state.last_iterate, copy=True)
        )
        model.last_gradient = (
            None if state.last_gradient is None else np.array(state.last_gradient, copy=True)
        )
        model.last_function = state.objective
        model.last_evaluation_was_optimal = state.optimality_checked
        the_function.set_variables(xk)
        restore_hook = getattr(the_function, "restore_state", None)
        if state.objective_state is not None and callable(restore_hook):
            try:
                restore_hook(state.objective_state)
            except Exception as error:  # pragma: no cover - objective-specific failure
                raise CheckpointError("objective restore_state() failed") from error
        _restore_counter(
            the_function, "nbr_function_evaluations", state.function_evaluations
        )
        _restore_counter(
            the_function, "nbr_gradient_evaluations", state.gradient_evaluations
        )
        _restore_counter(
            the_function, "nbr_hessian_evaluations", state.hessian_evaluations
        )
        for attribute in (
            "typx",
            "typf",
            "relative_gradient_norm",
        ):
            value = {
                "typx": state.typical_parameter_scales,
                "typf": state.typical_objective_scale,
                "relative_gradient_norm": state.relative_gradient_norm,
            }[attribute]
            if value is not None:
                try:
                    setattr(
                        the_function,
                        attribute,
                        np.array(value, copy=True) if attribute == "typx" else value,
                    )
                except (AttributeError, TypeError):
                    pass
        value_iterate = None if state.optimality_checked else state.objective
        g = None if state.gradient is None else np.array(state.gradient, copy=True)
        h = model.hessian_approx
        radius = state.trust_region_radius
        iteration = state.iteration
        accepted_iterations = state.accepted_iterations

    max_delta = np.finfo(float).max
    min_delta = np.finfo(float).eps
    current_state = _make_bfgs_state(
        the_function=the_function,
        model=model,
        iterate=xk,
        value=value_iterate,
        gradient=g,
        radius=radius,
        iteration=iteration,
        accepted_iterations=accepted_iterations,
        algorithm_options=algorithm_options,
    )
    _emit_bfgs_checkpoint(current_state, checkpoint_callback)

    def requested_stop() -> bool:
        return stop_requested is not None and bool(stop_requested())

    def finish(
        *,
        convergence: bool,
        reason: str,
        cause: str | None = None,
    ) -> OptimizationResults:
        nonlocal current_state
        current_state = _make_bfgs_state(
            the_function=the_function,
            model=model,
            iterate=xk,
            value=value_iterate,
            gradient=g,
            radius=radius,
            iteration=iteration,
            accepted_iterations=accepted_iterations,
            algorithm_options=algorithm_options,
            convergence=convergence,
            termination_reason=reason,
        )
        _emit_bfgs_checkpoint(current_state, checkpoint_callback)
        messages = the_function.messages
        if cause is not None:
            messages["Cause of termination"] = cause
        if convergence:
            messages["Number of iterations"] = iteration
        elif reason == "checkpoint_requested":
            messages["Termination reason"] = "checkpoint_requested"
            messages["Cause of termination"] = "Checkpoint requested"
        elif reason == "iteration_limit":
            messages["Cause of termination"] = (
                f"Maximum number of iterations reached: {maxiter}"
            )
        return OptimizationResults(
            solution=xk,
            messages=messages,
            convergence=convergence,
            state=current_state.copy(),
        )

    if requested_stop():
        return finish(convergence=False, reason="checkpoint_requested")

    for _ in range(maxiter):
        # The model's optimality check happened as part of the last f_g call.
        # Returning here therefore does not perform a duplicate evaluation.
        if value_iterate is None:
            cause = the_function.messages.get("Cause of termination")
            return finish(
                convergence=True,
                reason="converged",
                cause=None if cause is None else str(cause),
            )

        step, _ = solving_trust_region_subproblem(g, h, radius)
        candidate = xk + step
        the_function.set_variables(candidate)
        # A failure here leaves current_state as the last durable checkpoint.
        value_candidate = the_function.f()
        iteration += 1
        if value_candidate >= value_iterate:
            radius = np.linalg.norm(step) / 2.0
            current_state = _make_bfgs_state(
                the_function=the_function,
                model=model,
                iterate=xk,
                value=value_iterate,
                gradient=g,
                radius=radius,
                iteration=iteration,
                accepted_iterations=accepted_iterations,
                algorithm_options=algorithm_options,
            )
            _emit_bfgs_checkpoint(current_state, checkpoint_callback)
            if requested_stop():
                return finish(convergence=False, reason="checkpoint_requested")
            continue

        num = value_iterate - value_candidate
        denominator = -np.inner(step, g) - 0.5 * np.inner(step, h @ step)
        rho = num / denominator
        if rho < eta1:
            radius = np.linalg.norm(step) / 2.0
            current_state = _make_bfgs_state(
                the_function=the_function,
                model=model,
                iterate=xk,
                value=value_iterate,
                gradient=g,
                radius=radius,
                iteration=iteration,
                accepted_iterations=accepted_iterations,
                algorithm_options=algorithm_options,
            )
            _emit_bfgs_checkpoint(current_state, checkpoint_callback)
            if requested_stop():
                return finish(convergence=False, reason="checkpoint_requested")
            continue

        # Candidate accepted.  BfgsModel updates both the approximation and
        # its last-iterate/last-gradient pair in this call.
        xk = candidate
        accepted_iterations += 1
        the_output = model.get_f_g_h(xk)
        value_iterate = the_output.function
        g = the_output.gradient
        h = model.hessian_approx if the_output.hessian is None else the_output.hessian
        if rho >= eta2:
            radius = min(2 * radius, max_delta)

        current_state = _make_bfgs_state(
            the_function=the_function,
            model=model,
            iterate=xk,
            value=value_iterate,
            gradient=g,
            radius=radius,
            iteration=iteration,
            accepted_iterations=accepted_iterations,
            algorithm_options=algorithm_options,
        )
        _emit_bfgs_checkpoint(current_state, checkpoint_callback)
        if requested_stop():
            return finish(convergence=False, reason="checkpoint_requested")
        if radius <= min_delta:
            cause = f"Trust region is too small: {radius}"
            return finish(convergence=False, reason="trust_region_too_small", cause=cause)
        if value_iterate is None:
            cause = the_function.messages.get("Cause of termination")
            if _ == maxiter - 1:
                return finish(convergence=False, reason="iteration_limit")
            return finish(
                convergence=True,
                reason="converged",
                cause=None if cause is None else str(cause),
            )
    return finish(convergence=False, reason="iteration_limit")


def newton_trust_region(
    *,
    the_function: FunctionToMinimize,
    starting_point: np.ndarray,
    use_dogleg: bool = False,
    maxiter: int = 1000,
    initial_radius: float = 1.0,
    eta1: float = 0.01,
    eta2: float = 0.9,
):
    """Newton method with trust region

    :param the_function: object to calculate the objective function and its derivatives.
    :type the_function: optimization.functionToMinimize

    :param starting_point: starting point
    :type starting_point: numpy.array

    :param use_dogleg: True if the trust region sub-problem is solved using
        the dogleg method. False is it is solved with the truncated
        conjugate gradient algorithm.
    :type use_dogleg: bool

    :param maxiter: the algorithm stops if this number of iterations
                    is reached. Default: 1000.
    :type maxiter: int

    :param initial_radius: initial radius of the trust region. Default: 100.
    :type initial_radius: float

    :param eta1: threshold for failed iterations. Default: 0.01.
    :type eta1: float

    :param eta2: threshold for very successful iterations. Default 0.9.
    :type eta2: float

    :return: named tuple:

            - solution is the solution found,
            - message is a dictionary reporting various aspects
              related to the run of the algorithm.
            - convergence is a bool which is True if the algorithm has converged

    :rtype: OptimizationResults

    """
    the_model = NewtonModel(the_function)

    if use_dogleg:
        return minimization_with_trust_region(
            the_function,
            starting_point,
            the_model,
            dogleg,
            maxiter,
            initial_radius,
            eta1,
            eta2,
        )
    return minimization_with_trust_region(
        the_function,
        starting_point,
        the_model,
        truncated_conjugate_gradient,
        maxiter,
        initial_radius,
        eta1,
        eta2,
    )


def bfgs_trust_region(
    *,
    the_function: FunctionToMinimize,
    starting_point: np.ndarray,
    init_bfgs: np.ndarray | None = None,
    use_dogleg: bool = False,
    maxiter: int = 1000,
    initial_radius: float = 1.0,
    eta1: float = 0.01,
    eta2: float = 0.9,
    state: TrustRegionBFGSState | Mapping[str, object] | None = None,
    checkpoint_callback: Callable[[TrustRegionBFGSState], object] | None = None,
    stop_requested: Callable[[], bool] | None = None,
    _enable_resumption: bool = False,
    _algorithm_options: Mapping[str, object] | None = None,
):
    """BFGS method with trust region

    :param the_function: object to calculate the objective function and its derivatives.
    :type the_function: optimization.functionToMinimize

    :param starting_point: starting point
    :type starting_point: numpy.array

    :param init_bfgs: matrix used to initialize BFGS. If None, the
                     identity matrix is used. Default: None.
    :type init_bfgs: numpy.array

    :param use_dogleg: True if the trust region subproblem is solved using
        the dogleg method. False is it is solved with the truncated
        conjugate gradient algorithm.
    :type use_dogleg: bool

    :param maxiter: the algorithm stops if this number of iterations
                    is reached. Default: 1000.
    :type maxiter: int

    :param initial_radius: initial radius of the trust region. Default: 100.
    :type initial_radius: float

    :param eta1: threshold for failed iterations. Default: 0.01.
    :type eta1: float

    :param eta2: threshold for very successful iterations. Default 0.9.
    :type eta2: float

    :return: named tuple:

            - solution is the solution found,
            - message is a dictionary reporting various aspects
              related to the run of the algorithm.
            - convergence is a bool which is True if the algorithm has converged

    :rtype: OptimizationResults

    """
    if (
        state is not None
        or checkpoint_callback is not None
        or stop_requested is not None
        or _enable_resumption
    ):
        algorithm_options = dict(
            _algorithm_options
            if _algorithm_options is not None
            else {
                "dogleg": bool(use_dogleg),
                "eta1": float(eta1),
                "eta2": float(eta2),
                "initial_radius": float(initial_radius),
            }
        )
        solver = dogleg if use_dogleg else truncated_conjugate_gradient
        return _resumable_bfgs_trust_region(
            the_function=the_function,
            starting_point=starting_point,
            init_bfgs=init_bfgs,
            use_dogleg=use_dogleg,
            maxiter=maxiter,
            initial_radius=initial_radius,
            eta1=eta1,
            eta2=eta2,
            solving_trust_region_subproblem=solver,
            state=state,
            checkpoint_callback=checkpoint_callback,
            stop_requested=stop_requested,
            algorithm_options=algorithm_options,
        )

    the_model = BfgsModel(the_function, init_bfgs)

    if use_dogleg:
        return minimization_with_trust_region(
            the_function,
            starting_point,
            the_model,
            dogleg,
            maxiter,
            initial_radius,
            eta1,
            eta2,
        )
    return minimization_with_trust_region(
        the_function,
        starting_point,
        the_model,
        truncated_conjugate_gradient,
        maxiter,
        initial_radius,
        eta1,
        eta2,
    )
