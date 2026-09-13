"""Biogeme-facing optimization entry points.

The functions in this module are the public adapter layer for callers that
need the optimizer package without the full :mod:`biogeme` distribution.
They deliberately depend only on modules in :mod:`biogeme_optimization`.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Mapping, Sequence
from numbers import Integral
from typing import Any

import numpy as np

from biogeme_optimization.diagnostics import OptimizationResults
from biogeme_optimization.floating_point import MACHINE_EPSILON
from biogeme_optimization.function import FunctionToMinimize
from biogeme_optimization.state import TrustRegionBFGSState
from biogeme_optimization.trust_region import bfgs_trust_region

logger = logging.getLogger(__name__)

OptimizationResult = OptimizationResults

_DEFAULT_OPTIONS: dict[str, object] = {
    'maxiter': 100,
    'tolerance': float(MACHINE_EPSILON**0.3333),
    'objective_tolerance': float(MACHINE_EPSILON**0.3333),
}
_REQUIRED_OPTIONS = ('maxiter', 'tolerance', 'objective_tolerance')


def _as_nonnegative_float(name: str, value: object) -> float:
    """Validate and return a non-negative finite floating-point option."""
    if isinstance(value, bool):
        raise TypeError(f'{name} must be a non-negative finite number.')
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f'{name} must be a non-negative finite number.') from error
    if not np.isfinite(numeric) or numeric < 0:
        raise ValueError(f'{name} must be a non-negative finite number.')
    return numeric


def _as_positive_integer(name: str, value: object) -> int:
    """Validate and return a positive integer option."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f'{name} must be a positive integer.')
    numeric = int(value)
    if numeric <= 0:
        raise ValueError(f'{name} must be a positive integer.')
    return numeric


def _validate_inputs(
    function_to_minimize: FunctionToMinimize,
    initial_values: object,
    bounds: Sequence[tuple[float | None, float | None]],
    variable_names: Sequence[str],
    *,
    preserve_dtype: bool = False,
) -> tuple[np.ndarray, list[tuple[float | None, float | None]], list[str]]:
    """Validate the common Biogeme optimizer inputs."""
    supplied = np.asarray(initial_values)
    if preserve_dtype and supplied.dtype.kind in 'f':
        initial = np.array(supplied, copy=True)
    else:
        initial = np.asarray(initial_values, dtype=float)
    if initial.ndim != 1:
        raise ValueError('initial_values must be a one-dimensional NumPy array.')
    if not np.all(np.isfinite(initial)):
        raise ValueError('initial_values must contain finite values.')

    function_to_minimize.set_variables(initial)
    dimension = function_to_minimize.dimension()
    if dimension != initial.size:
        raise ValueError(
            'initial_values and function_to_minimize have incompatible dimensions: '
            f'{initial.size} and {dimension}.'
        )

    names = list(variable_names)
    if len(names) != initial.size:
        raise ValueError('variable_names must have one name per parameter.')
    if not all(isinstance(name, str) for name in names):
        raise TypeError('variable_names must contain strings.')

    try:
        resolved_bounds = list(bounds)
    except TypeError as error:
        raise TypeError('bounds must be a sequence of lower/upper pairs.') from error
    if len(resolved_bounds) != initial.size:
        raise ValueError('bounds must have one pair per parameter.')

    for index, bound in enumerate(resolved_bounds):
        try:
            lower, upper = bound
        except (TypeError, ValueError) as error:
            raise ValueError(
                f'bounds[{index}] must contain exactly two values.'
            ) from error
        for label, value in (('lower', lower), ('upper', upper)):
            if value is None:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError) as error:
                raise TypeError(
                    f'bounds[{index}][{label}] must be a number or None.'
                ) from error
            if np.isnan(numeric):
                raise ValueError(f'bounds[{index}][{label}] cannot be NaN.')
        if lower is not None and upper is not None and float(lower) > float(upper):
            raise ValueError(f'bounds[{index}] has lower bound greater than upper bound.')

    return initial, resolved_bounds, names


def _resolve_options(options: Mapping[str, Any] | None) -> dict[str, Any]:
    """Resolve and validate options, retaining the historical ``None`` default."""
    if options is None:
        resolved: dict[str, Any] = dict(_DEFAULT_OPTIONS)
    else:
        if not isinstance(options, Mapping):
            raise TypeError('options must be a mapping.')
        missing = [name for name in _REQUIRED_OPTIONS if name not in options]
        if missing:
            missing_options = ', '.join(missing)
            raise ValueError(f'options is missing required entries: {missing_options}.')
        resolved = dict(options)

    resolved['maxiter'] = _as_positive_integer('maxiter', resolved['maxiter'])
    resolved['tolerance'] = _as_nonnegative_float('tolerance', resolved['tolerance'])
    resolved['objective_tolerance'] = _as_nonnegative_float(
        'objective_tolerance', resolved['objective_tolerance']
    )

    if 'radius' in resolved:
        resolved['radius'] = _as_nonnegative_float('radius', resolved['radius'])
        if resolved['radius'] == 0:
            raise ValueError('radius must be strictly positive.')
    if 'dogleg' in resolved and not isinstance(resolved['dogleg'], bool):
        raise TypeError('dogleg must be a boolean.')
    for name in ('eta1', 'eta2'):
        if name in resolved:
            resolved[name] = _as_nonnegative_float(name, resolved[name])
    eta1 = float(resolved.get('eta1', 0.01))
    eta2 = float(resolved.get('eta2', 0.9))
    if not eta1 < eta2 or eta2 > 1.0:
        raise ValueError('options must satisfy 0 <= eta1 < eta2 <= 1.')
    if 'model_fingerprint' in resolved and (
        not isinstance(resolved['model_fingerprint'], str)
        or not resolved['model_fingerprint'].strip()
    ):
        raise ValueError('model_fingerprint must be a non-empty string.')
    return resolved


def _initial_hessian_fingerprint(value: object) -> str | None:
    """Return a stable JSON-friendly identity for an optional initial Hessian."""
    if value is None:
        return None
    array = np.asarray(value)
    digest = hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
    return f'{array.dtype.str}:{array.shape}:{digest}'


def bfgs_trust_region_for_biogeme(
    function_to_minimize: FunctionToMinimize,
    initial_values: np.ndarray,
    bounds: Sequence[tuple[float | None, float | None]],
    variable_names: Sequence[str],
    options: Mapping[str, Any] | None = None,
    *,
    state: TrustRegionBFGSState | Mapping[str, object] | None = None,
    checkpoint_callback: Callable[[TrustRegionBFGSState], object] | None = None,
    stop_requested: Callable[[], bool] | None = None,
) -> OptimizationResults:
    """Minimize an objective using trust-region BFGS.

    This is the canonical optimizer-only replacement for
    ``biogeme.optimization.bfgs_trust_region_for_biogeme``. It accepts an
    objective implementing the public protocol from
    :class:`biogeme_optimization.function.FunctionToMinimize` and returns a
    typed :class:`~biogeme_optimization.diagnostics.OptimizationResults` with
    ``solution``, ``convergence``, and ``messages`` attributes.

    ``options`` must contain ``maxiter``, ``tolerance``, and
    ``objective_tolerance`` for new callers. ``None`` remains supported for
    compatibility with older callers and supplies the historical defaults.
    ``tolerance`` controls the objective's relative-gradient stopping
    tolerance when the objective exposes the standard ``epsilon`` attribute.
    ``objective_tolerance`` is accepted and validated for compatibility with
    Biogeme callers; the historical TR-BFGS algorithm uses the relative
    gradient as its convergence criterion.

    The underlying trust-region BFGS algorithm historically does not enforce
    simple bounds. Bounds are validated and retained in the API, and finite
    bounds produce the same warning as the legacy Biogeme entry point.
    ``variable_names`` is validated for API compatibility and reporting by
    higher-level callers.

    The BFGS path calls only ``set_variables``, ``dimension``, ``f``, ``f_g``,
    ``check_optimality``, and the evaluation-counter methods. An objective
    does not need to implement Hessian evaluation for this entry point.
    """
    resolved_options = _resolve_options(options)
    initial, resolved_bounds, _ = _validate_inputs(
        function_to_minimize,
        initial_values,
        bounds,
        variable_names,
        preserve_dtype=state is not None,
    )

    if any(lower is not None or upper is not None for lower, upper in resolved_bounds):
        logger.warning(
            'This trust-region BFGS algorithm does not enforce bound constraints. '
            'The bounds will be ignored.'
        )

    try:
        function_to_minimize.epsilon = resolved_options['tolerance']
    except (AttributeError, TypeError):
        # Duck-typed objectives may implement their own stopping rule.
        pass

    algorithm_options: dict[str, object] = {
        'dogleg': resolved_options.get('dogleg', False),
        'eta1': resolved_options.get('eta1', 0.01),
        'eta2': resolved_options.get('eta2', 0.9),
        'initial_radius': resolved_options.get('radius', 1.0),
        'tolerance': resolved_options['tolerance'],
        'objective_tolerance': resolved_options['objective_tolerance'],
        'init_bfgs_fingerprint': _initial_hessian_fingerprint(
            resolved_options.get('initBfgs')
        ),
    }
    if 'model_fingerprint' in resolved_options:
        algorithm_options['model_fingerprint'] = resolved_options['model_fingerprint']

    return bfgs_trust_region(
        the_function=function_to_minimize,
        starting_point=initial,
        init_bfgs=resolved_options.get('initBfgs'),
        use_dogleg=resolved_options.get('dogleg', False),
        maxiter=resolved_options['maxiter'],
        initial_radius=resolved_options.get('radius', 1.0),
        eta1=resolved_options.get('eta1', 0.01),
        eta2=resolved_options.get('eta2', 0.9),
        state=state,
        checkpoint_callback=checkpoint_callback,
        stop_requested=stop_requested,
        _enable_resumption=True,
        _algorithm_options=algorithm_options,
    )


__all__ = [
    'OptimizationResult',
    'OptimizationResults',
    'TrustRegionBFGSState',
    'bfgs_trust_region_for_biogeme',
]
