"""Tests for the standalone Biogeme-facing optimizer API."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import tomlkit

from biogeme_optimization.function import FunctionData, FunctionToMinimize
from biogeme_optimization.optimization import bfgs_trust_region_for_biogeme


class GradientOnlyQuadratic(FunctionToMinimize):
    """A quadratic objective that intentionally does not implement a Hessian."""

    def __init__(self) -> None:
        super().__init__()
        self.target = np.array([1.0, -1.0])

    def dimension(self) -> int:
        return self.target.size

    def _f(self) -> float:
        delta = self.x - self.target
        return float(0.5 * np.dot(delta, delta))

    def _f_g(self) -> FunctionData:
        delta = self.x - self.target
        return FunctionData(
            function=float(0.5 * np.dot(delta, delta)),
            gradient=delta.copy(),
            hessian=None,
        )


def _options(**overrides: object) -> dict[str, object]:
    result: dict[str, object] = {
        'maxiter': 100,
        'tolerance': 1.0e-10,
        'objective_tolerance': 1.0e-12,
    }
    result.update(overrides)
    return result


def test_imports_without_full_biogeme() -> None:
    source_root = Path(__file__).parents[1] / 'src'
    environment = os.environ.copy()
    environment['PYTHONPATH'] = str(source_root)
    result = subprocess.run(
        [
            sys.executable,
            '-c',
            (
                'import sys; '
                'from biogeme_optimization.function import FunctionToMinimize; '
                'from biogeme_optimization.optimization import '
                'bfgs_trust_region_for_biogeme; '
                "assert not any(name == 'biogeme' or name.startswith('biogeme.') "
                'for name in sys.modules)'
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode == 0


def test_metadata_has_no_full_biogeme_dependency() -> None:
    metadata = tomlkit.parse(Path('pyproject.toml').read_text())
    dependencies = metadata['project']['dependencies']
    assert not any(dependency.lower().startswith('biogeme') for dependency in dependencies)


def test_gradient_only_quadratic_returns_typed_result() -> None:
    objective = GradientOnlyQuadratic()
    result = bfgs_trust_region_for_biogeme(
        objective,
        np.array([3.0, -3.0]),
        [(None, None), (None, None)],
        ['x', 'y'],
        _options(),
    )

    np.testing.assert_allclose(result.solution, objective.target, atol=1.0e-8)
    assert result.convergence is True
    assert 'Cause of termination' in result.messages
    assert result.messages['Cause of termination'].startswith('Relative gradient')
    assert objective.nbr_function_evaluations() > 0
    assert objective.nbr_gradient_evaluations() > 0
    assert objective.nbr_hessian_evaluations() == 0


def test_result_preserves_biogeme_three_value_unpacking() -> None:
    result = bfgs_trust_region_for_biogeme(
        GradientOnlyQuadratic(),
        np.array([3.0, -3.0]),
        [(None, None), (None, None)],
        ['x', 'y'],
        _options(),
    )

    solution, messages, convergence = result
    np.testing.assert_array_equal(solution, result.solution)
    assert messages is result.messages
    assert convergence is result.convergence
    assert len(result) == 3
    assert result.state is not None


def test_batch_compatible_public_methods() -> None:
    objective = GradientOnlyQuadratic()
    objective.set_variables(np.array([2.0, -2.0]))
    assert np.isfinite(objective.f(batch='unused'))
    evaluation = objective.f_g(batch='unused')
    np.testing.assert_allclose(evaluation.gradient, [1.0, -1.0])


@pytest.mark.parametrize('missing', ['maxiter', 'tolerance', 'objective_tolerance'])
def test_required_options_are_checked(missing: str) -> None:
    options = _options()
    del options[missing]
    with pytest.raises(ValueError, match='missing required entries'):
        bfgs_trust_region_for_biogeme(
            GradientOnlyQuadratic(),
            np.zeros(2),
            [(None, None), (None, None)],
            ['x', 'y'],
            options,
        )


def test_bounds_and_variable_names_are_validated() -> None:
    with pytest.raises(ValueError, match='variable_names'):
        bfgs_trust_region_for_biogeme(
            GradientOnlyQuadratic(),
            np.zeros(2),
            [(None, None), (None, None)],
            ['x'],
            _options(),
        )

    with pytest.raises(ValueError, match='bounds'):
        bfgs_trust_region_for_biogeme(
            GradientOnlyQuadratic(),
            np.zeros(2),
            [(None, None)],
            ['x', 'y'],
            _options(),
        )


def test_legacy_defaults_remain_available() -> None:
    result = bfgs_trust_region_for_biogeme(
        GradientOnlyQuadratic(),
        np.zeros(2),
        [(None, None), (None, None)],
        ['x', 'y'],
        None,
    )
    assert result.convergence is True
