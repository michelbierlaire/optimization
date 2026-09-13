"""Regression tests for resumable trust-region BFGS state."""

from __future__ import annotations

import json

import numpy as np
import pytest

from biogeme_optimization.function import FunctionData, FunctionToMinimize
from biogeme_optimization.optimization import (
    TrustRegionBFGSState,
    bfgs_trust_region_for_biogeme,
)
from biogeme_optimization.state import CheckpointError


class Quadratic(FunctionToMinimize):
    """Deterministic gradient-only quadratic with deliberately rejected steps."""

    def __init__(self) -> None:
        super().__init__(epsilon=1.0e-12)
        self.target = np.array([1.0, -2.0, 0.5])
        self.matrix = np.array(
            [[4.0, 1.0, 0.0], [1.0, 3.0, 0.2], [0.0, 0.2, 2.0]]
        )

    def dimension(self) -> int:
        return self.target.size

    def _f(self) -> float:
        difference = self.x - self.target
        return float(0.5 * difference @ self.matrix @ difference)

    def _f_g(self) -> FunctionData:
        difference = self.x - self.target
        return FunctionData(
            function=float(0.5 * difference @ self.matrix @ difference),
            gradient=self.matrix @ difference,
            hessian=None,
        )


def _options(maxiter: int = 100) -> dict[str, object]:
    return {
        "maxiter": maxiter,
        "tolerance": 1.0e-12,
        "objective_tolerance": 1.0e-12,
    }


def _run(objective: FunctionToMinimize, **kwargs: object):
    return bfgs_trust_region_for_biogeme(
        objective,
        np.array([5.0, 4.0, -3.0]),
        [(None, None)] * 3,
        ["x", "y", "z"],
        _options(),
        **kwargs,
    )


def test_interrupted_resume_is_identical_and_persists_rejected_radius() -> None:
    uninterrupted = _run(Quadratic())
    checkpoints = []

    def callback(state: TrustRegionBFGSState) -> None:
        checkpoints.append(state)

    interrupted = _run(
        Quadratic(),
        checkpoint_callback=callback,
        stop_requested=lambda: bool(checkpoints and checkpoints[-1].iteration >= 6),
    )
    assert interrupted.convergence is False
    assert interrupted.state is not None
    assert interrupted.state.accepted_iterations < interrupted.state.iteration
    assert interrupted.state.trust_region_radius < 1.0
    resumed = _run(Quadratic(), state=interrupted.state)

    np.testing.assert_array_equal(uninterrupted.solution, resumed.solution)
    np.testing.assert_array_equal(
        uninterrupted.state.hessian_approximation,
        resumed.state.hessian_approximation,
    )
    assert uninterrupted.convergence == resumed.convergence
    assert uninterrupted.messages["Cause of termination"] == resumed.messages[
        "Cause of termination"
    ]
    assert uninterrupted.state.to_dict() == resumed.state.to_dict()


def test_state_json_and_npz_round_trip(tmp_path) -> None:
    result = _run(Quadratic(), checkpoint_callback=lambda state: None)
    state = result.state
    assert state is not None
    restored = TrustRegionBFGSState.from_dict(json.loads(json.dumps(state.to_dict())))
    np.testing.assert_array_equal(restored.x, state.x)
    np.testing.assert_array_equal(
        restored.hessian_approximation, state.hessian_approximation
    )

    archive = tmp_path / "optimizer-state.npz"
    state.to_npz(archive)
    restored_npz = TrustRegionBFGSState.from_npz(archive)
    assert restored_npz.to_dict() == restored.to_dict()


def test_invalid_state_is_rejected() -> None:
    result = _run(Quadratic())
    payload = result.state.to_dict()

    bad_shape = json.loads(json.dumps(payload))
    bad_shape["gradient"]["shape"] = [2]
    with pytest.raises(ValueError, match="shape"):
        TrustRegionBFGSState.from_dict(bad_shape)

    nonfinite = json.loads(json.dumps(payload))
    nonfinite["x"]["data"][0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        TrustRegionBFGSState.from_dict(nonfinite)

    unsupported = json.loads(json.dumps(payload))
    unsupported["schema_version"] = 999
    with pytest.raises(ValueError, match="unsupported"):
        TrustRegionBFGSState.from_dict(unsupported)


def test_resume_does_not_evaluate_restored_point_and_options_are_validated() -> None:
    result = _run(Quadratic(), checkpoint_callback=lambda state: None)
    state = result.state
    assert state is not None
    objective = Quadratic()
    first_resume_snapshot = []

    def callback(snapshot: TrustRegionBFGSState) -> None:
        first_resume_snapshot.append(snapshot)

    resumed = _run(
        objective,
        state=state,
        checkpoint_callback=callback,
    )
    assert first_resume_snapshot[0].function_evaluations == state.function_evaluations
    assert first_resume_snapshot[0].gradient_evaluations == state.gradient_evaluations
    assert resumed.convergence is True

    incompatible = _options()
    incompatible["dogleg"] = True
    with pytest.raises(ValueError, match="algorithm options"):
        bfgs_trust_region_for_biogeme(
            Quadratic(),
            np.array([5.0, 4.0, -3.0]),
            [(None, None)] * 3,
            ["x", "y", "z"],
            incompatible,
            state=state,
        )


def test_checkpoint_callback_failure_is_not_silent() -> None:
    with pytest.raises(CheckpointError, match="callback"):
        _run(Quadratic(), checkpoint_callback=lambda state: (_ for _ in ()).throw(OSError()))
