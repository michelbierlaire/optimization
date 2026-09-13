"""Serializable state for the trust-region BFGS optimizer.

The state in this module is deliberately made up of ordinary values and NumPy
arrays.  In particular, it never stores an objective, a model, or a callback,
so it can be persisted as JSON or as an NPZ file without pickle.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from types import MappingProxyType

import numpy as np

from biogeme_optimization.exceptions import CheckpointError

TRUST_REGION_BFGS_STATE_SCHEMA_VERSION = 1
TRUST_REGION_BFGS_ALGORITHM = "biogeme_tr_bfgs"


def _json_clone(value: object, *, path: str = "value") -> object:
    """Copy a JSON-compatible value and reject executable/non-JSON values."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, Real) and not isinstance(value, bool):
        numeric = float(value)
        if not np.isfinite(numeric):
            raise ValueError(f"{path} must contain only finite numbers.")
        return numeric
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings.")
            result[key] = _json_clone(item, path=f"{path}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_json_clone(item, path=f"{path}[{index}]") for index, item in enumerate(value)]
    raise TypeError(
        f"{path} must be JSON-compatible; executable objects and NumPy arrays "
        "are not allowed in optimizer state."
    )


def _readonly_json(value: object) -> object:
    """Return an immutable copy of a JSON-compatible value."""
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _readonly_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_readonly_json(item) for item in value)
    return value


def _array_copy(value: object, name: str) -> np.ndarray:
    """Validate an in-memory numeric array and return an independent copy."""
    array = np.array(value, copy=True)
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must be a real numeric NumPy array.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _array_payload(value: np.ndarray) -> dict[str, object]:
    """Encode an array with enough metadata to validate it on restore."""
    return {
        "dtype": value.dtype.str,
        "shape": list(value.shape),
        "data": value.tolist(),
    }


def _array_from_payload(payload: object, name: str) -> np.ndarray:
    """Decode and validate an array payload."""
    if not isinstance(payload, Mapping):
        raise TypeError(f"{name} must be an object with dtype, shape, and data.")
    for key in ("dtype", "shape", "data"):
        if key not in payload:
            raise ValueError(f"{name} is missing {key!r}.")
    try:
        dtype = np.dtype(payload["dtype"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} has an invalid dtype.") from error
    if dtype.kind not in "iuf":
        raise TypeError(f"{name} must use a real numeric dtype.")
    shape_value = payload["shape"]
    if not isinstance(shape_value, (list, tuple)):
        raise TypeError(f"{name}.shape must be a list of non-negative integers.")
    shape: tuple[int, ...] = ()
    for dimension in shape_value:
        if isinstance(dimension, bool) or not isinstance(dimension, Integral):
            raise TypeError(f"{name}.shape must be a list of non-negative integers.")
        if int(dimension) < 0:
            raise ValueError(f"{name}.shape cannot contain negative dimensions.")
        shape += (int(dimension),)
    try:
        array = np.asarray(payload["data"], dtype=dtype)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name}.data cannot be decoded with dtype {dtype}.") from error
    if array.shape != shape:
        raise ValueError(
            f"{name} shape metadata {shape} does not match decoded shape {array.shape}."
        )
    return _array_copy(array, name)


def _counter(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a non-negative integer.")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return result


def _finite_float(name: str, value: object, *, strictly_positive: bool = False) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite number.")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a finite number.") from error
    if not np.isfinite(result) or (strictly_positive and result <= 0.0):
        qualifier = "strictly positive and " if strictly_positive else ""
        raise ValueError(f"{name} must be {qualifier}finite.")
    return result


def _validate_algorithm_options(options: Mapping[str, object]) -> dict[str, object]:
    cloned = _json_clone(options, path="algorithm_options")
    if not isinstance(cloned, dict):  # pragma: no cover - guarded by _json_clone
        raise TypeError("algorithm_options must be a mapping.")
    if "dogleg" in cloned and not isinstance(cloned["dogleg"], bool):
        raise TypeError("algorithm_options.dogleg must be a boolean.")
    if "use_dogleg" in cloned and not isinstance(cloned["use_dogleg"], bool):
        raise TypeError("algorithm_options.use_dogleg must be a boolean.")
    for name in ("eta1", "eta2"):
        if name in cloned:
            cloned[name] = _finite_float(
                f"algorithm_options.{name}", cloned[name]
            )
    if "eta1" in cloned and "eta2" in cloned and not (
        0.0 <= cloned["eta1"] < cloned["eta2"] <= 1.0
    ):
        raise ValueError("algorithm_options must satisfy 0 <= eta1 < eta2 <= 1.")
    for name in ("tolerance", "objective_tolerance"):
        if name in cloned:
            cloned[name] = _finite_float(
                f"algorithm_options.{name}", cloned[name]
            )
            if cloned[name] < 0.0:
                raise ValueError(f"algorithm_options.{name} must be non-negative.")
    for name in ("radius", "initial_radius"):
        if name in cloned:
            cloned[name] = _finite_float(
                f"algorithm_options.{name}", cloned[name], strictly_positive=True
            )
    if "model_fingerprint" in cloned and (
        not isinstance(cloned["model_fingerprint"], str)
        or not cloned["model_fingerprint"].strip()
    ):
        raise ValueError("algorithm_options.model_fingerprint must be non-empty.")
    return cloned


@dataclass(frozen=True, slots=True)
class TrustRegionBFGSState:
    """Complete numerical state at a stable trust-region BFGS boundary."""

    schema_version: int
    algorithm: str
    x: np.ndarray
    objective: float | None
    gradient: np.ndarray | None
    hessian_approximation: np.ndarray
    last_iterate: np.ndarray | None
    last_gradient: np.ndarray | None
    trust_region_radius: float
    iteration: int
    accepted_iterations: int
    function_evaluations: int
    gradient_evaluations: int
    hessian_evaluations: int
    algorithm_options: Mapping[str, object]
    convergence: bool = False
    termination_reason: str | None = None
    dtype: str | None = None
    dimension: int | None = None
    objective_state: object | None = None
    optimality_checked: bool = False
    typical_parameter_scales: np.ndarray | None = None
    typical_objective_scale: float | None = None
    relative_gradient_norm: float | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or self.schema_version != TRUST_REGION_BFGS_STATE_SCHEMA_VERSION
        ):
            raise ValueError(
                f"unsupported trust-region BFGS state schema version: "
                f"{self.schema_version}"
            )
        if self.algorithm != TRUST_REGION_BFGS_ALGORITHM:
            raise ValueError(f"unsupported optimizer algorithm: {self.algorithm!r}")

        x = _array_copy(self.x, "x")
        hessian = _array_copy(self.hessian_approximation, "hessian_approximation")
        if x.ndim != 1:
            raise ValueError("x must be one-dimensional.")
        dimension = x.size
        if hessian.shape != (dimension, dimension):
            raise ValueError(
                "hessian_approximation must have shape "
                f"({dimension}, {dimension}), got {hessian.shape}."
            )
        if self.gradient is not None:
            gradient = _array_copy(self.gradient, "gradient")
            if gradient.shape != (dimension,):
                raise ValueError("gradient must have one value per parameter.")
        else:
            gradient = None
        if self.last_iterate is not None:
            last_iterate = _array_copy(self.last_iterate, "last_iterate")
            if last_iterate.shape != (dimension,):
                raise ValueError("last_iterate must have one value per parameter.")
        else:
            last_iterate = None
        if self.last_gradient is not None:
            last_gradient = _array_copy(self.last_gradient, "last_gradient")
            if last_gradient.shape != (dimension,):
                raise ValueError("last_gradient must have one value per parameter.")
        else:
            last_gradient = None
        if self.objective is not None:
            objective = _finite_float("objective", self.objective)
        else:
            objective = None
        radius = _finite_float(
            "trust_region_radius", self.trust_region_radius, strictly_positive=True
        )
        iteration = _counter("iteration", self.iteration)
        accepted_iterations = _counter("accepted_iterations", self.accepted_iterations)
        function_evaluations = _counter("function_evaluations", self.function_evaluations)
        gradient_evaluations = _counter("gradient_evaluations", self.gradient_evaluations)
        hessian_evaluations = _counter("hessian_evaluations", self.hessian_evaluations)
        if accepted_iterations > iteration:
            raise ValueError("accepted_iterations cannot exceed iteration.")
        if not isinstance(self.convergence, bool):
            raise TypeError("convergence must be a boolean.")
        if not isinstance(self.optimality_checked, bool):
            raise TypeError("optimality_checked must be a boolean.")
        if self.termination_reason is not None and not isinstance(
            self.termination_reason, str
        ):
            raise TypeError("termination_reason must be a string or None.")
        options = _validate_algorithm_options(self.algorithm_options)

        dtype = str(np.dtype(x.dtype)) if self.dtype is None else str(self.dtype)
        try:
            requested_dtype = np.dtype(dtype)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid state dtype: {dtype!r}") from error
        if requested_dtype != x.dtype:
            raise ValueError(
                f"state dtype {requested_dtype} does not match x dtype {x.dtype}."
            )
        if self.dimension is not None:
            dimension_value = _counter("dimension", self.dimension)
            if dimension_value != dimension:
                raise ValueError("state dimension does not match x.")
        else:
            dimension_value = dimension
        objective_state = None
        if self.objective_state is not None:
            objective_state = _json_clone(self.objective_state, path="objective_state")
        if self.typical_parameter_scales is not None:
            typical_parameter_scales = _array_copy(
                self.typical_parameter_scales, "typical_parameter_scales"
            )
            if typical_parameter_scales.shape != (dimension,):
                raise ValueError(
                    "typical_parameter_scales must have one value per parameter."
                )
        else:
            typical_parameter_scales = None
        typical_objective_scale = (
            None
            if self.typical_objective_scale is None
            else _finite_float("typical_objective_scale", self.typical_objective_scale)
        )
        relative_gradient_norm = (
            None
            if self.relative_gradient_norm is None
            else _finite_float("relative_gradient_norm", self.relative_gradient_norm)
        )

        object.__setattr__(self, "x", x)
        object.__setattr__(self, "objective", objective)
        object.__setattr__(self, "gradient", gradient)
        object.__setattr__(self, "hessian_approximation", hessian)
        object.__setattr__(self, "last_iterate", last_iterate)
        object.__setattr__(self, "last_gradient", last_gradient)
        object.__setattr__(self, "trust_region_radius", radius)
        object.__setattr__(self, "iteration", iteration)
        object.__setattr__(self, "accepted_iterations", accepted_iterations)
        object.__setattr__(self, "function_evaluations", function_evaluations)
        object.__setattr__(self, "gradient_evaluations", gradient_evaluations)
        object.__setattr__(self, "hessian_evaluations", hessian_evaluations)
        object.__setattr__(self, "algorithm_options", options)
        object.__setattr__(self, "dtype", str(requested_dtype))
        object.__setattr__(self, "dimension", dimension_value)
        object.__setattr__(self, "objective_state", objective_state)
        object.__setattr__(self, "typical_parameter_scales", typical_parameter_scales)
        object.__setattr__(self, "typical_objective_scale", typical_objective_scale)
        object.__setattr__(self, "relative_gradient_norm", relative_gradient_norm)

    def copy(self) -> TrustRegionBFGSState:
        """Return an independent, mutable-array copy of this state."""
        return TrustRegionBFGSState(
            schema_version=self.schema_version,
            algorithm=self.algorithm,
            x=self.x,
            objective=self.objective,
            gradient=self.gradient,
            hessian_approximation=self.hessian_approximation,
            last_iterate=self.last_iterate,
            last_gradient=self.last_gradient,
            trust_region_radius=self.trust_region_radius,
            iteration=self.iteration,
            accepted_iterations=self.accepted_iterations,
            function_evaluations=self.function_evaluations,
            gradient_evaluations=self.gradient_evaluations,
            hessian_evaluations=self.hessian_evaluations,
            algorithm_options=self.algorithm_options,
            convergence=self.convergence,
            termination_reason=self.termination_reason,
            dtype=self.dtype,
            dimension=self.dimension,
            objective_state=self.objective_state,
            optimality_checked=self.optimality_checked,
            typical_parameter_scales=self.typical_parameter_scales,
            typical_objective_scale=self.typical_objective_scale,
            relative_gradient_norm=self.relative_gradient_norm,
        )

    def immutable_copy(self) -> TrustRegionBFGSState:
        """Return the deep immutable snapshot used by checkpoint callbacks."""
        snapshot = self.copy()
        for name in (
            "x",
            "gradient",
            "hessian_approximation",
            "last_iterate",
            "last_gradient",
            "typical_parameter_scales",
        ):
            value = getattr(snapshot, name)
            if value is not None:
                value.setflags(write=False)
        object.__setattr__(
            snapshot,
            "algorithm_options",
            _readonly_json(snapshot.algorithm_options),
        )
        object.__setattr__(
            snapshot, "objective_state", _readonly_json(snapshot.objective_state)
        )
        return snapshot

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible validated representation of the state."""
        return {
            "schema_version": self.schema_version,
            "algorithm": self.algorithm,
            "dtype": self.dtype,
            "dimension": self.dimension,
            "x": _array_payload(self.x),
            "objective": self.objective,
            "gradient": (
                None if self.gradient is None else _array_payload(self.gradient)
            ),
            "hessian_approximation": _array_payload(self.hessian_approximation),
            "last_iterate": (
                None if self.last_iterate is None else _array_payload(self.last_iterate)
            ),
            "last_gradient": (
                None if self.last_gradient is None else _array_payload(self.last_gradient)
            ),
            "trust_region_radius": self.trust_region_radius,
            "iteration": self.iteration,
            "accepted_iterations": self.accepted_iterations,
            "function_evaluations": self.function_evaluations,
            "gradient_evaluations": self.gradient_evaluations,
            "hessian_evaluations": self.hessian_evaluations,
            "algorithm_options": _json_clone(self.algorithm_options, path="algorithm_options"),
            "convergence": self.convergence,
            "termination_reason": self.termination_reason,
            "objective_state": _json_clone(self.objective_state, path="objective_state"),
            "optimality_checked": self.optimality_checked,
            "typical_parameter_scales": (
                None
                if self.typical_parameter_scales is None
                else _array_payload(self.typical_parameter_scales)
            ),
            "typical_objective_scale": self.typical_objective_scale,
            "relative_gradient_norm": self.relative_gradient_norm,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> TrustRegionBFGSState:
        """Validate and restore a state from :meth:`to_dict` output."""
        if not isinstance(payload, Mapping):
            raise TypeError("optimizer state payload must be a mapping.")
        schema_version = payload.get("schema_version")
        if (
            isinstance(schema_version, bool)
            or schema_version != TRUST_REGION_BFGS_STATE_SCHEMA_VERSION
        ):
            raise ValueError(f"unsupported trust-region BFGS state schema version: {schema_version}")
        required = (
            "algorithm",
            "x",
            "objective",
            "gradient",
            "hessian_approximation",
            "last_iterate",
            "last_gradient",
            "trust_region_radius",
            "iteration",
            "accepted_iterations",
            "function_evaluations",
            "gradient_evaluations",
            "hessian_evaluations",
            "algorithm_options",
        )
        missing = [name for name in required if name not in payload]
        if missing:
            raise ValueError(f"optimizer state is missing {', '.join(missing)}.")

        def optional_array(name: str) -> np.ndarray | None:
            value = payload[name]
            return None if value is None else _array_from_payload(value, name)

        x = _array_from_payload(payload["x"], "x")
        return cls(
            schema_version=int(schema_version),
            algorithm=str(payload["algorithm"]),
            x=x,
            objective=payload["objective"],
            gradient=optional_array("gradient"),
            hessian_approximation=_array_from_payload(
                payload["hessian_approximation"], "hessian_approximation"
            ),
            last_iterate=optional_array("last_iterate"),
            last_gradient=optional_array("last_gradient"),
            trust_region_radius=payload["trust_region_radius"],
            iteration=payload["iteration"],
            accepted_iterations=payload["accepted_iterations"],
            function_evaluations=payload["function_evaluations"],
            gradient_evaluations=payload["gradient_evaluations"],
            hessian_evaluations=payload["hessian_evaluations"],
            algorithm_options=payload["algorithm_options"],
            convergence=payload.get("convergence", False),
            termination_reason=payload.get("termination_reason"),
            dtype=payload.get("dtype"),
            dimension=payload.get("dimension"),
            objective_state=payload.get("objective_state"),
            optimality_checked=payload.get("optimality_checked", False),
            typical_parameter_scales=(
                None
                if payload.get("typical_parameter_scales") is None
                else _array_from_payload(
                    payload["typical_parameter_scales"], "typical_parameter_scales"
                )
            ),
            typical_objective_scale=payload.get("typical_objective_scale"),
            relative_gradient_norm=payload.get("relative_gradient_norm"),
        )

    def to_npz(self, path: str | Path) -> None:
        """Persist the state efficiently in a NumPy NPZ archive."""
        metadata = self.to_dict()
        array_names = (
            "x",
            "gradient",
            "hessian_approximation",
            "last_iterate",
            "last_gradient",
            "typical_parameter_scales",
        )
        arrays: dict[str, np.ndarray] = {}
        for name in array_names:
            value = getattr(self, name)
            metadata[name] = None if value is None else name
            if value is not None:
                arrays[name] = value
        np.savez_compressed(
            path,
            metadata=np.asarray(json.dumps(metadata, separators=(",", ":"))),
            **arrays,
        )

    @classmethod
    def from_npz(cls, path: str | Path) -> TrustRegionBFGSState:
        """Load and validate a state from an NPZ archive without pickle."""
        try:
            with np.load(path, allow_pickle=False) as archive:
                if "metadata" not in archive:
                    raise ValueError("NPZ state is missing metadata.")
                metadata = json.loads(str(archive["metadata"].item()))
                if not isinstance(metadata, dict):
                    raise TypeError("NPZ state metadata must be an object.")
                for name in (
                    "x",
                    "gradient",
                    "hessian_approximation",
                    "last_iterate",
                    "last_gradient",
                    "typical_parameter_scales",
                ):
                    reference = metadata.get(name)
                    if reference is None:
                        metadata[name] = None
                    else:
                        if reference != name or name not in archive:
                            raise ValueError(f"NPZ state is missing array {name!r}.")
                        metadata[name] = _array_payload(np.asarray(archive[name]))
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            if isinstance(error, ValueError) and str(error).startswith(
                ("unsupported trust-region", "x", "hessian")
            ):
                raise
            raise ValueError(f"cannot read valid trust-region BFGS NPZ state: {path}") from error
        return cls.from_dict(metadata)


__all__ = [
    "TRUST_REGION_BFGS_ALGORITHM",
    "TRUST_REGION_BFGS_STATE_SCHEMA_VERSION",
    "CheckpointError",
    "TrustRegionBFGSState",
]
