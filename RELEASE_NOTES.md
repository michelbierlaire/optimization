# Release notes

## Unreleased

- Preserved the historical three-item tuple shape of `OptimizationResults` for
  Biogeme and older callers while exposing resumable state through
  `result.state`.
- Restored compatibility with Biogeme's optimizer integration, which unpacks
  optimization results as `(solution, messages, convergence)`.
- Added regression coverage for the Biogeme-style three-value unpacking and a
  Biogeme estimation smoke test using the state-enabled optimizer.

## 0.0.13 (2026-09-14)

- Added schema version 1 resumable state for Biogeme trust-region BFGS.
- `TrustRegionBFGSState.to_dict()` produces JSON-compatible array payloads with
  dtype and shape metadata; `to_npz()`/`from_npz()` store large arrays
  efficiently without pickle.
- Added `state=`, `checkpoint_callback=`, and `stop_requested=` to
  `bfgs_trust_region_for_biogeme`. Checkpoints are emitted after
  initialization, rejected and accepted trust-region boundaries, and before
  every termination return. Callbacks receive deep immutable snapshots.
- Existing calls without a state retain the legacy trust-region BFGS behavior,
  and objectives need only the objective/gradient protocol. The full
  `biogeme` package is not required.
