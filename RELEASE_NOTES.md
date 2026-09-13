# Release notes

## Unreleased

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
