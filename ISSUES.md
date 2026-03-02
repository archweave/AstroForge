# AstroForge — Known Issues (archweave fork)

## 1. `ITRSToTIRS` / `TIRSToITRS` — `isinstance` incompatible with Numba

**Date**: 2026-03-02
**Severity**: High — blocks any `@njit` caller that needs polar motion
**Status**: **FIXED** in archweave fork

### Description

`ITRSToTIRS` and `TIRSToITRS` use `isinstance()` checks and `raise TypeError`, which are incompatible with Numba's `@njit` mode. Any jitted function calling them will fail.

### Fix Applied

Split each into a `@njit` inner function + non-jitted wrapper:

```python
@njit
def _ITRSToTIRS_inner(xp: float, yp: float, X):
    return Ry(xp * pi / 180 / 3600) @ Rx(yp * pi / 180 / 3600) @ X

def ITRSToTIRS(mjd, X):
    xp, yp = polarmotion(mjd)
    return _ITRSToTIRS_inner(float(xp), float(yp), X)
```

Same pattern for `_TIRSToITRS_inner` / `TIRSToITRS`.

### Additional changes

- `ITRSToMEMED`: restored `@njit` decorator (safe — uses ERA path, no polar motion call).
  This ensures `F_geo_MEMED` (@njit) → `ITRSToMEMED` chain works correctly.
- `GCRSToITRS`, `ITRSToGCRS`, `ITRSToTETED`: correctly left without `@njit`
  (they call polar motion wrappers and also hit Bug #3 via `CIRSToTETED`).

---

## 2. `propagator()` returns list instead of ndarray (numpy 2.x)

**Date**: 2026-03-02
**Severity**: Medium — blocks MaDDG `ContinuousThrustSatellite.propagate()`
**Status**: **FIXED** in archweave fork

### Description

`astroforge/propagators/_propagator.py` line 56: `return out.y.T` assumes `solve_ivp` returns `out.y` as a numpy array. Under numpy 2.x, `out.y` can be a Python list, causing `AttributeError: 'list' object has no attribute 'T'`.

### Fix Applied

```python
return np.asarray(out.y).T  # Convert list to array before transposing
```

---

## 3. `CIRSToTETED` (@njit) calls non-jittable `nutate` — PRE-EXISTING

**Date**: 2026-03-02
**Severity**: High — blocks `ITRSToTETED`, `ITRSToGCRS`, `GCRSToITRS`, `TETEDToITRS`
**Status**: OPEN (pre-existing in upstream MIT-LL code)

### Description

`CIRSToTETED` (line ~184) is decorated with `@njit` but calls `nutate()` from `_utilities.py`, which is not Numba-compatible. This causes a `TypingError` at Numba JIT compilation time.

### Error

```
numba.core.errors.TypingError: Failed in nopython mode pipeline (step: nopython frontend)
Untyped global name 'nutate': Cannot determine Numba type of <class 'function'>

File "_transformations.py", line 208:
def CIRSToTETED(
    ...
    (dpsi, deps, TrueObliquity, _) = nutate(time)
    ^
```

### Impact

All functions that call `CIRSToTETED` fail:
- `ITRSToTETED` — used by MaDDG `GroundOpticalSensor.observe()` (via `_site_loc_TETED`)
- `ITRSToGCRS` / `GCRSToITRS` — GCRS frame conversions
- `TETEDToITRS` — reverse TETED conversion

Functions using the MEMED path (`ITRSToMEMED` → `CIRSToMEMED`) are **unaffected** since `CIRSToMEMED` does not call `nutate`.

### Workaround (won-sbss)

Our `compute_ground_visibility()` uses a self-contained GMST rotation (`_ecef_to_eci_gmst`) instead of AstroForge's `ITRSToTETED`.

### Suggested Fix

Either make `nutate()` Numba-compatible, or remove `@njit` from `CIRSToTETED` (and propagate that change to all callers in the `@njit` chain).

### Environment

- Python: 3.13.5
- NumPy: 2.3.4
- Numba: 0.61.x
