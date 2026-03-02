# AstroForge — Known Issues (archweave fork)

## 1. ITRSToTETED crashes: ITRSToTIRS missing @njit decorator

**Date**: 2025-03-02
**Severity**: High — blocks MaDDG GroundOpticalSensor.observe()
**Commit**: `23f2ad2` (main, latest)
**Upstream**: https://github.com/mit-ll/AstroForge

### Description

`ITRSToTETED` (line 554 in `_transformations.py`) is decorated with `@njit`, but it calls `ITRSToTIRS` (line 521) which is a plain Python function without `@njit`. Numba cannot call non-jitted functions from within a jitted function.

### Error

```
numba.core.errors.TypingError: Failed in nopython mode pipeline (step: nopython frontend)
Untyped global name 'ITRSToTIRS': Cannot determine Numba type of <class 'function'>

File "astroforge/coordinates/_transformations.py", line 587:
def ITRSToTETED(
    ...
    X2 = ITRSToTIRS(time, X)
    ^
```

### Root Cause

`ITRSToTIRS` (line 521) uses `isinstance()` checks and `raise TypeError`, which prevent it from being decorated with `@njit`. However, `ITRSToTETED` (decorated with `@njit`) calls it directly.

```python
# Line 521 — NOT @njit decorated
def ITRSToTIRS(mjd, X):
    xp, yp = polarmotion(mjd)
    if (isinstance(xp, float)) and (isinstance(yp, float)):  # incompatible with @njit
        Y = Ry(xp * pi / 180 / 3600) @ Rx(yp * pi / 180 / 3600) @ X
    else:
        raise TypeError(...)
    return Y

# Line 554 — @njit decorated
@njit
def ITRSToTETED(time, X, V=None):
    ...
    X2 = ITRSToTIRS(time, X)  # CRASH: calling non-jitted from jitted
```

### Impact

- **MaDDG `GroundOpticalSensor.observe()`** calls `ITRSToTETED` internally via `_site_loc_TETED()`, so all ground sensor observations crash.
- Any code using `afc.PosVelConversion(afc.ITRSToTETED, ...)` will also crash.
- `TETEDToITRS` (line 460) has the same pattern — it also calls `ITRSToTIRS` from within `@njit`.

### Workaround (won-sbss)

Our `compute_ground_visibility()` uses a self-contained GMST rotation (`_ecef_to_eci_gmst`) instead of AstroForge's `ITRSToTETED`, so our ground visibility pipeline is unaffected.

### Suggested Fix

Either:
1. Make `ITRSToTIRS` compatible with `@njit` (remove `isinstance` checks, use typed alternatives)
2. Remove `@njit` from `ITRSToTETED` and `TETEDToITRS`
3. Inline the polar motion rotation within the jitted functions

### Environment

- Python: 3.13.5
- NumPy: 2.3.4
- Numba: latest (via pip)
