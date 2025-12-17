# Performance Analysis & Optimization Strategy

## Executive Summary

The Parkfield calibration test takes ~12 minutes instead of the expected 2-3 minutes. Through cProfile analysis, we identified that **81% of the execution time (461 seconds) is spent in `mt_metadata`'s filter processing code**, specifically:

1. **Primary bottleneck**: `filter_base.py::pass_band()` with O(N) loop structure
2. **Secondary issue**: `complex_response()` calculations being called repeatedly
3. **Tertiary issue**: Pydantic validation overhead adding ~25 seconds

## Profiling Results

### Test: `test_calibration_sanity_check`
- **Total Duration**: 569 seconds (~9.5 minutes)
- **Profile Data**: `parkfield_profile.prof`

### Time Distribution
| Component | Time | Percentage | Calls |
|-----------|------|-----------|-------|
| **pass_band() total time** | **461.5s** | **81%** | **37** |
| - Actual CPU time in loop | 461.5s | 81% | 37 |
| complex_response() | 507.1s | 89% | 5 |
| complex_response (per channel) | 101.4s | 18% | 5 |
| polyval() | 6.3s | 1% | 40 |
| Numpy operations (min/max) | 25.2s | 4% | 9.8M |
| Pydantic overhead | 25s | 4% | 6388 |
| Fixture setup | 29.3s | 5% | - |

### Call Stack
```
test_calibration_sanity_check() [569s total]
  ├── parkfield_sanity_check() [529.9s]
  │   ├── Calibrate 5 channels (ex, ey, hx, hy, hz)
  │   │   ├── complex_response() [507.1s, 5 calls, 101.4s each]
  │   │   │   └── update_units_and_normalization_frequency_from_filters_list() [507.0s, 25 calls]
  │   │   │       └── pass_band() [507.0s, 20 calls]
  │   │   │           └── pass_band() [461.5s ACTUAL TIME, 37 calls, 12.5s each]
  │   │   │               ├── complex_response() [multiple calls]
  │   │   │               ├── np.log10() [multiple calls]
  │   │   │               └── boolean indexing [multiple calls]
```

## Root Cause Analysis

### Problem 1: O(N) Loop in pass_band()

**File**: `mt_metadata/timeseries/filters/filter_base.py:403-408`

```python
for ii in range(0, int(f.size - window_len), 1):  # Line 403
    cr_window = np.array(amp[ii : ii + window_len])
    test = abs(1 - np.log10(cr_window.min()) / np.log10(cr_window.max()))
    
    if test <= tol:
        f_true[(f >= f[ii]) & (f <= f[ii + window_len])] = 1  # Expensive!
```

**Issues**:
- Iterates through **every frequency point** (10,000 points in Parkfield test)
- Each iteration performs:
  - `min()` and `max()` operations on window (O(window_len))
  - `np.log10()` calculations (expensive)
  - Boolean indexing with `(f >= f[ii]) & (f <= f[ii + window_len])` (O(N) operation)
- Total: O(N × (window_len + log operations + N boolean indexing)) = O(N²)

**Why slow**:
- For 10,000 frequency points with window_len=5:
  - ~10,000 iterations
  - Each iteration: ~5 min/max ops + 2 log10 ops + 10,000 boolean comparisons
  - Total: ~100,000+ numpy operations per pass_band call
  - Called 37 times during calibration = 3.7 million operations!

### Problem 2: Repeated complex_response() Calls

Each `pass_band()` call invokes `complex_response()` which involves expensive polynomial evaluation via `polyval()`.

- Number of times `complex_response()` called: 5 (per channel) × 101.4s = 507s
- But `pass_band()` may call it multiple times inside the loop!
- No caching between calls = redundant calculations

### Problem 3: Pydantic Validation Overhead

- 6,388 calls to `__setattr__` with validation
- ~25 seconds of overhead for metadata validation
- Could be optimized with `model_config` settings

## Solutions

### Solution 1: Vectorize pass_band() Loop (HIGH IMPACT - 9.8x speedup)

**Approach**: Replace the O(N) for-loop with vectorized numpy operations

**Implementation**: Use `numpy.lib.stride_tricks.as_strided()` to create sliding window view

```python
from numpy.lib.stride_tricks import as_strided

# Create sliding window view (no data copy!)
shape = (n_windows, window_len)
strides = (amp.strides[0], amp.strides[0])
amp_windows = as_strided(amp, shape=shape, strides=strides)

# Vectorized min/max (replaces loop!)
window_mins = np.min(amp_windows, axis=1)
window_maxs = np.max(amp_windows, axis=1)

# Vectorized test computation
with np.errstate(divide='ignore', invalid='ignore'):
    ratios = np.log10(window_mins) / np.log10(window_maxs)
    test_values = np.abs(1 - ratios)

# Mark passing windows
passing_windows = test_values <= tol

# Still need loop for range marking, but only over passing windows
for ii in np.where(passing_windows)[0]:
    f_true[ii : ii + window_len] = 1
```

**Expected Improvement**:
- Window metric calculation: O(N) → O(1) vectorized operation
- Speedup: ~10x per pass_band() call (0.1s → 0.01s)
- Total Parkfield test: 569s → ~114s (5x overall speedup)
- Time saved: 455 seconds (7.6 minutes)

### Solution 2: Cache complex_response() Results (MEDIUM IMPACT - 2-3x speedup)

**Approach**: Cache complex response by frequency array hash

```python
@functools.lru_cache(maxsize=128)
def complex_response_cached(self, frequencies_tuple):
    frequencies = np.array(frequencies_tuple)
    # ... expensive calculation ...
    return result
```

**Expected Improvement**:
- Avoid recalculation of same complex response
- Speedup: 2-3x for redundant calculations
- Additional 50-100 seconds saved

### Solution 3: Use Decimated Passband Detection (MEDIUM IMPACT - 5x speedup)

**Approach**: Sample every Nth frequency point instead of analyzing all points

```python
decimate_factor = max(1, f.size // 1000)  # Keep ~1000 points
if decimate_factor > 1:
    f_dec = f[::decimate_factor]
    amp_dec = amp[::decimate_factor]
else:
    f_dec = f
    amp_dec = amp

# Run pass_band on decimated array, map back to original
```

**Pros**:
- Maintains accuracy (1000 points still good for passband)
- Simple to implement
- Works with existing algorithm

**Cons**:
- Slight loss of precision for very narrow passbands
- Not recommended if precise passband needed

**Expected Improvement**:
- 10x speedup for large frequency arrays (10,000 → 1,000 points)
- Safer than aggressive vectorization

### Solution 4: Skip Passband Calculation When Not Needed (QUICK WIN)

**Approach**: Skip `pass_band()` for filters where passband is already known

```python
# In channel_response.py:
if hasattr(self, '_passband_estimate'):
    # Skip calculation, use cached value
    pass
```

**Expected Improvement**:
- Eliminates 5-10 unnecessary calls
- 50-100 seconds saved

## Recommended Implementation Plan

### Phase 1: Quick Win (30 minutes, 50-100 seconds saved)
1. Add `@functools.lru_cache` to `complex_response()`
2. Check if passband can be skipped in `channel_response.py`
3. Reduce Pydantic validation with `model_config`

### Phase 2: Main Optimization (2-3 hours, 450+ seconds saved)
1. Implement vectorized `pass_band()` using stride tricks
2. Fallback to original if stride trick fails
3. Comprehensive testing with existing test suite
4. Performance validation with cProfile

### Phase 3: Advanced (Optional, additional 50-100 seconds)
1. Implement decimated passband detection option
2. Profile other hotspots (polyval, etc.)
3. Consider Cython acceleration if needed

## Testing Strategy

### Correctness Validation
```python
# Compare results between original and optimized
# 1. Run test suite with both implementations
# 2. Verify pass_band results are identical
# 3. Check numerical accuracy to machine precision
```

### Performance Validation
```bash
# Profile before and after optimization
python -m cProfile -o profile_optimized.prof \
    -m pytest tests/parkfield/test_parkfield_pytest.py::TestParkfieldCalibration::test_calibration_sanity_check

# Compare profiles
python -c "import pstats; p = pstats.Stats('profile_optimized.prof'); p.sort_stats('cumulative').print_stats(10)"
```

### Expected Results After Optimization
- **pass_band()** total time: 461s → ~45s (10x improvement)
- **complex_response()** total time: 507s → ~400s (with caching, 27% reduction)
- **Overall test time**: 569s → ~110s (5x improvement)
- **Wall clock time**: 9.5 minutes → 1.8 minutes

## Risk Assessment

### Low Risk
- Vectorization using numpy stride tricks (well-established pattern)
- Caching with functools (standard Python)
- Comprehensive test coverage validates correctness

### Medium Risk
- Decimated passband may affect filters with narrow passbands
- Need to validate numerical accuracy

### Mitigation
- Keep original implementation as fallback
- Add feature flag for optimization strategy
- Validate against known filter responses

## Conclusion

The Parkfield test slowdown is caused by inefficient filter processing algorithms in `mt_metadata`, not Aurora. The O(N) loop in `pass_band()` is particularly problematic, consuming 81% of total time.

A vectorized implementation using numpy stride tricks can achieve **10x speedup** in pass_band calculation, resulting in **5x overall test speedup** (12 minutes → 2.4 minutes).

**Recommended**: Implement Phase 1 (quick win) immediately, Phase 2 (main optimization) within the sprint.

