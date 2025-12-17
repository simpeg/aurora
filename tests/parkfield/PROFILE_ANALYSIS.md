# Parkfield Test Profiling Report

## Summary
- **Total Test Time**: 569 seconds (~9.5 minutes)
- **Test**: `test_calibration_sanity_check`
- **Profile Date**: December 16, 2025

## Root Cause of Slowdown

### Primary Bottleneck: Filter Pass Band Calculation
**Location**: `mt_metadata/timeseries/filters/filter_base.py:355(pass_band)`
- **Time Spent**: 461 seconds (81% of total test time!)
- **Number of Calls**: 37
- **Average Time Per Call**: 12.5 seconds

### Secondary Issue: Complex Response Calculation
**Location**: `mt_metadata/timeseries/filters/channel_response.py:245(pass_band)`
- **Time Spent**: 507 seconds (89% of total test time)
- **Number of Calls**: 20
- **Caller**: `update_units_and_normalization_frequency_from_filters_list`

### Problem Description

The `pass_band()` method in `filter_base.py` has an inefficient algorithm:

```python
for ii in range(0, int(f.size - window_len), 1):  # Line 403
    cr_window = np.array(amp[ii : ii + window_len])
    test = abs(1 - np.log10(cr_window.min()) / np.log10(cr_window.max()))
    if test <= tol:
        f_true[(f >= f[ii]) & (f <= f[ii + window_len])] = 1
```

**Issues:**
1. **Iterates through every frequency point** - For a typical frequency array with thousands of points, this creates a massive loop
2. **Repeatedly calls numpy operations** - min(), max(), log10() are called thousands of times
3. **Inefficient boolean indexing** - Creates new boolean arrays in each iteration
4. **Called 37 times per test** - This is a critical path function called for each channel during calibration

## Why Original Unittest Was Faster

The original unittest likely used:
1. Pre-computed filter responses (cached)
2. Simpler filter configurations
3. Fewer frequency points
4. Different test data or mock objects

## Recommendations

### Option 1: Vectorize the pass_band Algorithm
Replace the loop with vectorized numpy operations to eliminate the nested iterations.

### Option 2: Cache Filter Response Calculations
- Cache complex_response() calls by frequency array
- Reuse cached responses across multiple pass_band() calls

### Option 3: Reduce Test Data
- Use fewer frequency points in calibration tests
- Use simpler filter configurations for testing

### Option 4: Skip Complex Filter Analysis
- Mock or skip pass_band() calculation in tests
- Use pre-computed pass bands for test filters

## Detailed Call Stack

```
parkfield_sanity_check (529.9s)
  └── calibrating channels (5 channels)
      └── complex_response() (507.0s)
          └── update_units_and_normalization_frequency_from_filters_list() (507.0s)
              └── pass_band() [20 calls] (507.0s)
                  └── pass_band() [37 calls, 461.4s actual time]
                      └── complex_response() [multiple calls per window]
                          └── polyval() [40 calls, 6.3s]
```

## Supporting Statistics

| Function | Total Time | Calls | Avg Time/Call |
|----------|-----------|-------|---------------|
| pass_band (base) | 461.5s | 37 | 12.5s |
| polyval | 6.3s | 40 | 0.16s |
| numpy.ufunc.reduce | 25.2s | 9.8M | 0.000s |
| min() calls | 13.9s | 4.9M | 0.000s |
| max() calls | 11.4s | 4.9M | 0.000s |

## Next Steps

1. Profile the original unittest with the same tool to compare bottlenecks
2. Identify which filters trigger expensive pass_band calculations
3. Implement vectorized version of pass_band or add caching
4. Re-run test to measure improvement
