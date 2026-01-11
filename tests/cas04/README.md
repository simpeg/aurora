# CAS04 Processing Test Suite

Comprehensive test suite for Aurora MT processing pipeline using CAS04 dataset.

## Quick Start

### Fast Tests (2 minutes)
```bash
# Skip slow integration tests
pytest tests/cas04/test_cas04_processing.py -m "not slow"
```

### Complete Suite (3.5 minutes)
```bash
# Run all tests including slow integration tests
pytest tests/cas04/test_cas04_processing.py
```

### EMTF Comparison Only (2.5 minutes)
```bash
pytest tests/cas04/test_cas04_processing.py::TestEMTFComparison -v
```

## Test Structure

### Test Classes
1. **TestConfigCreation** (4 tests) - Config generation from KernelDataset
2. **TestProcessingWorkflow** (4 tests) - Basic processing pipeline validation
3. **TestEMTFComparison** (10 tests) - Comparison with EMTF reference results
4. **TestDataQuality** (2 tests) - Error estimates and quality metrics
5. **TestEndToEndIntegration** (2 tests) - Complete pipeline integration
6. **TestEdgeCases** (2 tests) - Error handling and edge cases

### Parameterization
- Tests run for both MTH5 v0.1.0 and v0.2.0 formats
- Total: 38 tests (36 fast + 2 slow)

## Performance Optimizations

### Session-Scoped Fixtures
Expensive operations cached per test session:
- `session_cas04_tf_result` - Process MTH5 once (~40s per version)
- `session_interpolated_comparison` - Interpolate TF once for EMTF comparison
- `global_fdsn_miniseed_v010/v020` - Create MTH5 from test data once

### Slow Test Markers
The `test_complete_pipeline_from_run_summary` test is marked `@pytest.mark.slow` because it:
- Re-runs `process_mth5()` (duplicates work in session fixture)
- Adds ~40s per MTH5 version (80s total)
- Provides integration testing but not essential for quick validation

## Test Data

### Source
- **Package**: `mth5_test_data`
- **Files**: `cas04_stationxml.xml`, `cas_04_streams.mseed`
- **Location**: `mth5_test_data.get_test_data_path("miniseed")`

### EMTF Reference
- **File**: `emtf_results/CAS04-CAS04bcd_REV06-CAS04bcd_NVR08.zmm`
- **Periods**: 33 bands (4.65s - 29127s)
- **Purpose**: Validate Aurora results against EMTF processing

## Key Findings from EMTF Comparison

### Excellent Agreement (Zxy - Primary Mode)
- Magnitude correlation: 0.95 (log-log)
- Median ratio: 1.027 (within 3%)
- **Conclusion**: No systematic calibration errors

### Good Agreement (Zyx - Secondary Mode)
- Magnitude correlation: 0.44 (affected by 3D structure)
- Median ratio: 0.999 (nearly perfect)
- **Conclusion**: Some outliers at specific frequencies

### Expected Differences (Diagonal Components)
- Zxx, Zyy show poor correlation (< 0.3)
- **Conclusion**: Normal for small, noisy diagonal components in 1D/2D structures

See `EMTF_comparison_analysis.md` for detailed statistical analysis.

## Runtime Breakdown

### Fast Tests (126s)
- v010 session setup: ~62s (49%)
- v020 session setup: ~40s (32%)
- Individual tests: ~24s (19%)

### Complete Suite (227s)
- Fast tests: 126s (55%)
- Slow integration tests: 93s (41%)
- Teardown: 8s (4%)

### Optimization Results
- **Initial**: ~12 minutes (naive implementation)
- **With session fixtures**: ~3.5 minutes (70% faster)
- **Fast mode**: ~2 minutes (83% faster than initial)

## Usage Patterns

### During Development
```bash
# Quick validation
pytest tests/cas04/test_cas04_processing.py -m "not slow" -v

# Check specific component
pytest tests/cas04/test_cas04_processing.py::TestProcessingWorkflow -v
```

### Before Commit
```bash
# Run complete suite
pytest tests/cas04/test_cas04_processing.py -v
```

### CI/CD
```bash
# Fast mode for quick feedback
pytest tests/cas04/test_cas04_processing.py -m "not slow" --tb=short
```

### Debugging
```bash
# Show stdout/stderr
pytest tests/cas04/test_cas04_processing.py::TestEMTFComparison::test_impedance_components_correlation -v -s

# Show detailed timing
pytest tests/cas04/test_cas04_processing.py -v --durations=20
```

## Markers

- `slow` - Long-running integration tests (skip with `-m "not slow"`)

To add more markers, update `pytest.ini`:
```ini
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks integration tests
```
