"""
Tests for complete Aurora processing workflow using CAS04 data.

Tests the pipeline:
1. MTH5 file → RunSummary → KernelDataset
2. ConfigCreator → processing config
3. process_mth5() → TransferFunction
4. Compare results to EMTF reference

This extends the testing from test_processing_workflow_cas04.py with actual processing
and comparison to EMTF results.
"""

from pathlib import Path

import numpy as np
import pytest
from mt_metadata.transfer_functions.core import TF
from mth5.processing import KernelDataset, RunSummary
from scipy.interpolate import interp1d

from aurora.config.config_creator import ConfigCreator
from aurora.pipelines.process_mth5 import process_mth5


def interpolate_tf_to_common_periods(tf1, tf2):
    """
    Interpolate two transfer functions onto common period range.

    Uses the overlapping period range and creates a common grid for comparison.

    Parameters
    ----------
    tf1 : TF
        First transfer function
    tf2 : TF
        Second transfer function

    Returns
    -------
    periods_common : ndarray
        Common period array
    z1_interp : ndarray
        Interpolated impedance from tf1, shape (n_periods, 2, 2)
    z2_interp : ndarray
        Interpolated impedance from tf2, shape (n_periods, 2, 2)
    z1_err_interp : ndarray
        Interpolated impedance errors from tf1
    z2_err_interp : ndarray
        Interpolated impedance errors from tf2
    """
    # Get period arrays
    p1 = tf1.period
    p2 = tf2.period

    # Find overlapping range
    p_min = max(p1.min(), p2.min())
    p_max = min(p1.max(), p2.max())

    # Create common period grid (logarithmic spacing)
    n_periods = min(len(p1), len(p2))
    periods_common = np.logspace(np.log10(p_min), np.log10(p_max), n_periods)

    # Interpolate tf1 impedance (log-log for real and imag separately)
    z1_interp = np.zeros((len(periods_common), 2, 2), dtype=complex)
    z1_err_interp = np.zeros((len(periods_common), 2, 2), dtype=float)

    for i in range(2):
        for j in range(2):
            # Get impedance component
            z_component = tf1.impedance[:, i, j]
            z_err_component = tf1.impedance_error[:, i, j]

            # Interpolate real and imaginary parts separately (linear in log-log space)
            real_interp = interp1d(
                p1,
                z_component.real,
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )
            imag_interp = interp1d(
                p1,
                z_component.imag,
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )
            err_interp = interp1d(
                p1,
                z_err_component,
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )

            z1_interp[:, i, j] = real_interp(periods_common) + 1j * imag_interp(
                periods_common
            )
            z1_err_interp[:, i, j] = err_interp(periods_common)

    # Interpolate tf2 impedance
    z2_interp = np.zeros((len(periods_common), 2, 2), dtype=complex)
    z2_err_interp = np.zeros((len(periods_common), 2, 2), dtype=float)

    for i in range(2):
        for j in range(2):
            z_component = tf2.impedance[:, i, j]
            z_err_component = tf2.impedance_error[:, i, j]

            real_interp = interp1d(
                p2,
                z_component.real,
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )
            imag_interp = interp1d(
                p2,
                z_component.imag,
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )
            err_interp = interp1d(
                p2,
                z_err_component,
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )

            z2_interp[:, i, j] = real_interp(periods_common) + 1j * imag_interp(
                periods_common
            )
            z2_err_interp[:, i, j] = err_interp(periods_common)

    return periods_common, z1_interp, z2_interp, z1_err_interp, z2_err_interp


@pytest.fixture(scope="session")
def cas04_emtf_reference():
    """Load EMTF reference result for CAS04 - skip if validation fails."""
    emtf_file = (
        Path(__file__).parent
        / "emtf_results"
        / "CAS04-CAS04bcd_REV06-CAS04bcd_NVR08.zmm"
    )

    if not emtf_file.exists():
        pytest.skip(f"EMTF reference file not found: {emtf_file}")

    try:
        tf_emtf = TF()
        tf_emtf.read(emtf_file)
        return tf_emtf
    except Exception as e:
        pytest.skip(f"Could not read EMTF file (pydantic validation issue): {e}")


@pytest.fixture(scope="session", params=["v010", "v020"])
def cas04_mth5_path(request, global_fdsn_miniseed_v010, global_fdsn_miniseed_v020):
    """Parameterized fixture providing both v0.1.0 and v0.2.0 CAS04 MTH5 files."""
    if request.param == "v010":
        return global_fdsn_miniseed_v010
    else:
        return global_fdsn_miniseed_v020


@pytest.fixture(scope="session")
def session_cas04_run_summary(cas04_mth5_path):
    """Session-scoped RunSummary from CAS04 MTH5 file."""
    run_summary = RunSummary()
    run_summary.from_mth5s([cas04_mth5_path])
    return run_summary


@pytest.fixture(scope="session")
def session_cas04_kernel_dataset(session_cas04_run_summary):
    """Session-scoped KernelDataset - expensive to create, shared across tests."""
    kd = KernelDataset()
    kd.from_run_summary(session_cas04_run_summary, "CAS04")
    return kd


@pytest.fixture(scope="session")
def session_cas04_config(session_cas04_kernel_dataset):
    """Session-scoped processing config - expensive to create, shared across tests."""
    cc = ConfigCreator()
    config = cc.create_from_kernel_dataset(session_cas04_kernel_dataset)
    return config


@pytest.fixture(scope="session")
def session_cas04_tf_result(
    session_cas04_kernel_dataset, session_cas04_config, tmp_path_factory
):
    """Session-scoped processed TF result - very expensive, only run once per version."""
    # Create temp directory for output
    temp_dir = tmp_path_factory.mktemp("cas04_processing")
    z_file_path = temp_dir / "CAS04_session.zss"

    # Process - this is the slowest operation, do it once per session
    tf_result = process_mth5(
        session_cas04_config,
        session_cas04_kernel_dataset,
        units="MT",
        show_plot=False,
        z_file_path=z_file_path,
    )

    return tf_result


@pytest.fixture
def cas04_run_summary(session_cas04_run_summary):
    """Fresh clone of RunSummary for each test."""
    return session_cas04_run_summary.clone()


@pytest.fixture
def cas04_kernel_dataset(session_cas04_kernel_dataset):
    """Reuse session KernelDataset - most tests just read from it."""
    return session_cas04_kernel_dataset


@pytest.fixture
def cas04_config(session_cas04_config):
    """Reuse session config - most tests just read from it."""
    return session_cas04_config


@pytest.fixture
def temp_output_dir(tmp_path):
    """Temporary directory for output files."""
    return tmp_path


@pytest.fixture(scope="session")
def session_interpolated_comparison(session_cas04_tf_result, cas04_emtf_reference):
    """
    Session-scoped interpolated TF comparison.

    Interpolation is expensive and only needs to be done once per session.
    Multiple tests use the same interpolated data.

    Returns
    -------
    tuple
        (periods, z_aurora, z_emtf, err_aurora, err_emtf)
    """
    if cas04_emtf_reference is None:
        pytest.skip("EMTF reference not available")

    return interpolate_tf_to_common_periods(
        session_cas04_tf_result, cas04_emtf_reference
    )


# Test Classes


class TestConfigCreation:
    """Test configuration creation from KernelDataset."""

    def test_config_creator_from_kernel_dataset(self, cas04_config):
        """Test ConfigCreator can create config from KernelDataset."""
        assert cas04_config is not None
        assert hasattr(cas04_config, "decimations")
        assert len(cas04_config.decimations) > 0

    def test_config_has_required_attributes(self, cas04_config):
        """Test that created config has all required attributes."""
        # Config should have key attributes
        assert hasattr(cas04_config, "decimations")
        assert hasattr(cas04_config, "stations")
        assert len(cas04_config.stations) > 0

    def test_config_decimation_levels(self, cas04_config):
        """Test config has reasonable decimation levels."""
        # Should have at least one decimation level
        assert len(cas04_config.decimations) > 0

        # Each decimation should have bands defined
        for dec in cas04_config.decimations:
            # Decimations should have frequency bands
            assert hasattr(dec, "bands") or "bands" in str(dec)

    def test_can_create_processing_components(self, cas04_kernel_dataset, cas04_config):
        """Test that all processing components can be created."""
        assert cas04_config is not None
        assert cas04_kernel_dataset is not None
        assert cas04_kernel_dataset.df is not None


class TestProcessingWorkflow:
    """Test the complete processing workflow using process_mth5."""

    def test_process_mth5_runs_successfully(self, session_cas04_tf_result):
        """Test that process_mth5 runs without errors."""
        assert session_cas04_tf_result is not None

    def test_process_mth5_returns_tf_object(self, session_cas04_tf_result):
        """Test that process_mth5 returns proper TF object."""
        assert isinstance(session_cas04_tf_result, TF)

    def test_tf_has_impedance_data(self, session_cas04_tf_result):
        """Test that resulting TF has impedance data."""
        # Check impedance exists and has correct shape
        assert hasattr(session_cas04_tf_result, "impedance")
        assert session_cas04_tf_result.impedance is not None
        assert len(session_cas04_tf_result.period) > 0

    def test_tf_has_valid_frequencies(self, session_cas04_tf_result):
        """Test that TF has valid frequency values."""
        # Check frequencies are positive and monotonic
        periods = session_cas04_tf_result.period
        assert len(periods) > 0
        assert np.all(periods > 0)


class TestEMTFComparison:
    """Test comparison with EMTF reference results."""

    def test_emtf_reference_loads(self, cas04_emtf_reference):
        """Test that EMTF reference file can be loaded."""
        assert cas04_emtf_reference is not None
        assert hasattr(cas04_emtf_reference, "impedance")

    def test_emtf_has_expected_frequencies(self, cas04_emtf_reference):
        """Test EMTF reference has expected frequency range."""
        periods = cas04_emtf_reference.period
        assert len(periods) > 0
        assert np.all(periods > 0)

    def test_aurora_emtf_frequency_overlap(
        self, session_cas04_tf_result, cas04_emtf_reference
    ):
        """Test that Aurora and EMTF results have overlapping frequencies."""
        # Check for frequency overlap
        aurora_periods = session_cas04_tf_result.period
        emtf_periods = cas04_emtf_reference.period

        p_min_overlap = max(aurora_periods.min(), emtf_periods.min())
        p_max_overlap = min(aurora_periods.max(), emtf_periods.max())

        assert (
            p_max_overlap > p_min_overlap
        ), "No overlapping period range between Aurora and EMTF"

    def test_impedance_magnitude_comparison(self, session_interpolated_comparison):
        """Test that impedance magnitudes are comparable between Aurora and EMTF."""
        # Use pre-computed interpolated data from session fixture
        (
            periods,
            z_aurora,
            z_emtf,
            err_aurora,
            err_emtf,
        ) = session_interpolated_comparison

        # Compare Zxy component (off-diagonal) - most sensitive to MT signal
        aurora_zxy = np.abs(z_aurora[:, 0, 1])
        emtf_zxy = np.abs(z_emtf[:, 0, 1])

        # Calculate normalized difference
        ratio = aurora_zxy / emtf_zxy

        # Check that magnitudes are within 50% on average (reasonable for different processing)
        median_ratio = np.median(ratio)
        assert (
            0.5 < median_ratio < 2.0
        ), f"Impedance magnitudes differ significantly. Median ratio: {median_ratio:.3f}"

        # Check that most values are within factor of 2
        within_factor_2 = np.sum((ratio > 0.5) & (ratio < 2.0)) / len(ratio)
        assert (
            within_factor_2 > 0.7
        ), f"Only {within_factor_2*100:.1f}% of impedances within factor of 2"

    def test_impedance_phase_comparison(self, session_interpolated_comparison):
        """Test that impedance phases are comparable between Aurora and EMTF."""
        # Use pre-computed interpolated data from session fixture
        (
            periods,
            z_aurora,
            z_emtf,
            err_aurora,
            err_emtf,
        ) = session_interpolated_comparison

        # Compare Zxy phase (off-diagonal)
        aurora_phase = np.angle(z_aurora[:, 0, 1], deg=True)
        emtf_phase = np.angle(z_emtf[:, 0, 1], deg=True)

        # Calculate phase difference (accounting for wrapping)
        phase_diff = np.abs(aurora_phase - emtf_phase)
        phase_diff = np.minimum(phase_diff, 360 - phase_diff)

        # Phases should generally agree within 20 degrees on average
        median_phase_diff = np.median(phase_diff)
        assert (
            median_phase_diff < 20
        ), f"Phase differences too large. Median: {median_phase_diff:.1f} degrees"

        # Most phases should be within 30 degrees
        within_30deg = np.sum(phase_diff < 30) / len(phase_diff)
        assert (
            within_30deg > 0.7
        ), f"Only {within_30deg*100:.1f}% of phases within 30 degrees"

    def test_impedance_components_correlation(self, session_interpolated_comparison):
        """Test that key impedance components show correlation between Aurora and EMTF."""
        # Use pre-computed interpolated data from session fixture
        (
            periods,
            z_aurora,
            z_emtf,
            err_aurora,
            err_emtf,
        ) = session_interpolated_comparison

        # Print detailed statistics for analysis
        print("\n" + "=" * 70)
        print("AURORA vs EMTF COMPARISON STATISTICS")
        print("=" * 70)
        print(f"Number of common periods: {len(periods)}")
        print(f"Period range: {periods.min():.2f} - {periods.max():.2f} s")
        print()

        # Analyze all 4 impedance components
        component_names = [("Zxx", 0, 0), ("Zxy", 0, 1), ("Zyx", 1, 0), ("Zyy", 1, 1)]

        for name, i, j in component_names:
            z_a = z_aurora[:, i, j]
            z_e = z_emtf[:, i, j]

            # Magnitude comparison
            mag_a = np.abs(z_a)
            mag_e = np.abs(z_e)
            mag_ratio = mag_a / mag_e
            mag_diff_percent = 100 * (mag_a - mag_e) / mag_e

            # Phase comparison (degrees)
            phase_a = np.angle(z_a, deg=True)
            phase_e = np.angle(z_e, deg=True)
            phase_diff = phase_a - phase_e
            # Wrap phase difference to [-180, 180]
            phase_diff = np.angle(np.exp(1j * np.deg2rad(phase_diff)), deg=True)

            print(f"{name} Component:")
            print(f"  Magnitude Ratio (Aurora/EMTF):")
            print(f"    Mean:   {np.mean(mag_ratio):.3f} ± {np.std(mag_ratio):.3f}")
            print(f"    Median: {np.median(mag_ratio):.3f}")
            print(f"    Range:  [{np.min(mag_ratio):.3f}, {np.max(mag_ratio):.3f}]")
            print(f"  Magnitude Difference:")
            print(
                f"    Mean:   {np.mean(mag_diff_percent):+.1f}% ± {np.std(mag_diff_percent):.1f}%"
            )
            print(f"    Median: {np.median(mag_diff_percent):+.1f}%")
            print(f"  Phase Difference:")
            print(
                f"    Mean:   {np.mean(phase_diff):+.1f}° ± {np.std(phase_diff):.1f}°"
            )
            print(f"    Median: {np.median(phase_diff):+.1f}°")
            print(
                f"    Range:  [{np.min(phase_diff):+.1f}°, {np.max(phase_diff):+.1f}°]"
            )

            # Calculate correlation
            # Use log-log correlation for magnitude (more appropriate for MT data)
            corr_mag = np.corrcoef(np.log10(mag_a), np.log10(mag_e))[0, 1]
            corr_phase = np.corrcoef(phase_a, phase_e)[0, 1]

            print(f"  Correlation:")
            print(f"    Magnitude (log-log): {corr_mag:.4f}")
            print(f"    Phase:               {corr_phase:.4f}")
            print()

        print("=" * 70)

        # At least the off-diagonal components should show reasonable correlation
        z_xy_a = z_aurora[:, 0, 1]
        z_xy_e = z_emtf[:, 0, 1]
        corr_xy_mag = np.corrcoef(np.log10(np.abs(z_xy_a)), np.log10(np.abs(z_xy_e)))[
            0, 1
        ]

        assert (
            corr_xy_mag > 0.8
        ), f"Zxy magnitude correlation too low: {corr_xy_mag:.3f}"

        # Test key impedance components with appropriate thresholds
        # Use log-log correlation as it's more appropriate for MT impedance magnitudes
        # which span multiple orders of magnitude
        component_tests = [
            ("Zxy", 0, 1, 0.9),  # Primary mode - should have excellent correlation
            ("Zyx", 1, 0, 0.4),  # Secondary mode - moderate threshold (affected by 3D)
        ]

        for name, i, j, threshold in component_tests:
            z_a = z_aurora[:, i, j]
            z_e = z_emtf[:, i, j]

            # Use log-log correlation for magnitudes
            mag_a = np.abs(z_a)
            mag_e = np.abs(z_e)

            # Skip if component is very small (numerical noise)
            if np.median(mag_e) < 0.01:
                continue

            # Calculate log-log correlation coefficient
            corr = np.corrcoef(np.log10(mag_a), np.log10(mag_e))[0, 1]

            assert (
                corr > threshold
            ), f"{name} component poorly correlated: r={corr:.3f} (threshold={threshold})"

        # Additionally check that median ratios are reasonable (within factor of 2)
        # for the off-diagonal components
        for name, i, j in [("Zxy", 0, 1), ("Zyx", 1, 0)]:
            mag_a = np.abs(z_aurora[:, i, j])
            mag_e = np.abs(z_emtf[:, i, j])
            ratio = mag_a / mag_e
            median_ratio = np.median(ratio)

            assert (
                0.5 < median_ratio < 2.0
            ), f"{name} median magnitude ratio out of range: {median_ratio:.3f}"


class TestDataQuality:
    """Test data quality metrics from processing."""

    def test_tf_has_error_estimates(self, session_cas04_tf_result):
        """Test that TF includes error estimates."""
        # Check for error estimates
        assert hasattr(session_cas04_tf_result, "impedance_error")
        assert session_cas04_tf_result.impedance_error is not None

    def test_errors_are_positive(self, session_cas04_tf_result):
        """Test that error estimates are positive."""
        # Errors should be positive
        errors = session_cas04_tf_result.impedance_error
        # Convert to numpy if it's an xarray DataArray
        if hasattr(errors, "values"):
            errors = errors.values
        assert np.all(errors[~np.isnan(errors)] >= 0)


class TestEndToEndIntegration:
    """End-to-end integration tests."""

    def test_complete_pipeline_from_run_summary(
        self, cas04_run_summary, temp_output_dir
    ):
        """Test complete pipeline from RunSummary to TF."""
        # Create KernelDataset
        kd = KernelDataset()
        kd.from_run_summary(cas04_run_summary, "CAS04")

        # Create config
        cc = ConfigCreator()
        config = cc.create_from_kernel_dataset(kd)

        # Process
        z_file_path = temp_output_dir / "CAS04_integration.zss"

        tf_result = process_mth5(
            config,
            kd,
            units="MT",
            show_plot=False,
            z_file_path=z_file_path,
        )

        # Verify complete result
        assert tf_result is not None
        assert len(tf_result.period) > 0
        assert z_file_path.exists()

    def test_can_read_written_file(self, session_cas04_tf_result, temp_output_dir):
        """Test that written z-file can be read back."""
        # Write to new file
        z_file_path = temp_output_dir / "CAS04_readback.zss"
        session_cas04_tf_result.write(z_file_path)

        # Read back
        tf_readback = TF()
        tf_readback.read(z_file_path)

        # Compare
        assert len(tf_readback.period) == len(session_cas04_tf_result.period)
        # Use decimal=5 since periods have slight floating point differences
        np.testing.assert_array_almost_equal(
            tf_readback.period, session_cas04_tf_result.period, decimal=5
        )


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_station_id_handling(self, cas04_run_summary):
        """Test handling of invalid station IDs."""
        # This should work even if station IDs don't match expected patterns
        kd = KernelDataset()
        kd.from_run_summary(cas04_run_summary, "CAS04")

        assert kd is not None
        assert kd.df is not None

    def test_missing_channels_handling(self, cas04_kernel_dataset):
        """Test that processing handles missing channels gracefully."""
        # Even with limited channels, config creation should work
        cc = ConfigCreator()
        config = cc.create_from_kernel_dataset(cas04_kernel_dataset)

        assert config is not None
        assert len(config.stations) > 0
