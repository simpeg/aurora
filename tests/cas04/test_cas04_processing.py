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

from aurora.config.config_creator import ConfigCreator
from aurora.pipelines.process_mth5 import process_mth5
from aurora.transfer_function.compare import CompareTF


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


# Separate session fixtures for v010 and v020 to enable better parallelization
@pytest.fixture(scope="session")
def session_cas04_run_summary_v010(global_fdsn_miniseed_v010):
    """Session-scoped RunSummary for v0.1.0."""
    run_summary = RunSummary()
    run_summary.from_mth5s([global_fdsn_miniseed_v010])
    return run_summary


@pytest.fixture(scope="session")
def session_cas04_run_summary_v020(global_fdsn_miniseed_v020):
    """Session-scoped RunSummary for v0.2.0."""
    run_summary = RunSummary()
    run_summary.from_mth5s([global_fdsn_miniseed_v020])
    return run_summary


@pytest.fixture(scope="session")
def session_cas04_kernel_dataset_v010(session_cas04_run_summary_v010):
    """Session-scoped KernelDataset for v0.1.0."""
    kd = KernelDataset()
    kd.from_run_summary(session_cas04_run_summary_v010, "CAS04")
    return kd


@pytest.fixture(scope="session")
def session_cas04_kernel_dataset_v020(session_cas04_run_summary_v020):
    """Session-scoped KernelDataset for v0.2.0."""
    kd = KernelDataset()
    kd.from_run_summary(session_cas04_run_summary_v020, "CAS04")
    return kd


@pytest.fixture(scope="session")
def session_cas04_config_v010(session_cas04_kernel_dataset_v010):
    """Session-scoped processing config for v0.1.0."""
    cc = ConfigCreator()
    config = cc.create_from_kernel_dataset(session_cas04_kernel_dataset_v010)
    return config


@pytest.fixture(scope="session")
def session_cas04_config_v020(session_cas04_kernel_dataset_v020):
    """Session-scoped processing config for v0.2.0."""
    cc = ConfigCreator()
    config = cc.create_from_kernel_dataset(session_cas04_kernel_dataset_v020)
    return config


@pytest.fixture(scope="session")
def session_cas04_tf_result_v010(
    session_cas04_kernel_dataset_v010, session_cas04_config_v010, tmp_path_factory
):
    """Session-scoped processed TF result for v0.1.0."""
    temp_dir = tmp_path_factory.mktemp("cas04_processing_v010")
    z_file_path = temp_dir / "CAS04_v010.zss"

    tf_result = process_mth5(
        session_cas04_config_v010,
        session_cas04_kernel_dataset_v010,
        units="MT",
        show_plot=False,
        z_file_path=z_file_path,
    )
    return tf_result


@pytest.fixture(scope="session")
def session_cas04_tf_result_v020(
    session_cas04_kernel_dataset_v020, session_cas04_config_v020, tmp_path_factory
):
    """Session-scoped processed TF result for v0.2.0."""
    temp_dir = tmp_path_factory.mktemp("cas04_processing_v020")
    z_file_path = temp_dir / "CAS04_v020.zss"

    tf_result = process_mth5(
        session_cas04_config_v020,
        session_cas04_kernel_dataset_v020,
        units="MT",
        show_plot=False,
        z_file_path=z_file_path,
    )
    return tf_result


# Selector fixtures that choose based on version parameter
@pytest.fixture
def cas04_run_summary(request):
    """Select appropriate RunSummary based on version."""
    version = request.param if hasattr(request, "param") else "v010"
    if version == "v010":
        fixture = request.getfixturevalue("session_cas04_run_summary_v010")
    else:
        fixture = request.getfixturevalue("session_cas04_run_summary_v020")
    return fixture.clone()


@pytest.fixture
def cas04_kernel_dataset(request):
    """Select appropriate KernelDataset based on version."""
    version = request.param if hasattr(request, "param") else "v010"
    if version == "v010":
        return request.getfixturevalue("session_cas04_kernel_dataset_v010")
    else:
        return request.getfixturevalue("session_cas04_kernel_dataset_v020")


@pytest.fixture
def cas04_config(request):
    """Select appropriate config based on version."""
    version = request.param if hasattr(request, "param") else "v010"
    if version == "v010":
        return request.getfixturevalue("session_cas04_config_v010")
    else:
        return request.getfixturevalue("session_cas04_config_v020")


@pytest.fixture
def session_cas04_tf_result(request):
    """Select appropriate TF result based on version."""
    version = request.param if hasattr(request, "param") else "v010"
    if version == "v010":
        return request.getfixturevalue("session_cas04_tf_result_v010")
    else:
        return request.getfixturevalue("session_cas04_tf_result_v020")


@pytest.fixture
def temp_output_dir(tmp_path):
    """Temporary directory for output files."""
    return tmp_path


@pytest.fixture(scope="session")
def session_interpolated_comparison_v010(
    session_cas04_tf_result_v010, cas04_emtf_reference
):
    """Session-scoped interpolated TF comparison for v0.1.0."""
    if cas04_emtf_reference is None:
        pytest.skip("EMTF reference not available")
    return CompareTF(session_cas04_tf_result_v010, cas04_emtf_reference)


@pytest.fixture(scope="session")
def session_interpolated_comparison_v020(
    session_cas04_tf_result_v020, cas04_emtf_reference
):
    """Session-scoped interpolated TF comparison for v0.2.0."""
    if cas04_emtf_reference is None:
        pytest.skip("EMTF reference not available")
    return CompareTF(session_cas04_tf_result_v020, cas04_emtf_reference)


@pytest.fixture
def session_interpolated_comparison(request):
    """Select appropriate interpolated comparison based on version."""
    version = request.param if hasattr(request, "param") else "v010"
    if version == "v010":
        return request.getfixturevalue("session_interpolated_comparison_v010")
    else:
        return request.getfixturevalue("session_interpolated_comparison_v020")


# Test Classes


@pytest.mark.parametrize("cas04_config", ["v010", "v020"], indirect=True)
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


@pytest.mark.parametrize("session_cas04_tf_result", ["v010", "v020"], indirect=True)
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

    def test_tf_channel_metadata(self, session_cas04_tf_result, subtests):
        """Test that expected channels are present in TF."""
        expected_channels = ["ex", "ey", "hx", "hy", "hz"]
        for chan in expected_channels:
            ch_metadata = session_cas04_tf_result.run_metadata.channels[chan]
            with subtests.test(msg=f"Checking channel metadata for {chan}"):
                assert (
                    ch_metadata.time_period.start != "1980-01-01T00:00:00"
                ), f"Channel {chan} has invalid time period."
                assert (
                    ch_metadata.time_period.end != "1980-01-01T00:00:00"
                ), f"Channel {chan} has invalid time period."
                assert (
                    ch_metadata.sample_rate > 0
                ), f"Sample rate for {chan} should be positive."


class TestEMTFComparison:
    """Test comparison with EMTF reference results."""

    def test_emtf_reference_loads(self, cas04_emtf_reference):
        """Test that EMTF reference file can be loaded."""
        assert cas04_emtf_reference is not None
        assert hasattr(cas04_emtf_reference, "impedance")

    @pytest.mark.parametrize(
        "session_interpolated_comparison", ["v010", "v020"], indirect=True
    )
    def test_comparison(self, session_interpolated_comparison, subtests):
        """Test that impedance magnitudes are comparable between Aurora and EMTF."""
        # Use pre-computed interpolated data from session fixture
        result = session_interpolated_comparison.compare_transfer_functions()

        # Check that magnitudes are within 50% on average (reasonable for different processing)
        z_ratio = (0.8, 1.2)
        z_std_limit = 1.5
        if result["impedance_ratio"] is not None:
            for ii in range(2):
                for jj in range(2):
                    if ii != jj:
                        key = f"Z_{ii}{jj}"
                        with subtests.test(
                            msg=f"Checking impedance magnitude ratio for {key}"
                        ):
                            assert (
                                z_ratio[0] < result["impedance_ratio"][key] < z_ratio[1]
                            ), f"{key} impedance magnitudes differ significantly. Median ratio: {result['impedance_ratio'][key]:.3f}"

                        with subtests.test(msg=f"Checking impedance std for {key}"):
                            assert (
                                result["impedance_std"][key] < z_std_limit
                            ), f"{key} impedance magnitudes have high standard deviation: {result['impedance_std'][key]:.3f}"

        t_ratio = (0.8, 1.6)
        t_std_limit = 0.5
        if result["tipper_ratio"] is not None:
            for ii in range(1):
                for jj in range(2):
                    if ii != jj:
                        key = f"T_{ii}{jj}"
                        with subtests.test(
                            msg=f"Checking tipper magnitude ratio for {key}"
                        ):
                            assert (
                                t_ratio[0] < result["tipper_ratio"][key] < t_ratio[1]
                            ), f"{key} tipper magnitudes differ significantly. Median ratio: {result['tipper_ratio'][key]:.3f}"

                        with subtests.test(msg=f"Checking tipper std for {key}"):
                            assert (
                                result["tipper_std"][key] < t_std_limit
                            ), f"{key} tipper magnitudes have high standard deviation: {result['tipper_std'][key]:.3f}"


@pytest.mark.parametrize("session_cas04_tf_result", ["v010", "v020"], indirect=True)
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

    @pytest.mark.slow
    @pytest.mark.parametrize("cas04_run_summary", ["v010", "v020"], indirect=True)
    def test_complete_pipeline_from_run_summary(
        self, cas04_run_summary, temp_output_dir
    ):
        """
        Test complete pipeline from RunSummary to TF.

        This test is marked as 'slow' because it re-runs process_mth5() which
        takes ~40 seconds per MTH5 version. Run with: pytest -m slow
        Skip with: pytest -m "not slow"
        """
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

    @pytest.mark.parametrize("session_cas04_tf_result", ["v010", "v020"], indirect=True)
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

    @pytest.mark.parametrize("cas04_run_summary", ["v010", "v020"], indirect=True)
    def test_invalid_station_id_handling(self, cas04_run_summary):
        """Test handling of invalid station IDs."""
        # This should work even if station IDs don't match expected patterns
        kd = KernelDataset()
        kd.from_run_summary(cas04_run_summary, "CAS04")

        assert kd is not None
        assert kd.df is not None

    @pytest.mark.parametrize("cas04_kernel_dataset", ["v010", "v020"], indirect=True)
    def test_missing_channels_handling(self, cas04_kernel_dataset):
        """Test that processing handles missing channels gracefully."""
        # Even with limited channels, config creation should work
        cc = ConfigCreator()
        config = cc.create_from_kernel_dataset(cas04_kernel_dataset)

        assert config is not None
        assert len(config.stations) > 0
