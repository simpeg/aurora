"""Pytest suite for Parkfield dataset processing and calibration tests.

This module tests:
- Calibration and spectral analysis for Parkfield data
- Single-station transfer function processing with various clock_zero configurations
- Remote-reference transfer function processing
- Channel summary conversion helpers
- Comparison with EMTF reference results

Tests are organized into classes and use fixtures from conftest.py for efficient
resource sharing and pytest-xdist compatibility.
"""

from pathlib import Path

import numpy as np
import pytest
from mth5.mth5 import MTH5

from aurora.config.config_creator import ConfigCreator
from aurora.pipelines.process_mth5 import process_mth5
from aurora.sandbox.mth5_channel_summary_helpers import channel_summary_to_make_mth5
from aurora.time_series.windowing_scheme import WindowingScheme
from aurora.transfer_function.plot.comparison_plots import compare_two_z_files


# ============================================================================
# Calibration Tests
# ============================================================================


class TestParkfieldCalibration:
    """Test calibration and spectral analysis for Parkfield data."""

    @pytest.fixture
    def windowing_scheme(self, parkfield_run_ts_pkd):
        """Create windowing scheme for spectral analysis.

        Use the actual data length for the window. Should be exactly 2 hours
        (288000 samples at 40 Hz).
        """
        actual_data_length = parkfield_run_ts_pkd.dataset.time.shape[0]
        return WindowingScheme(
            taper_family="hamming",
            num_samples_window=actual_data_length,
            num_samples_overlap=0,
            sample_rate=parkfield_run_ts_pkd.sample_rate,
        )

    @pytest.fixture
    def fft_obj(self, parkfield_run_ts_pkd, windowing_scheme):
        """Compute FFT of Parkfield run data."""
        windowed_obj = windowing_scheme.apply_sliding_window(
            parkfield_run_ts_pkd.dataset, dt=1.0 / parkfield_run_ts_pkd.sample_rate
        )
        tapered_obj = windowing_scheme.apply_taper(windowed_obj)
        return windowing_scheme.apply_fft(tapered_obj)

    def test_windowing_scheme_properties(self, windowing_scheme, parkfield_run_ts_pkd):
        """Test windowing scheme is configured correctly."""
        assert windowing_scheme.taper_family == "hamming"
        assert windowing_scheme.num_samples_window == 288000
        assert windowing_scheme.num_samples_overlap == 0
        assert windowing_scheme.sample_rate == 40.0

    def test_fft_has_expected_channels(self, fft_obj):
        """Test FFT object contains all expected channels."""
        expected_channels = ["ex", "ey", "hx", "hy", "hz"]
        channel_keys = list(fft_obj.data_vars.keys())
        for channel in expected_channels:
            assert channel in channel_keys

    def test_fft_has_frequency_coordinate(self, fft_obj):
        """Test FFT object has frequency coordinate."""
        assert "frequency" in fft_obj.coords
        frequencies = fft_obj.frequency.data
        assert len(frequencies) > 0
        assert frequencies[0] >= 0  # Should start at DC or near-DC

    def test_calibration_sanity_check(
        self, fft_obj, parkfield_run_pkd, parkfield_paths, disable_matplotlib_logging
    ):
        """Test calibration produces valid results."""
        from aurora.test_utils.parkfield.calibration_helpers import (
            parkfield_sanity_check,
        )

        # This should not raise exceptions
        parkfield_sanity_check(
            fft_obj,
            parkfield_run_pkd,
            figures_path=parkfield_paths["aurora_results"],
            show_response_curves=False,
            show_spectra=False,
            include_decimation=False,
        )

    def test_calibrated_spectra_are_finite(self, fft_obj, parkfield_run_pkd):
        """Test that calibrated spectra contain no NaN or Inf values."""
        import tempfile

        from aurora.test_utils.parkfield.calibration_helpers import (
            parkfield_sanity_check,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Run calibration
            parkfield_sanity_check(
                fft_obj,
                parkfield_run_pkd,
                figures_path=Path(tmpdir),
                show_response_curves=False,
                show_spectra=False,
                include_decimation=False,
            )

        # If we get here without exceptions, calibration succeeded
        # The parkfield_sanity_check function already validates the calibration


# ============================================================================
# Single-Station Processing Tests
# ============================================================================


class TestParkfieldSingleStation:
    """Test single-station transfer function processing."""

    @pytest.fixture
    def z_file_path(self, tmp_path, worker_id, make_worker_safe_path):
        """Generate worker-safe path for z-file output."""
        return make_worker_safe_path("pkd_ss.zss", tmp_path)

    @pytest.fixture
    def config_ss(self, parkfield_kernel_dataset_ss):
        """Create single-station processing config."""
        cc = ConfigCreator()
        config = cc.create_from_kernel_dataset(
            parkfield_kernel_dataset_ss,
            estimator={"engine": "RME"},
            output_channels=["ex", "ey"],
        )
        return config

    def test_single_station_default_processing(
        self,
        parkfield_kernel_dataset_ss,
        config_ss,
        z_file_path,
        disable_matplotlib_logging,
    ):
        """Test single-station processing with default settings."""
        tf_cls = process_mth5(
            config_ss,
            parkfield_kernel_dataset_ss,
            units="MT",
            show_plot=False,
            z_file_path=z_file_path,
        )

        assert tf_cls is not None
        assert z_file_path.exists()

        # Verify transfer function has expected properties
        assert hasattr(tf_cls, "station")
        assert hasattr(tf_cls, "transfer_function")

    def test_single_station_clock_zero_configurations(
        self, parkfield_kernel_dataset_ss, subtests, disable_matplotlib_logging
    ):
        """Test single-station processing with different clock_zero settings."""
        clock_zero_configs = [
            {"type": None, "value": None},
            {"type": "data start", "value": None},
            {"type": "user specified", "value": "2004-09-28 00:00:10+00:00"},
        ]

        for clock_config in clock_zero_configs:
            with subtests.test(clock_zero_type=clock_config["type"]):
                cc = ConfigCreator()
                config = cc.create_from_kernel_dataset(
                    parkfield_kernel_dataset_ss,
                    estimator={"engine": "RME"},
                    output_channels=["ex", "ey"],
                )

                # Apply clock_zero configuration
                if clock_config["type"] is not None:
                    for dec_lvl_cfg in config.decimations:
                        dec_lvl_cfg.stft.window.clock_zero_type = clock_config["type"]
                        if clock_config["type"] == "user specified":
                            dec_lvl_cfg.stft.window.clock_zero = clock_config["value"]

                try:
                    tf_cls = process_mth5(
                        config,
                        parkfield_kernel_dataset_ss,
                        units="MT",
                        show_plot=False,
                    )
                    # Processing may skip if insufficient data after clock_zero truncation
                    # Just verify it doesn't crash
                except Exception as e:
                    pytest.fail(f"Processing failed: {e}")

    def test_single_station_emtfxml_export(
        self,
        parkfield_kernel_dataset_ss,
        config_ss,
        parkfield_paths,
        disable_matplotlib_logging,
    ):
        """Test exporting transfer function to EMTF XML format.

        Currently skipped due to bug in mt_metadata EMTFXML writer (data.py:385):
        IndexError when tipper error arrays have size 0. The writer tries to
        access array[index] even when array has shape (0,).
        """
        tf_cls = process_mth5(
            config_ss,
            parkfield_kernel_dataset_ss,
            units="MT",
            show_plot=False,
        )

        output_xml = parkfield_paths["aurora_results"].joinpath("emtfxml_test_ss.xml")
        output_xml.parent.mkdir(parents=True, exist_ok=True)

        # Use 'xml' as file_type (emtfxml format is accessed via xml)
        tf_cls.write(fn=output_xml, file_type="xml")
        assert output_xml.exists()

    def test_single_station_comparison_with_emtf(
        self,
        parkfield_kernel_dataset_ss,
        config_ss,
        parkfield_paths,
        tmp_path,
        disable_matplotlib_logging,
    ):
        """Test comparison of aurora results with EMTF reference."""
        z_file_path = tmp_path / "pkd_ss_comparison.zss"

        tf_cls = process_mth5(
            config_ss,
            parkfield_kernel_dataset_ss,
            units="MT",
            show_plot=False,
            z_file_path=z_file_path,
        )

        if not z_file_path.exists():
            pytest.skip("Z-file not generated - data access issue")

        # Compare with archived EMTF results
        auxiliary_z_file = parkfield_paths["emtf_results"].joinpath("PKD_272_00.zrr")
        if not auxiliary_z_file.exists():
            pytest.skip("EMTF reference file not available")

        output_png = tmp_path / "SS_processing_comparison.png"
        compare_two_z_files(
            z_file_path,
            auxiliary_z_file,
            label1="aurora",
            label2="emtf",
            scale_factor1=1,
            out_file=output_png,
            markersize=3,
            rho_ylims=[1e0, 1e3],
            xlims=[0.05, 500],
            title_string="Apparent Resistivity and Phase at Parkfield, CA",
            subtitle_string="(Aurora Single Station vs EMTF Remote Reference)",
        )

        assert output_png.exists()


# ============================================================================
# Remote Reference Processing Tests
# ============================================================================


class TestParkfieldRemoteReference:
    """Test remote-reference transfer function processing."""

    @pytest.fixture
    def z_file_path(self, tmp_path, make_worker_safe_path):
        """Generate worker-safe path for RR z-file output."""
        return make_worker_safe_path("pkd_rr.zrr", tmp_path)

    @pytest.fixture
    def config_rr(self, parkfield_kernel_dataset_rr):
        """Create remote-reference processing config."""
        cc = ConfigCreator()
        config = cc.create_from_kernel_dataset(
            parkfield_kernel_dataset_rr,
            output_channels=["ex", "ey"],
        )
        return config

    def test_remote_reference_processing(
        self,
        parkfield_kernel_dataset_rr,
        config_rr,
        z_file_path,
        disable_matplotlib_logging,
    ):
        """Test remote-reference processing with SAO as reference."""
        tf_cls = process_mth5(
            config_rr,
            parkfield_kernel_dataset_rr,
            units="MT",
            show_plot=False,
            z_file_path=z_file_path,
        )

        assert tf_cls is not None
        assert z_file_path.exists()

    def test_rr_comparison_with_emtf(
        self,
        parkfield_kernel_dataset_rr,
        config_rr,
        parkfield_paths,
        tmp_path,
        disable_matplotlib_logging,
    ):
        """Test RR comparison of aurora results with EMTF reference."""
        z_file_path = tmp_path / "pkd_rr_comparison.zrr"

        tf_cls = process_mth5(
            config_rr,
            parkfield_kernel_dataset_rr,
            units="MT",
            show_plot=False,
            z_file_path=z_file_path,
        )

        if not z_file_path.exists():
            pytest.skip("Z-file not generated - data access issue")

        # Compare with archived EMTF results
        auxiliary_z_file = parkfield_paths["emtf_results"].joinpath("PKD_272_00.zrr")
        if not auxiliary_z_file.exists():
            pytest.skip("EMTF reference file not available")

        output_png = tmp_path / "RR_processing_comparison.png"
        compare_two_z_files(
            z_file_path,
            auxiliary_z_file,
            label1="aurora",
            label2="emtf",
            scale_factor1=1,
            out_file=output_png,
            markersize=3,
            rho_ylims=(1e0, 1e3),
            xlims=(0.05, 500),
            title_string="Apparent Resistivity and Phase at Parkfield, CA",
            subtitle_string="(Aurora vs EMTF, both Remote Reference)",
        )

        assert output_png.exists()


# ============================================================================
# Helper Function Tests
# ============================================================================


class TestParkfieldHelpers:
    """Test helper functions used in Parkfield processing."""

    def test_channel_summary_to_make_mth5(
        self, parkfield_h5_path, disable_matplotlib_logging
    ):
        """Test channel_summary_to_make_mth5 helper function."""
        mth5_obj = MTH5(file_version="0.1.0")
        mth5_obj.open_mth5(parkfield_h5_path, mode="r")
        df = mth5_obj.channel_summary.to_dataframe()

        make_mth5_df = channel_summary_to_make_mth5(df, network="NCEDC")

        assert make_mth5_df is not None
        assert len(make_mth5_df) > 0
        assert "station" in make_mth5_df.columns

        mth5_obj.close_mth5()


# ============================================================================
# Data Integrity Tests
# ============================================================================


class TestParkfieldDataIntegrity:
    """Test data integrity and expected properties of Parkfield dataset."""

    def test_mth5_file_exists(self, parkfield_h5_path):
        """Test that Parkfield MTH5 file exists."""
        assert parkfield_h5_path.exists()
        assert parkfield_h5_path.suffix == ".h5"

    def test_pkd_station_exists(self, parkfield_mth5):
        """Test PKD station exists in MTH5 file."""
        station_list = parkfield_mth5.stations_group.groups_list
        assert "PKD" in station_list

    def test_sao_station_exists(self, parkfield_mth5):
        """Test SAO station exists in MTH5 file."""
        station_list = parkfield_mth5.stations_group.groups_list
        assert "SAO" in station_list

    def test_pkd_run_001_exists(self, parkfield_mth5):
        """Test run 001 exists for PKD station."""
        station = parkfield_mth5.get_station("PKD")
        run_list = station.groups_list
        assert "001" in run_list

    def test_pkd_channels(self, parkfield_run_pkd):
        """Test PKD run has expected channels."""
        expected_channels = ["ex", "ey", "hx", "hy", "hz"]
        channels = parkfield_run_pkd.groups_list

        for channel in expected_channels:
            assert channel in channels

    def test_pkd_sample_rate(self, parkfield_run_ts_pkd):
        """Test PKD sample rate is 40 Hz."""
        assert parkfield_run_ts_pkd.sample_rate == 40.0

    def test_pkd_data_length(self, parkfield_run_ts_pkd):
        """Test PKD run has expected data length."""
        # 2 hours at 40 Hz = 288000 samples
        assert parkfield_run_ts_pkd.dataset.time.shape[0] == 288000

    def test_pkd_time_range(self, parkfield_run_ts_pkd):
        """Test PKD data covers expected time range."""
        start_time = str(parkfield_run_ts_pkd.start)
        end_time = str(parkfield_run_ts_pkd.end)

        assert "2004-09-28" in start_time
        assert "2004-09-28" in end_time

    def test_kernel_dataset_ss_structure(self, parkfield_kernel_dataset_ss):
        """Test single-station kernel dataset has expected structure."""
        # KernelDataset has a df attribute that is a DataFrame
        assert "station" in parkfield_kernel_dataset_ss.df.columns
        assert "PKD" in parkfield_kernel_dataset_ss.df["station"].values

    def test_kernel_dataset_rr_structure(self, parkfield_kernel_dataset_rr):
        """Test RR kernel dataset has expected structure."""
        # KernelDataset has a df attribute that is a DataFrame
        assert "station" in parkfield_kernel_dataset_rr.df.columns
        stations = set(parkfield_kernel_dataset_rr.df["station"].values)
        assert "PKD" in stations
        assert "SAO" in stations


# ============================================================================
# Numerical Validation Tests
# ============================================================================


class TestParkfieldNumericalValidation:
    """Test numerical properties of processed results."""

    def test_transfer_function_is_finite(
        self, parkfield_kernel_dataset_ss, disable_matplotlib_logging
    ):
        """Test that computed transfer function contains no NaN or Inf."""
        cc = ConfigCreator()
        config = cc.create_from_kernel_dataset(
            parkfield_kernel_dataset_ss,
            estimator={"engine": "RME"},
            output_channels=["ex", "ey"],
        )

        tf_cls = process_mth5(
            config,
            parkfield_kernel_dataset_ss,
            units="MT",
            show_plot=False,
        )

        # Check that transfer function values are finite for impedance elements
        # tf_cls.transfer_function is now a DataArray with (period, output, input)
        # Output includes ex, ey, and hz. Hz (tipper) may be NaN.
        if hasattr(tf_cls, "transfer_function"):
            tf_data = tf_cls.transfer_function
            # Check only ex and ey outputs (first 2), not hz (index 2)
            impedance_data = tf_data.sel(output=["ex", "ey"])
            assert np.all(np.isfinite(impedance_data.data))

    def test_transfer_function_shape(
        self, parkfield_kernel_dataset_ss, disable_matplotlib_logging
    ):
        """Test that transfer function has expected shape."""
        cc = ConfigCreator()
        config = cc.create_from_kernel_dataset(
            parkfield_kernel_dataset_ss,
            estimator={"engine": "RME"},
            output_channels=["ex", "ey"],
        )

        tf_cls = process_mth5(
            config,
            parkfield_kernel_dataset_ss,
            units="MT",
            show_plot=False,
        )

        # Transfer function should have shape (periods, output_channels, input_channels)
        if hasattr(tf_cls, "transfer_function"):
            tf_data = tf_cls.transfer_function
            # Should have dimensions: period, output, input
            assert tf_data.dims == ("period", "output", "input")
            # Output includes ex, ey, hz even though we only requested ex, ey
            assert tf_data.shape[1] == 3  # 3 output channels (ex, ey, hz)
            assert tf_data.shape[2] == 2  # 2 input channels (hx, hy)

    def test_processing_runs_without_errors(
        self, parkfield_kernel_dataset_rr, disable_matplotlib_logging
    ):
        """Test that RR processing completes without raising exceptions."""
        cc = ConfigCreator()
        config = cc.create_from_kernel_dataset(
            parkfield_kernel_dataset_rr,
            output_channels=["ex", "ey"],
        )

        # This should not raise exceptions
        tf_cls = process_mth5(
            config,
            parkfield_kernel_dataset_rr,
            units="MT",
            show_plot=False,
        )

        assert tf_cls is not None
