"""Pytest translation of test_processing.py

Runs several synthetic processing tests from config creation to tf_cls.
"""

import logging

import numpy as np
import pytest

from aurora.pipelines.process_mth5 import process_mth5
from aurora.test_utils.synthetic.make_processing_configs import (
    make_processing_config_and_kernel_dataset,
)
from aurora.test_utils.synthetic.processing_helpers import (
    process_synthetic_1,
    process_synthetic_1r2,
    process_synthetic_2,
)


@pytest.fixture(autouse=True)
def setup_logging():
    """Disable noisy matplotlib loggers."""
    logging.getLogger("matplotlib.font_manager").disabled = True
    logging.getLogger("matplotlib.ticker").disabled = True


def _synthetic_rho_phase(tf):
    """Return median off-diagonal apparent resistivity and phase."""
    frequency = 1.0 / np.asarray(tf.period)
    impedance = np.asarray(tf.impedance)
    rho_phase = []
    for output_index, input_index in ((0, 1), (1, 0)):
        z_component = impedance[:, output_index, input_index]
        # Aurora's MT impedance convention is mV/km per nT.
        apparent_resistivity = 0.2 * np.abs(z_component) ** 2 / frequency
        phase = np.mod(np.angle(z_component, deg=True), 180.0)
        rho_phase.append((np.nanmedian(apparent_resistivity), np.nanmedian(phase)))
    return np.asarray(rho_phase)


def test_arma_prewhitening_recovers_synthetic_half_space(worker_safe_test1_h5):
    """ARMA(3, 0) FCs recover the test1 100 Ohm-m, 45-degree half-space."""
    arma_dataset, arma_config = make_processing_config_and_kernel_dataset(
        config_keyword="test1",
        station_id="test1",
        mth5s=[worker_safe_test1_h5],
    )
    for decimation in arma_config.decimations:
        decimation.stft.prewhitening_type = "ARMA"
        decimation.stft.prewhitening_ar_order = 3
        decimation.stft.prewhitening_ma_order = 0

    arma_tf = process_mth5(arma_config, tfk_dataset=arma_dataset)
    arma_rho_phase = _synthetic_rho_phase(arma_tf)

    np.testing.assert_allclose(
        arma_rho_phase[:, 0], 100.0, rtol=0.1, err_msg="apparent resistivity"
    )
    np.testing.assert_allclose(arma_rho_phase[:, 1], 45.0, atol=5.0, err_msg="phase")

    baseline_dataset, baseline_config = make_processing_config_and_kernel_dataset(
        config_keyword="test1",
        station_id="test1",
        mth5s=[worker_safe_test1_h5],
    )
    baseline_tf = process_mth5(baseline_config, tfk_dataset=baseline_dataset)
    baseline_rho_phase = _synthetic_rho_phase(baseline_tf)
    np.testing.assert_allclose(arma_rho_phase, baseline_rho_phase, rtol=0.1, atol=5.0)


@pytest.mark.skip(
    reason="mt_metadata pydantic branch has issue with provenance.archive.comments.value being None"
)
def test_no_crash_with_too_many_decimations(synthetic_test_paths):
    """Test processing with many decimation levels."""
    z_file_path = synthetic_test_paths.aurora_results_path.joinpath("syn1_tfk.zss")
    xml_file_name = synthetic_test_paths.aurora_results_path.joinpath("syn1_tfk.xml")
    tf_cls = process_synthetic_1(config_keyword="test1_tfk", z_file_path=z_file_path)
    tf_cls.write(fn=xml_file_name, file_type="emtfxml")
    tf_cls.write(
        fn=z_file_path.parent.joinpath(f"{z_file_path.stem}_from_tf.zss"),
        file_type="zss",
    )

    xml_file_name = synthetic_test_paths.aurora_results_path.joinpath("syn1r2_tfk.xml")
    tf_cls = process_synthetic_1r2(config_keyword="test1r2_tfk")
    tf_cls.write(fn=xml_file_name, file_type="emtfxml")


class TestSyntheticTest1Processing:
    """Tests for test1 synthetic processing - share processed TF across tests."""

    @pytest.fixture(scope="class")
    def processed_tf_test1(self, worker_safe_test1_h5):
        """Process test1 once and reuse across all tests in this class."""
        return process_synthetic_1(file_version="0.1.0", mth5_path=worker_safe_test1_h5)

    def test_can_output_tf_class_and_write_tf_xml(
        self, synthetic_test_paths, processed_tf_test1
    ):
        """Test basic TF processing and XML output."""
        xml_file_name = synthetic_test_paths.aurora_results_path.joinpath(
            "syn1_mth5-010.xml"
        )
        processed_tf_test1.write(fn=xml_file_name, file_type="emtfxml")

    def test_can_use_mth5_file_version_020(
        self, synthetic_test_paths, processed_tf_test1
    ):
        """Test processing with MTH5 file version 0.2.0."""
        file_version = "0.2.0"
        z_file_path = synthetic_test_paths.aurora_results_path.joinpath(
            f"syn1-{file_version}.zss"
        )
        xml_file_name = synthetic_test_paths.aurora_results_path.joinpath(
            f"syn1_mth5v{file_version}.xml"
        )
        processed_tf_test1.write(fn=xml_file_name, file_type="emtfxml")
        processed_tf_test1.write(
            fn=z_file_path.parent.joinpath(f"{z_file_path.stem}_from_tf.zss"),
            file_type="zss",
        )

    @pytest.fixture(scope="class")
    def processed_tf_scaled(self, worker_safe_test1_h5, synthetic_test_paths):
        """Process test1 with scale factors once and reuse."""
        z_file_path = synthetic_test_paths.aurora_results_path.joinpath(
            "syn1-scaled.zss"
        )
        return process_synthetic_1(
            z_file_path=z_file_path,
            test_scale_factor=True,
            mth5_path=worker_safe_test1_h5,
        )

    def test_can_use_scale_factor_dictionary(
        self, processed_tf_scaled, synthetic_test_paths
    ):
        """Test channel scale factors in mt_metadata processing class.

        Expected outputs are four .png:
        - xy_syn1.png: Shows expected 100 Ohm-m resistivity
        - xy_syn1-scaled.png: Overestimates by 4x for 300 Ohm-m resistivity
        - yx_syn1.png: Shows expected 100 Ohm-m resistivity
        - yx_syn1-scaled.png: Underestimates by 4x for 25 Ohm-m resistivity
        """
        z_file_path = synthetic_test_paths.aurora_results_path.joinpath(
            "syn1-scaled.zss"
        )
        processed_tf_scaled.write(
            fn=z_file_path.parent.joinpath(f"{z_file_path.stem}_from_tf.zss"),
            file_type="zss",
        )

    @pytest.fixture(scope="class")
    def processed_tf_simultaneous(self, worker_safe_test1_h5, synthetic_test_paths):
        """Process test1 with simultaneous regression once and reuse."""
        z_file_path = synthetic_test_paths.aurora_results_path.joinpath(
            "syn1_simultaneous_estimate.zss"
        )
        return process_synthetic_1(
            z_file_path=z_file_path,
            simultaneous_regression=True,
            mth5_path=worker_safe_test1_h5,
        )

    def test_simultaneous_regression(
        self, processed_tf_simultaneous, synthetic_test_paths
    ):
        """Test simultaneous regression processing."""
        xml_file_name = synthetic_test_paths.aurora_results_path.joinpath(
            "syn1_simultaneous_estimate.xml"
        )
        z_file_path = synthetic_test_paths.aurora_results_path.joinpath(
            "syn1_simultaneous_estimate.zss"
        )
        processed_tf_simultaneous.write(fn=xml_file_name, file_type="emtfxml")
        processed_tf_simultaneous.write(
            fn=z_file_path.parent.joinpath(f"{z_file_path.stem}_from_tf.zss"),
            file_type="zss",
        )


def test_can_use_channel_nomenclature(synthetic_test_paths, mth5_target_dir, worker_id):
    """Test processing with custom channel nomenclature.

    Note: This test creates its own MTH5 with specific nomenclature, so it cannot
    share fixtures with other tests.
    """
    from mth5.data.make_mth5_from_asc import create_test1_h5

    channel_nomenclature = "LEMI12"
    # Create MTH5 with specific nomenclature in worker-safe directory
    mth5_path = create_test1_h5(
        file_version="0.1.0",
        channel_nomenclature=channel_nomenclature,
        target_folder=mth5_target_dir,
    )

    z_file_path = synthetic_test_paths.aurora_results_path.joinpath(
        f"syn1-{channel_nomenclature}.zss"
    )
    tf_cls = process_synthetic_1(
        z_file_path=z_file_path,
        file_version="0.1.0",
        channel_nomenclature=channel_nomenclature,
        mth5_path=mth5_path,
    )
    xml_file_name = synthetic_test_paths.aurora_results_path.joinpath(
        f"syn1_mth5-0.1.0_{channel_nomenclature}.xml"
    )
    tf_cls.write(fn=xml_file_name, file_type="emtfxml")


class TestSyntheticTest2Processing:
    """Tests for test2 synthetic processing."""

    @pytest.fixture(scope="class")
    def processed_tf_test2(self, worker_safe_test2_h5):
        """Process test2 once and reuse."""
        return process_synthetic_2(force_make_mth5=True, mth5_path=worker_safe_test2_h5)

    def test_can_process_other_station(self, synthetic_test_paths, processed_tf_test2):
        """Test processing a different synthetic station."""
        xml_file_name = synthetic_test_paths.aurora_results_path.joinpath("syn2.xml")
        processed_tf_test2.write(fn=xml_file_name, file_type="emtfxml")


class TestRemoteReferenceProcessing:
    """Tests for remote reference processing."""

    @pytest.fixture(scope="class")
    def processed_tf_test12rr(self, worker_safe_test12rr_h5):
        """Process test12rr once and reuse."""
        return process_synthetic_1r2(
            channel_nomenclature="default", mth5_path=worker_safe_test12rr_h5
        )

    def test_can_process_remote_reference_data(
        self, synthetic_test_paths, processed_tf_test12rr
    ):
        """Test remote reference processing with default channel nomenclature."""
        xml_file_name = synthetic_test_paths.aurora_results_path.joinpath(
            "syn12rr_mth5-010.xml"
        )
        processed_tf_test12rr.write(fn=xml_file_name, file_type="emtfxml")


def test_can_process_remote_reference_data_with_channel_nomenclature(
    synthetic_test_paths,
    mth5_target_dir,
    worker_id,
):
    """Test remote reference processing with custom channel nomenclature.

    Note: This test creates its own MTH5 with specific nomenclature, so it cannot
    share fixtures with other tests.
    """
    from mth5.data.make_mth5_from_asc import create_test12rr_h5

    channel_nomenclature = "LEMI34"
    # Create MTH5 with specific nomenclature in worker-safe directory
    mth5_path = create_test12rr_h5(
        channel_nomenclature=channel_nomenclature,
        target_folder=mth5_target_dir,
    )

    tf_cls = process_synthetic_1r2(
        channel_nomenclature=channel_nomenclature, mth5_path=mth5_path
    )
    xml_file_name = synthetic_test_paths.aurora_results_path.joinpath(
        "syn12rr_mth5-010_LEMI34.xml"
    )
    tf_cls.write(fn=xml_file_name, file_type="emtfxml")
