"""Pytest translation of test_processing.py

Runs several synthetic processing tests from config creation to tf_cls.
"""

import logging

import pytest

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
