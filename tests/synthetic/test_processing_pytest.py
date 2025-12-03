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


def test_can_output_tf_class_and_write_tf_xml(synthetic_test_paths):
    """Test basic TF processing and XML output."""
    tf_cls = process_synthetic_1(file_version="0.1.0")
    xml_file_name = synthetic_test_paths.aurora_results_path.joinpath(
        "syn1_mth5-010.xml"
    )
    tf_cls.write(fn=xml_file_name, file_type="emtfxml")


def test_can_use_channel_nomenclature(synthetic_test_paths):
    """Test processing with custom channel nomenclature."""
    channel_nomenclature = "LEMI12"
    z_file_path = synthetic_test_paths.aurora_results_path.joinpath(
        f"syn1-{channel_nomenclature}.zss"
    )
    tf_cls = process_synthetic_1(
        z_file_path=z_file_path,
        file_version="0.1.0",
        channel_nomenclature=channel_nomenclature,
    )
    xml_file_name = synthetic_test_paths.aurora_results_path.joinpath(
        f"syn1_mth5-0.1.0_{channel_nomenclature}.xml"
    )
    tf_cls.write(fn=xml_file_name, file_type="emtfxml")


def test_can_use_mth5_file_version_020(synthetic_test_paths):
    """Test processing with MTH5 file version 0.2.0."""
    file_version = "0.2.0"
    z_file_path = synthetic_test_paths.aurora_results_path.joinpath(
        f"syn1-{file_version}.zss"
    )
    tf_cls = process_synthetic_1(z_file_path=z_file_path, file_version=file_version)
    xml_file_name = synthetic_test_paths.aurora_results_path.joinpath(
        f"syn1_mth5v{file_version}.xml"
    )
    tf_cls.write(fn=xml_file_name, file_type="emtfxml")
    tf_cls.write(
        fn=z_file_path.parent.joinpath(f"{z_file_path.stem}_from_tf.zss"),
        file_type="zss",
    )


def test_can_use_scale_factor_dictionary(synthetic_test_paths):
    """Test channel scale factors in mt_metadata processing class.

    Expected outputs are four .png:
    - xy_syn1.png: Shows expected 100 Ohm-m resistivity
    - xy_syn1-scaled.png: Overestimates by 4x for 300 Ohm-m resistivity
    - yx_syn1.png: Shows expected 100 Ohm-m resistivity
    - yx_syn1-scaled.png: Underestimates by 4x for 25 Ohm-m resistivity
    """
    z_file_path = synthetic_test_paths.aurora_results_path.joinpath("syn1-scaled.zss")
    tf_cls = process_synthetic_1(z_file_path=z_file_path, test_scale_factor=True)
    tf_cls.write(
        fn=z_file_path.parent.joinpath(f"{z_file_path.stem}_from_tf.zss"),
        file_type="zss",
    )


def test_simultaneous_regression(synthetic_test_paths):
    """Test simultaneous regression processing."""
    z_file_path = synthetic_test_paths.aurora_results_path.joinpath(
        "syn1_simultaneous_estimate.zss"
    )
    tf_cls = process_synthetic_1(z_file_path=z_file_path, simultaneous_regression=True)
    xml_file_name = synthetic_test_paths.aurora_results_path.joinpath(
        "syn1_simultaneous_estimate.xml"
    )
    tf_cls.write(fn=xml_file_name, file_type="emtfxml")
    tf_cls.write(
        fn=z_file_path.parent.joinpath(f"{z_file_path.stem}_from_tf.zss"),
        file_type="zss",
    )


def test_can_process_other_station(synthetic_test_paths):
    """Test processing a different synthetic station."""
    tf_cls = process_synthetic_2(force_make_mth5=True)
    xml_file_name = synthetic_test_paths.aurora_results_path.joinpath("syn2.xml")
    tf_cls.write(fn=xml_file_name, file_type="emtfxml")


def test_can_process_remote_reference_data(synthetic_test_paths):
    """Test remote reference processing with default channel nomenclature."""
    tf_cls = process_synthetic_1r2(channel_nomenclature="default")
    xml_file_name = synthetic_test_paths.aurora_results_path.joinpath(
        "syn12rr_mth5-010.xml"
    )
    tf_cls.write(fn=xml_file_name, file_type="emtfxml")


def test_can_process_remote_reference_data_with_channel_nomenclature(
    synthetic_test_paths,
):
    """Test remote reference processing with custom channel nomenclature."""
    tf_cls = process_synthetic_1r2(channel_nomenclature="LEMI34")
    xml_file_name = synthetic_test_paths.aurora_results_path.joinpath(
        "syn12rr_mth5-010_LEMI34.xml"
    )
    tf_cls.write(fn=xml_file_name, file_type="emtfxml")
