import logging
import unittest

from aurora.test_utils.synthetic.paths import SyntheticTestPaths
from aurora.test_utils.synthetic.processing_helpers import process_synthetic_1
from aurora.test_utils.synthetic.processing_helpers import process_synthetic_1r2
from aurora.test_utils.synthetic.processing_helpers import process_synthetic_2
from mth5.helpers import close_open_files

# from typing import Optional, Union

synthetic_test_paths = SyntheticTestPaths()
synthetic_test_paths.mkdirs()
AURORA_RESULTS_PATH = synthetic_test_paths.aurora_results_path

# =============================================================================
#  Tests
# =============================================================================


class TestSyntheticProcessing(unittest.TestCase):
    """
    Runs several synthetic processing tests from config creation to tf_cls.

    """

    def setUp(self):
        close_open_files()
        self.file_version = "0.1.0"
        logging.getLogger("matplotlib.font_manager").disabled = True
        logging.getLogger("matplotlib.ticker").disabled = True

    def test_no_crash_with_too_many_decimations(self):
        z_file_path = AURORA_RESULTS_PATH.joinpath("syn1_tfk.zss")
        xml_file_base = "syn1_tfk.xml"
        xml_file_name = AURORA_RESULTS_PATH.joinpath(xml_file_base)
        tf_cls = process_synthetic_1(
            config_keyword="test1_tfk", z_file_path=z_file_path
        )
        tf_cls.write(fn=xml_file_name, file_type="emtfxml")
        tf_cls.write(
            fn=z_file_path.parent.joinpath(f"{z_file_path.stem}_from_tf.zss"),
            file_type="zss",
        )

        xml_file_base = "syn1r2_tfk.xml"
        xml_file_name = AURORA_RESULTS_PATH.joinpath(xml_file_base)
        tf_cls = process_synthetic_1r2(config_keyword="test1r2_tfk")
        tf_cls.write(fn=xml_file_name, file_type="emtfxml")

    def test_can_output_tf_class_and_write_tf_xml(self):
        tf_cls = process_synthetic_1(file_version=self.file_version)
        xml_file_base = "syn1_mth5-010.xml"
        xml_file_name = AURORA_RESULTS_PATH.joinpath(xml_file_base)
        tf_cls.write(fn=xml_file_name, file_type="emtfxml")

    def test_can_use_channel_nomenclature(self):
        channel_nomenclature = "LEMI12"
        z_file_path = AURORA_RESULTS_PATH.joinpath(f"syn1-{channel_nomenclature}.zss")
        tf_cls = process_synthetic_1(
            z_file_path=z_file_path,
            file_version=self.file_version,
            channel_nomenclature=channel_nomenclature,
        )
        xml_file_base = f"syn1_mth5-{self.file_version}_{channel_nomenclature}.xml"
        xml_file_name = AURORA_RESULTS_PATH.joinpath(xml_file_base)
        tf_cls.write(fn=xml_file_name, file_type="emtfxml")

    def test_can_use_mth5_file_version_020(self):
        file_version = "0.2.0"
        z_file_path = AURORA_RESULTS_PATH.joinpath(f"syn1-{file_version}.zss")
        tf_cls = process_synthetic_1(z_file_path=z_file_path, file_version=file_version)
        xml_file_base = f"syn1_mth5v{file_version}.xml"
        xml_file_name = AURORA_RESULTS_PATH.joinpath(xml_file_base)
        tf_cls.write(fn=xml_file_name, file_type="emtfxml")
        tf_cls.write(
            fn=z_file_path.parent.joinpath(f"{z_file_path.stem}_from_tf.zss"),
            file_type="zss",
        )

    def test_can_use_scale_factor_dictionary(self):
        """
        2022-05-13: Added a duplicate run of process_synthetic_1, which is intended to
        test the channel_scale_factors in the new mt_metadata processing class.
        Expected outputs are four .png:

        xy_syn1.png : Shows expected 100 Ohm-m resisitivity
        xy_syn1-scaled.png : Overestimates by 4x for 300 Ohm-m resistivity
        yx_syn1.png : Shows expected 100 Ohm-m resisitivity
        yx_syn1-scaled.png : Underestimates by 4x for 25 Ohm-m resistivity
        These .png are stores in aurora_results folder

        """
        z_file_path = AURORA_RESULTS_PATH.joinpath("syn1-scaled.zss")
        tf_cls = process_synthetic_1(
            z_file_path=z_file_path,
            test_scale_factor=True,
        )
        tf_cls.write(
            fn=z_file_path.parent.joinpath(f"{z_file_path.stem}_from_tf.zss"),
            file_type="zss",
        )

    def test_simultaneous_regression(self):
        z_file_path = AURORA_RESULTS_PATH.joinpath("syn1_simultaneous_estimate.zss")
        tf_cls = process_synthetic_1(
            z_file_path=z_file_path, simultaneous_regression=True
        )
        xml_file_base = "syn1_simultaneous_estimate.xml"
        xml_file_name = AURORA_RESULTS_PATH.joinpath(xml_file_base)
        tf_cls.write(fn=xml_file_name, file_type="emtfxml")
        tf_cls.write(
            fn=z_file_path.parent.joinpath(f"{z_file_path.stem}_from_tf.zss"),
            file_type="zss",
        )

    def test_can_process_other_station(self, force_make_mth5=True):
        tf_cls = process_synthetic_2(force_make_mth5=force_make_mth5)
        xml_file_name = AURORA_RESULTS_PATH.joinpath("syn2.xml")
        tf_cls.write(fn=xml_file_name, file_type="emtfxml")

    def test_can_process_remote_reference_data(self):
        tf_cls = process_synthetic_1r2(channel_nomenclature="default")
        xml_file_base = "syn12rr_mth5-010.xml"
        xml_file_name = AURORA_RESULTS_PATH.joinpath(xml_file_base)
        tf_cls.write(
            fn=xml_file_name,
            file_type="emtfxml",
        )

    def test_can_process_remote_reference_data_with_channel_nomenclature(self):
        tf_cls = process_synthetic_1r2(channel_nomenclature="LEMI34")
        xml_file_base = "syn12rr_mth5-010_LEMI34.xml"
        xml_file_name = AURORA_RESULTS_PATH.joinpath(xml_file_base)
        tf_cls.write(
            fn=xml_file_name,
            file_type="emtfxml",
        )


def main():
    """
    Testing the processing of synthetic data
    """
    # tmp = TestSyntheticProcessing()
    # tmp.setUp()
    # tmp.test_can_process_other_station() # makes FC csvs

    # tmp.test_can_output_tf_class_and_write_tf_xml()
    # tmp.test_no_crash_with_too_many_decimations()
    # tmp.test_can_use_scale_factor_dictionary()

    unittest.main()


if __name__ == "__main__":
    main()


# def process_synthetic_1_underdetermined():
#     """
#     Just like process_synthetic_1, but the window is ridiculously long so that we
#     encounter the underdetermined problem. We actually pass that test but in testing
#     I found that at the next band over, which has more data because there are multipe
#     FCs the sigma in RME comes out as negative. see issue #4 and issue #55.
#     Returns
#     -------
#
#     """
#     test_config = CONFIG_PATH.joinpath("test1_run_config_underdetermined.json")
#     # test_config = Path("config", "test1_run_config_underdetermined.json")
#     run_id = "001"
#     process_mth5(test_config, run_id, units="MT")
#
#
# def process_synthetic_1_with_nans():
#     """
#
#     Returns
#     -------
#
#     """
#     test_config = CONFIG_PATH.joinpath("test1_run_config_nan.json")
#     #    test_config = Path("config", "test1_run_config_nan.json")
#     run_id = "001"
#     process_mth5(test_config, run_id, units="MT")
