import pathlib
import unittest

from aurora.pipelines.run_summary import RunSummary
from aurora.test_utils.synthetic.make_mth5_from_asc import create_test1_h5
from aurora.test_utils.synthetic.make_mth5_from_asc import create_test2_h5
from aurora.test_utils.synthetic.make_mth5_from_asc import create_test12rr_h5
from aurora.test_utils.synthetic.make_processing_configs import create_test_run_config
from aurora.test_utils.synthetic.paths import AURORA_RESULTS_PATH
from aurora.test_utils.synthetic.processing_helpers import process_sythetic_data
from aurora.transfer_function.kernel_dataset import KernelDataset


# =============================================================================
#  Tests
# =============================================================================


class TestSyntheticProcessing(unittest.TestCase):
    """
    Runs several synthetic processing tests from config creation to tf_collection.

    """

    def setUp(self):
        self.file_version = "0.1.0"

    @property
    def z_file_path(self):
        return AURORA_RESULTS_PATH.joinpath(self.z_file_base)

    def test_can_output_tf_collection(self):
        z_file_path = AURORA_RESULTS_PATH.joinpath("syn1.zss")
        tf_collection = process_synthetic_1(
            z_file_path=z_file_path, file_version=self.file_version
        )
        assert tf_collection.tf_dict is not None

    def test_can_output_tf_class_and_write_tf_xml(self):
        tf_cls = process_synthetic_1(
            file_version=self.file_version, return_collection=False
        )
        xml_file_base = "syn1_mth5-010.xml"
        xml_file_name = AURORA_RESULTS_PATH.joinpath(xml_file_base)
        tf_cls.write_tf_file(fn=xml_file_name, file_type="emtfxml")

    def test_can_use_channel_nomenclature(self):
        channel_nomencalture = "LEMI12"
        z_file_path = AURORA_RESULTS_PATH.joinpath(f"syn1-{channel_nomencalture}.zss")
        tf_cls = process_synthetic_1(
            z_file_path=z_file_path,
            file_version=self.file_version,
            return_collection=False,
            channel_nomenclature=channel_nomencalture,
        )
        xml_file_base = f"syn1_mth5-{self.file_version}_{channel_nomencalture}.xml"
        xml_file_name = AURORA_RESULTS_PATH.joinpath(xml_file_base)
        tf_cls.write_tf_file(fn=xml_file_name, file_type="emtfxml")

    def test_can_use_mth5_file_version_020(self):
        file_version = "0.2.0"
        z_file_path = AURORA_RESULTS_PATH.joinpath(f"syn1-{file_version}.zss")
        tf_cls = process_synthetic_1(
            z_file_path=z_file_path, file_version=file_version, return_collection=False
        )
        xml_file_base = f"syn1_mth5v{file_version}.xml"
        xml_file_name = AURORA_RESULTS_PATH.joinpath(xml_file_base)
        tf_cls.write_tf_file(fn=xml_file_name, file_type="emtfxml")

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
        tf_collection = process_synthetic_1(
            z_file_path=z_file_path, return_collection=True, test_scale_factor=True
        )
        assert tf_collection.tf_dict is not None

    def test_simultaneos_regression(self):
        z_file_path = AURORA_RESULTS_PATH.joinpath("syn1_simultaneous_estimate.zss")
        tf_collection = process_synthetic_1(
            z_file_path=z_file_path, test_simultaneous_regression=True
        )
        assert tf_collection.tf_dict is not None

    def test_can_process_other_station(self):
        tf_collection = process_synthetic_2()
        assert tf_collection.tf_dict is not None

    def test_can_process_remote_reference_data_to_tf_collection(self):
        tf_collection = process_synthetic_rr12()
        assert tf_collection.tf_dict is not None

    def test_can_process_remote_reference_data_to_tf_class(self):
        tf_cls = process_synthetic_rr12(
            channel_nomenclature="default", return_collection=False
        )
        xml_file_base = "syn12rr_mth5-010.xml"
        xml_file_name = AURORA_RESULTS_PATH.joinpath(xml_file_base)
        tf_cls.write_tf_file(
            fn=xml_file_name, file_type="emtfxml", channel_nomenclature="default"
        )

    def test_can_process_remote_reference_data_with_channel_nomenclature(self):
        tf_cls = process_synthetic_rr12(
            channel_nomenclature="LEMI34", return_collection=False
        )
        xml_file_base = "syn12rr_mth5-010_LEMI34.xml"
        xml_file_name = AURORA_RESULTS_PATH.joinpath(xml_file_base)
        tf_cls.write_tf_file(
            fn=xml_file_name, file_type="emtfxml", channel_nomenclature="LEMI34"
        )


def process_synthetic_1(
    z_file_path="",
    test_scale_factor=False,
    test_simultaneous_regression=False,
    file_version="0.1.0",
    return_collection=True,
    channel_nomenclature="default",
):
    """

    Parameters
    ----------
    z_file_path: str or path
        Where the z-file will be output
    test_scale_factor: bool
        If true, will assign scale factors to the channels
    test_simultaneous_regression: bool
        If True will do regression all outut channels in one step, rather than the
        usual, channel-by-channel method

    Returns
    -------
    tf_result: TransferFunctionCollection or mt_metadata.transfer_functions.TF
        Should change so that it is mt_metadata.TF (see Issue #143)
    """
    mth5_path = create_test1_h5(
        file_version=file_version, channel_nomenclature=channel_nomenclature
    )
    run_summary = RunSummary()
    run_summary.from_mth5s(
        [
            mth5_path,
        ]
    )
    # next two lines purely for codecov
    run_summary.print_mini_summary
    run_summary_clone = run_summary.clone()
    # run_summary.drop_runs_shorter_than(100000)
    tfk_dataset = KernelDataset()
    tfk_dataset.from_run_summary(run_summary_clone, "test1")

    # Test that channel_scale_factors column is optional
    if test_scale_factor:
        scale_factors = {"ex": 10.0, "ey": 3.0, "hx": 6.0, "hy": 5.0, "hz": 100.0}
        tfk_dataset.df["channel_scale_factors"].at[0] = scale_factors
    else:
        tfk_dataset.df.drop(columns=["channel_scale_factors"], inplace=True)

    processing_config = create_test_run_config(
        "test1", tfk_dataset.df, channel_nomenclature=channel_nomenclature
    )

    if test_simultaneous_regression:
        for decimation in processing_config.decimations:
            decimation.estimator.estimate_per_channel = False

    tf_result = process_sythetic_data(
        processing_config,
        tfk_dataset,
        z_file_path=z_file_path,
        return_collection=return_collection,
    )

    if return_collection:
        z_figure_name = z_file_path.name.replace("zss", "png")
        for xy_or_yx in ["xy", "yx"]:
            ttl_str = f"{xy_or_yx} component, test_scale_factor = {test_scale_factor}"
            out_png_name = f"{xy_or_yx}_{z_figure_name}"
            tf_result.rho_phi_plot(
                xy_or_yx=xy_or_yx,
                ttl_str=ttl_str,
                show=False,
                figure_basename=out_png_name,
                figure_path=AURORA_RESULTS_PATH,
            )
    return tf_result


def process_synthetic_2():
    mth5_path = create_test2_h5()
    run_summary = RunSummary()
    run_summary.from_mth5s(
        [
            mth5_path,
        ]
    )
    tfk_dataset = KernelDataset()
    tfk_dataset.from_run_summary(run_summary, "test2")
    processing_config = create_test_run_config("test2", tfk_dataset.df)
    tfc = process_sythetic_data(processing_config, tfk_dataset)
    return tfc


def process_synthetic_rr12(channel_nomenclature="default", return_collection=True):
    mth5_path = create_test12rr_h5(channel_nomenclature=channel_nomenclature)
    run_summary = RunSummary()
    run_summary.from_mth5s(
        [
            mth5_path,
        ]
    )
    tfk_dataset = KernelDataset()
    tfk_dataset.from_run_summary(run_summary, "test1", "test2")
    processing_config = create_test_run_config(
        "test1r2", tfk_dataset.df, channel_nomenclature=channel_nomenclature
    )
    tfc = process_sythetic_data(
        processing_config,
        tfk_dataset,
        return_collection=return_collection,
    )
    return tfc


def main():
    """
    Testing the processing of synthetic data
    """
    unittest.main()


if __name__ == "__main__":
    main()


# def process_synthetic_1_underdetermined():
#     """
#     Just like process_synthetic_1, but the window is ridiculously long so that we
#     encounter the underdetermined problem. We actually pass that test but in testing
#     I found that at the next band over, which has more data because there are multipe
#     FCs the sigma in TRME comes out as negative. see issue #4 and issue #55.
#     Returns
#     -------
#
#     """
#     test_config = CONFIG_PATH.joinpath("test1_run_config_underdetermined.json")
#     # test_config = Path("config", "test1_run_config_underdetermined.json")
#     run_id = "001"
#     process_sythetic_data(test_config, run_id, units="MT")
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
#     process_sythetic_data(test_config, run_id, units="MT")