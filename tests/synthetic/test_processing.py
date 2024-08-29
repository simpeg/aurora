import logging
import unittest

from aurora.pipelines.process_mth5 import process_mth5

from aurora.test_utils.synthetic.make_processing_configs import (
    create_test_run_config,
)
from aurora.test_utils.synthetic.paths import SyntheticTestPaths
from mth5.data.make_mth5_from_asc import create_test1_h5
from mth5.data.make_mth5_from_asc import create_test2_h5
from mth5.data.make_mth5_from_asc import create_test12rr_h5
from mth5.helpers import close_open_files

# from mtpy-v2
from mtpy.processing import RunSummary, KernelDataset

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
        channel_nomencalture = "LEMI12"
        z_file_path = AURORA_RESULTS_PATH.joinpath(f"syn1-{channel_nomencalture}.zss")
        tf_cls = process_synthetic_1(
            z_file_path=z_file_path,
            file_version=self.file_version,
            channel_nomenclature=channel_nomencalture,
        )
        xml_file_base = f"syn1_mth5-{self.file_version}_{channel_nomencalture}.xml"
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


def process_synthetic_1(
    config_keyword="test1",
    z_file_path="",
    test_scale_factor=False,
    simultaneous_regression=False,
    file_version="0.1.0",
    return_collection=False,
    channel_nomenclature="default",
    reload_config=False,
):
    """

    Parameters
    ----------
    config_keyword: str
        "test1", "test1_tfk", this is an argument passed to the create_test_run_config
        as test_case_id.
    z_file_path: str or path
        Where the z-file will be output
    test_scale_factor: bool
        If true, will assign scale factors to the channels
    simultaneous_regression: bool
        If True will do regression all outut channels in one step, rather than the
        usual, channel-by-channel method
    file_version: str
        one of ["0.1.0", "0.2.0"]
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
        scale_factors = {
            "ex": 10.0,
            "ey": 3.0,
            "hx": 6.0,
            "hy": 5.0,
            "hz": 100.0,
        }
        tfk_dataset.df["channel_scale_factors"].at[0] = scale_factors
    else:
        tfk_dataset.df.drop(columns=["channel_scale_factors"], inplace=True)

    processing_config = create_test_run_config(
        config_keyword, tfk_dataset, channel_nomenclature=channel_nomenclature
    )
    # Relates to issue #172
    # reload_config = True
    # if reload_config:
    #     from mt_metadata.transfer_functions.processing.aurora import Processing
    #     p = Processing()
    #     config_path = pathlib.Path("config")
    #     json_fn = config_path.joinpath(processing_config.json_fn())
    #     p.from_json(json_fn)

    if simultaneous_regression:
        for decimation in processing_config.decimations:
            decimation.estimator.estimate_per_channel = False

    tf_result = process_mth5(
        processing_config,
        tfk_dataset=tfk_dataset,
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
                figures_path=AURORA_RESULTS_PATH,
            )
    return tf_result


def process_synthetic_2(
    force_make_mth5=True, z_file_path=None, save_fc=False, file_version="0.2.0"
):
    """"""
    station_id = "test2"
    mth5_path = create_test2_h5(
        force_make_mth5=force_make_mth5, file_version=file_version
    )
    mth5_paths = [
        mth5_path,
    ]
    run_summary = RunSummary()
    run_summary.from_mth5s(mth5_paths)
    tfk_dataset = KernelDataset()
    tfk_dataset.from_run_summary(run_summary, station_id)
    processing_config = create_test_run_config(station_id, tfk_dataset)
    for decimation_level in processing_config.decimations:
        if save_fc:
            decimation_level.save_fcs = True
            decimation_level.save_fcs_type = "h5"
        decimation_level.window.type = "boxcar"
        # decimation_level.save_fcs_type = "csv"
    tfc = process_mth5(
        processing_config,
        tfk_dataset=tfk_dataset,
        z_file_path=z_file_path,
    )
    return tfc


def process_synthetic_1r2(
    config_keyword="test1r2",
    channel_nomenclature="default",
    return_collection=False,
):
    mth5_path = create_test12rr_h5(channel_nomenclature=channel_nomenclature)
    mth5_paths = [
        mth5_path,
    ]
    run_summary = RunSummary()
    run_summary.from_mth5s(mth5_paths)
    tfk_dataset = KernelDataset()
    tfk_dataset.from_run_summary(run_summary, "test1", "test2")
    processing_config = create_test_run_config(
        config_keyword, tfk_dataset, channel_nomenclature=channel_nomenclature
    )
    tfc = process_mth5(
        processing_config,
        tfk_dataset=tfk_dataset,
        return_collection=return_collection,
    )
    return tfc


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
