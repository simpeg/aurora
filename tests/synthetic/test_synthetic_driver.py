from aurora.pipelines.run_summary import RunSummary
from aurora.test_utils.synthetic.make_mth5_from_asc import create_test1_h5
from aurora.test_utils.synthetic.make_mth5_from_asc import create_test2_h5
from aurora.test_utils.synthetic.make_mth5_from_asc import create_test12rr_h5
from aurora.test_utils.synthetic.make_processing_configs import create_test_run_config
from aurora.test_utils.synthetic.paths import AURORA_RESULTS_PATH

# from aurora.test_utils.synthetic.paths import CONFIG_PATH
from aurora.test_utils.synthetic.processing_helpers import process_sythetic_data
from aurora.transfer_function.kernel_dataset import KernelDataset

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


def process_synthetic_1(
    z_file_path="",
    test_scale_factor=False,
    test_simultaneous_regression=False,
    file_version="0.1.0",
    return_collection=True,
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
    mth5_path = create_test1_h5(file_version=file_version)
    run_summary = RunSummary()
    run_summary.from_mth5s(
        [
            mth5_path,
        ]
    )
    # run_summary.drop_runs_shorter_than(100000)
    tfk_dataset = KernelDataset()
    tfk_dataset.from_run_summary(run_summary, "test1")

    # Test that channel_scale_factors column is optional
    if test_scale_factor:
        scale_factors = {"ex": 10.0, "ey": 3.0, "hx": 6.0, "hy": 5.0, "hz": 100.0}
        tfk_dataset.df["channel_scale_factors"].at[0] = scale_factors
    else:
        tfk_dataset.df.drop(columns=["channel_scale_factors"], inplace=True)

    processing_config = create_test_run_config("test1", tfk_dataset.df)

    if test_simultaneous_regression:
        for decimation in processing_config.decimations:
            decimation.estimator.estimate_per_channel = False

    tf_result = process_sythetic_data(
        processing_config,
        tfk_dataset,
        z_file_path=z_file_path,
        return_collection=return_collection,
    )

    z_figure_name = z_file_path.name.replace("zss", "png")
    if return_collection:
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


def process_synthetic_rr12():
    mth5_path = create_test12rr_h5()
    run_summary = RunSummary()
    run_summary.from_mth5s(
        [
            mth5_path,
        ]
    )
    tfk_dataset = KernelDataset()
    tfk_dataset.from_run_summary(run_summary, "test1", "test2")
    processing_config = create_test_run_config("test1r2", tfk_dataset.df)
    tfc = process_sythetic_data(processing_config, tfk_dataset)
    return tfc


def test_process_mth5():
    """
    Runs several synthetic processing tests from config creation to tf_collection.
    TODO: Modify these tests so that they output tfs using mt_metadata transfer function

    2022-05-13: Added a duplicate run of process_synthetic_1, which is intended to
    test the channel_scale_factors in the new mt_metadata processing class.  Expected
    outputs are four .png:
    xy_syn1.png : Shows expected 100 Ohm-m resisitivity
    xy_syn1_scaled.png : Overestimates by 4x for 300 Ohm-m resistivity
    yx_syn1.png : Shows expected 100 Ohm-m resisitivity
    yx_syn1_scaled.png : Underestimates by 4x for 25 Ohm-m resistivity
    These .png are stores in aurora_results folder

    Returns
    -------

    """
    # process_synthetic_1_underdetermined()
    # process_synthetic_1_with_nans()

    z_file_path = AURORA_RESULTS_PATH.joinpath("syn1.zss")
    tf_collection = process_synthetic_1(z_file_path=z_file_path, file_version="0.1.0")
    tf_cls = process_synthetic_1(
        z_file_path=z_file_path, file_version="0.1.0", return_collection=False
    )
    xml_file_base = "syn1_mth5-010.xml"
    xml_file_name = AURORA_RESULTS_PATH.joinpath(xml_file_base)
    tf_cls.write_tf_file(fn=xml_file_name, file_type="emtfxml")
    tf_cls = process_synthetic_1(
        z_file_path=z_file_path, file_version="0.2.0", return_collection=False
    )
    xml_file_base = "syn1_mth5-020.xml"
    xml_file_name = AURORA_RESULTS_PATH.joinpath(xml_file_base)
    tf_cls.write_tf_file(fn=xml_file_name, file_type="emtfxml")
    z_file_path = AURORA_RESULTS_PATH.joinpath("syn1_scaled.zss")
    tf_collection = process_synthetic_1(z_file_path=z_file_path, test_scale_factor=True)
    z_file_path = AURORA_RESULTS_PATH.joinpath("syn1_simultaneous_estimate.zss")
    tf_collection = process_synthetic_1(
        z_file_path=z_file_path, test_simultaneous_regression=True
    )
    tf_collection = process_synthetic_2()
    tf_collection = process_synthetic_rr12()
    return tf_collection


def main():
    # test_create_mth5()
    # test_create_run_configs()
    test_process_mth5()


if __name__ == "__main__":
    main()
