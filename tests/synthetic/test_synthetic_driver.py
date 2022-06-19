from aurora.test_utils.synthetic.make_mth5_from_asc import create_test1_h5
from aurora.test_utils.synthetic.make_mth5_from_asc import create_test2_h5
from aurora.test_utils.synthetic.make_mth5_from_asc import create_test12rr_h5
from aurora.test_utils.synthetic.make_processing_configs import create_test_run_config
from aurora.test_utils.synthetic.paths import AURORA_RESULTS_PATH
from aurora.test_utils.synthetic.paths import CONFIG_PATH
from aurora.test_utils.synthetic.processing_helpers import process_sythetic_data
from aurora.tf_kernel.dataset import Dataset as TFKDataset
from aurora.tf_kernel.helpers import extract_run_summaries_from_mth5s



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


def process_synthetic_1(z_file_path="", test_scale_factor=False,
                        test_simultaneous_regression=False):
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
    tfc: TransferFunctionCollection
        Should change so that it is mt_metadata.TF (see Issue #143)
    """
    mth5_path = create_test1_h5()
    super_summary = extract_run_summaries_from_mth5s([mth5_path,])
    dataset_df = super_summary[super_summary.station_id=="test1"]
    dataset_df["remote"] = False
    tfk_dataset = TFKDataset()
    tfk_dataset.df = dataset_df

    #Test that channel_scale_factors column is optional
    if test_scale_factor:
        scale_factors = {'ex': 10.0, 'ey': 3.0, 'hx': 6.0, 'hy': 5.0, 'hz': 100.0}
        dataset_df["channel_scale_factors"].at[0] = scale_factors
    else:
        dataset_df.drop(columns=["channel_scale_factors"], inplace=True)

    processing_config = create_test_run_config("test1", dataset_df)

    if test_simultaneous_regression:
        for decimation in processing_config.decimations:
            decimation.estimator.estimate_per_channel=False

    tfc = process_sythetic_data(processing_config,
                                tfk_dataset,
                                z_file_path=z_file_path)

    z_figure_name = z_file_path.name.replace("zss", "png")
    for xy_or_yx in ["xy", "yx"]:
        ttl_str = f"{xy_or_yx} component, test_scale_factor = {test_scale_factor}"
        out_png_name = f"{xy_or_yx}_{z_figure_name}"
        tfc.rho_phi_plot(
            xy_or_yx=xy_or_yx,
            ttl_str=ttl_str,
            show=False,
            figure_basename=out_png_name,
            figure_path=AURORA_RESULTS_PATH
        )
    return tfc


def process_synthetic_2():
    mth5_path = create_test2_h5()
    super_summary = extract_run_summaries_from_mth5s([mth5_path,])
    dataset_df = super_summary[super_summary.station_id=="test2"]
    dataset_df["remote"] = False
    tfk_dataset = TFKDataset()
    tfk_dataset.df = dataset_df
    processing_config = create_test_run_config("test2", dataset_df)
    tfc = process_sythetic_data(processing_config, tfk_dataset)
    return tfc


def process_synthetic_rr12():
    mth5_path = create_test12rr_h5()
    mth5_paths = [mth5_path,]
    super_summary = extract_run_summaries_from_mth5s([mth5_path,])
    dataset_df = super_summary
    dataset_df["remote"] = [False, True]
    tfk_dataset = TFKDataset()
    tfk_dataset.df = dataset_df
    processing_config = create_test_run_config("test1r2", dataset_df)
    tfc = process_sythetic_data(processing_config, tfk_dataset)


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

    z_file_path=AURORA_RESULTS_PATH.joinpath("syn1.zss")
    tfc = process_synthetic_1(z_file_path=z_file_path)
    z_file_path=AURORA_RESULTS_PATH.joinpath("syn1_scaled.zss")
    tfc = process_synthetic_1(z_file_path=z_file_path, test_scale_factor=True)
    z_file_path=AURORA_RESULTS_PATH.joinpath("syn1_simultaneous_estimate.zss")
    tfc = process_synthetic_1(z_file_path=z_file_path,
                              test_simultaneous_regression=True)
    tfc = process_synthetic_2()
    tfc = process_synthetic_rr12()


def main():
    # test_create_mth5()
    # test_create_run_configs()
    test_process_mth5()


if __name__ == "__main__":
    main()
