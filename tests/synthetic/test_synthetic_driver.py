from aurora.general_helper_functions import TEST_PATH
from aurora.pipelines.process_mth5 import process_mth5_run
from aurora.test_utils.synthetic.make_mth5_from_asc import create_test1_h5
from aurora.test_utils.synthetic.make_mth5_from_asc import create_test2_h5
from aurora.test_utils.synthetic.make_mth5_from_asc import create_test12rr_h5
from aurora.test_utils.synthetic.make_processing_configs_new import \
    create_test_run_config
from aurora.test_utils.synthetic.processing_helpers import process_sythetic_data
from aurora.tf_kernel.dataset import DatasetDefinition
from aurora.tf_kernel.helpers import extract_run_summaries_from_mth5s

CONFIG_PATH = TEST_PATH.joinpath("synthetic", "config")




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
#     process_mth5_run(test_config, run_id, units="MT")
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
#     process_mth5_run(test_config, run_id, units="MT")


def process_synthetic_1():
    """

    Returns
    -------
    tfc: TransferFunctionCollection
        Should change so that it is mt_metadata.TF (see Issue #143)
    """
    mth5_path = create_test1_h5()
    super_summary = extract_run_summaries_from_mth5s([mth5_path,])
    dataset_df = super_summary[super_summary.station_id=="test1"]
    dataset_df["remote"] = False
    dataset_definition = DatasetDefinition()
    dataset_definition.df = dataset_df
    processing_config = create_test_run_config("test1", dataset_df)
    tfc = process_sythetic_data(processing_config, dataset_definition)
    return tfc


def process_synthetic_2():
    mth5_path = create_test2_h5()
    super_summary = extract_run_summaries_from_mth5s([mth5_path,])
    dataset_df = super_summary[super_summary.station_id=="test2"]
    dataset_df["remote"] = False
    dataset_definition = DatasetDefinition()
    dataset_definition.df = dataset_df
    processing_config = create_test_run_config("test2", dataset_df)
    tfc = process_sythetic_data(processing_config, dataset_definition)
    return tfc


def process_synthetic_rr12():
    mth5_path = create_test12rr_h5()
    mth5_paths = [mth5_path,]
    super_summary = extract_run_summaries_from_mth5s([mth5_path,])
    dataset_df = super_summary
    dataset_df["remote"] = [False, True]
    dataset_definition = DatasetDefinition()
    dataset_definition.df = dataset_df
    processing_config = create_test_run_config("test1r2", dataset_df)
    tfc = process_sythetic_data(processing_config, dataset_definition)


def test_process_mth5():
    """
    Runs several synthetic processing tests from config creation to tf_collection.
    TODO: Modify these tests so that they output tfs using mt_metadata transfer function
    Returns
    -------

    """
    # process_synthetic_1_underdetermined()
    # process_synthetic_1_with_nans()
    tfc = process_synthetic_1()
    tfc = process_synthetic_2()
    tfc = process_synthetic_rr12()


def main():
    # test_create_mth5()
    # test_create_run_configs()
    test_process_mth5()


if __name__ == "__main__":
    main()
