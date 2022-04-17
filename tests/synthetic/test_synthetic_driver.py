from aurora.general_helper_functions import TEST_PATH
from aurora.pipelines.process_mth5 import process_mth5_run
from aurora.test_utils.synthetic.make_processing_configs import create_test_run_config
#from aurora.test_utils.synthetic.make_processing_configs_new import
# create_test_run_config

from make_mth5_from_asc import create_test1_h5
from make_mth5_from_asc import create_test2_h5
from make_mth5_from_asc import create_test12rr_h5


CONFIG_PATH = TEST_PATH.joinpath("synthetic", "config")


def test_create_mth5():
    create_test1_h5()
    create_test2_h5()
    create_test12rr_h5()


def test_create_run_configs():
    create_test_run_config("test1")
    create_test_run_config("test2")
    create_test_run_config("test1r2")
    create_test_run_config("test2r1")


def process_synthetic_1_underdetermined():
    """
    Just like process_synthetic_1, but the window is ridiculously long so that we
    encounter the underdetermined problem. We actually pass that test but in testing
    I found that at the next band over, which has more data because there are multipe
    FCs the sigma in TRME comes out as negative. see issue #4 and issue #55.
    Returns
    -------

    """
    test_config = CONFIG_PATH.joinpath("test1_run_config_underdetermined.json")
    # test_config = Path("config", "test1_run_config_underdetermined.json")
    run_id = "001"
    process_mth5_run(test_config, run_id, units="MT")


def process_synthetic_1_with_nans():
    """

    Returns
    -------

    """
    test_config = CONFIG_PATH.joinpath("test1_run_config_nan.json")
    #    test_config = Path("config", "test1_run_config_nan.json")
    run_id = "001"
    process_mth5_run(test_config, run_id, units="MT")


def process_synthetic_1():
    """

    Returns
    -------
    tfc: TransferFunctionCollection
        Should change so that it is mt_metadata.TF (see Issue #143)
    """
    test_config = create_test_run_config("test1")
    run_id = "001"
    tfc = process_mth5_run(test_config, run_id, units="MT")
    return tfc


def process_synthetic_2():
    test_config = create_test_run_config("test2")
    run_id = "001"
    process_mth5_run(test_config, run_id, units="MT")


def process_synthetic_rr12():
    test_config = create_test_run_config("test1r2")
    from synthetic_station_config import make_station_01_config_dict
    station_01_params = make_station_01_config_dict()
    run_id = station_01_params["run_id"]
    process_mth5_run(test_config, run_id, units="MT", show_plot=False)


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
    process_synthetic_2()
    process_synthetic_rr12()


def main():
    test_create_mth5()
    test_create_run_configs()
    test_process_mth5()


if __name__ == "__main__":
    main()
