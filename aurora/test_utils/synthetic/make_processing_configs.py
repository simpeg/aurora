from aurora.config import BANDS_DEFAULT_FILE
from aurora.config import BANDS_256_FILE
from aurora.config.config_creator import ConfigCreator
from aurora.test_utils.synthetic.paths import CONFIG_PATH
from aurora.test_utils.synthetic.paths import DATA_PATH


def create_test_run_config(
    test_case_id,
    kernel_dataset,
    matlab_or_fortran="",
    save="json",
    channel_nomenclature="default",
):
    """
    Use config creator to generate a processing config file for the synthetic data.

    Parameters
    ----------
    test_case_id: string
        "test1", "test2", "test12rr"
    matlab_or_fortran: str
        "", "matlab", "fortran"

    Returns
    -------


    """
    estimation_engine = "RME"
    local_station_id = test_case_id
    remote_station_id = ""

    if test_case_id == "test1r2":
        estimation_engine = "RME_RR"
        local_station_id = "test1"
        remote_station_id = "test2"
    if test_case_id == "test2r1":
        estimation_engine = "RME_RR"
        local_station_id = "test2"
        remote_station_id = "test1"

    if matlab_or_fortran == "matlab":
        emtf_band_setup_file = BANDS_256_FILE
        num_samples_window = 256
        num_samples_overlap = 64
        config_id = f"{local_station_id}-{matlab_or_fortran}"
    elif matlab_or_fortran == "fortran":
        emtf_band_setup_file = BANDS_DEFAULT_FILE
        num_samples_window = 128
        num_samples_overlap = 32
        config_id = f"{local_station_id}-{matlab_or_fortran}"
    else:
        emtf_band_setup_file = BANDS_DEFAULT_FILE
        num_samples_window = 128
        num_samples_overlap = 32
        config_id = f"{local_station_id}"

    cc = ConfigCreator(config_path=CONFIG_PATH)

    if test_case_id in ["test1", "test2"]:
        p = cc.create_from_kernel_dataset(
            kernel_dataset,
            emtf_band_file=emtf_band_setup_file,
        )
        p.id = config_id
        p.channel_nomenclature.keyword = channel_nomenclature
        p.set_default_input_output_channels()
        p.drop_reference_channels()

    elif test_case_id in ["test2r1", "test1r2"]:
        config_id = f"{config_id}-RR{remote_station_id}"
        p = cc.create_from_kernel_dataset(
            kernel_dataset,
            emtf_band_file=emtf_band_setup_file,
        )
        p.id = config_id
        p.channel_nomenclature.keyword = channel_nomenclature
        p.set_default_input_output_channels()
        p.set_default_reference_channels()

    for decimation in p.decimations:
        decimation.estimator.engine = estimation_engine
        decimation.window.type = "hamming"
        decimation.window.num_samples = num_samples_window
        decimation.window.overlap = num_samples_overlap
        decimation.regression.max_redescending_iterations = 2

    if save == "json":
        cc.to_json(p)

    return p


def test_to_from_json():
    """
    Test related to issue #172
    Trying to save to json and then read back a Processing object

    Start by manually creating the dataset_df for syntehtic test1


    Returns
    -------

    """
    import pandas as pd
    from aurora.config.metadata import Processing
    from aurora.tf_kernel.dataset import RUN_SUMMARY_COLUMNS

    # Specify path to mth5
    data_path = DATA_PATH.joinpath("test1.h5")
    if not data_path.exists():
        print("You need to run make_mth5_from_asc.py")
        raise Exception

    # create run summary
    df = pd.DataFrame(columns=RUN_SUMMARY_COLUMNS)
    df["station_id"] = [
        "test1",
    ]
    df["run_id"] = [
        "001",
    ]
    df["remote"] = False
    df["output_channels"] = [
        ["ex", "ey", "hz"],
    ]
    df["input_channels"] = [
        ["hx", "hy"],
    ]
    df["start"] = [pd.Timestamp("1980-01-01 00:00:00")]
    df["end"] = [pd.Timestamp("1980-01-01 11:06:39")]
    df["sample_rate"] = [1.0]
    df["mth5_path"] = str(data_path)

    # create processing config, and save to a json file
    p = create_test_run_config("test1", df, save="json")

    # test can read json back into Processing obj:
    json_fn = CONFIG_PATH.joinpath(p.json_fn())
    p2 = Processing()
    p2.from_json(json_fn)
    return


def main():
    test_to_from_json()
    # create_test_run_config("test1", df)
    # create_test_run_config("test2")
    # create_test_run_config("test1r2")
    # create_test_run_config("test2r1")


if __name__ == "__main__":
    main()
