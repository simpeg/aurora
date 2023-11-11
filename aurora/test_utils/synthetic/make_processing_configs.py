from aurora.config import BANDS_DEFAULT_FILE
from aurora.config import BANDS_256_26_FILE
from aurora.config.config_creator import ConfigCreator
from aurora.test_utils.synthetic.paths import SyntheticTestPaths

synthetic_test_paths = SyntheticTestPaths()
CONFIG_PATH = synthetic_test_paths.config_path
MTH5_PATH = synthetic_test_paths.mth5_path


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
        Must be in ["test1", "test2", "test1r2", "test2r1", "test1_tfk", "test1r2_tfk"]
    kernel_dataset: aurora.transfer_function.kernel_dataset.KernelDataset
    matlab_or_fortran: str
        "", "matlab", "fortran"

    Returns
    -------


    """
    estimation_engine = "RME"
    local_station_id = test_case_id
    remote_station_id = ""

    if test_case_id in ["test1r2", "test1r2_tfk"]:
        estimation_engine = "RME_RR"
        local_station_id = "test1"
        remote_station_id = "test2"
    if test_case_id == "test2r1":
        estimation_engine = "RME_RR"
        local_station_id = "test2"
        remote_station_id = "test1"

    if matlab_or_fortran == "matlab":
        emtf_band_setup_file = BANDS_256_26_FILE
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

    cc = ConfigCreator()

    if test_case_id in ["test1", "test2"]:
        p = cc.create_from_kernel_dataset(
            kernel_dataset,
            emtf_band_file=emtf_band_setup_file,
            num_samples_window=num_samples_window,
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
            num_samples_window=num_samples_window,
        )
        p.id = config_id
        p.channel_nomenclature.keyword = channel_nomenclature
        p.set_default_input_output_channels()
        p.set_default_reference_channels()

    elif test_case_id in ["test1_tfk", "test1r2_tfk"]:
        from aurora.general_helper_functions import BAND_SETUP_PATH

        emtf_band_setup_file = BAND_SETUP_PATH.joinpath("bs_six_level.cfg")
        if test_case_id == "test1r2_tfk":
            config_id = f"{config_id}-RR{remote_station_id}_tfk"
        p = cc.create_from_kernel_dataset(
            kernel_dataset,
            emtf_band_file=emtf_band_setup_file,
            num_samples_window=num_samples_window,
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
        filename = CONFIG_PATH.joinpath(p.json_fn())
        p.save_as_json(filename=filename)

    return p


def test_to_from_json():
    """
    Test related to issue #172
    This is deprecated in its current form, but should be modified to save the json
    from the processing object (not the config class)
    Trying to save to json and then read back a Processing object

    Start by manually creating the dataset_df for syntehtic test1


    Returns
    -------

    """
    import pandas as pd
    from mt_metadata.transfer_functions.processing.aurora import Processing
    from aurora.pipelines.run_summary import RunSummary
    from aurora.transfer_function.kernel_dataset import KernelDataset

    # Specify path to mth5
    data_path = MTH5_PATH.joinpath("test1.h5")
    if not data_path.exists():
        print("You need to run make_mth5_from_asc.py")
        raise Exception
    mth5_paths = [
        data_path,
    ]
    run_summary = RunSummary()
    run_summary.from_mth5s(mth5_paths)
    tfk_dataset = KernelDataset()
    station_id = "test1"
    tfk_dataset.from_run_summary(run_summary, station_id)

    processing_config = create_test_run_config(station_id, tfk_dataset, save="json")
    p = Processing()
    json_fn = CONFIG_PATH.joinpath(processing_config.json_fn())
    p.from_json(json_fn)
    print("Assert equal needed here")
    return


def main():
    test_to_from_json()
    # create_test_run_config("test1", df)
    # create_test_run_config("test2")
    # create_test_run_config("test1r2")
    # create_test_run_config("test2r1")


if __name__ == "__main__":
    main()
