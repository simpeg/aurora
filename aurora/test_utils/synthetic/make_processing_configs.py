"""
    This module contains methods for generating processing config objects that are
    used in aurora's tests of processing synthetic data.
"""

from aurora.config import BANDS_DEFAULT_FILE
from aurora.config import BANDS_256_26_FILE
from aurora.config.config_creator import ConfigCreator
from aurora.test_utils.synthetic.paths import SyntheticTestPaths
from loguru import logger

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
        Description of the dataset to process
    matlab_or_fortran: str
        "", "matlab", "fortran"
    save: str
        if this has the value "json" a copy of the processing config will be written to a json file
        The json file name is p.json_fn() with p the processing config object.
    channel_nomenclature: str
        A label for the channel nomenclature.  This should be one of the values in
        mt_metadata/transfer_functions/processing/aurora/standards/channel_nomenclatures.json
        currently ["default", "lemi12", "lemi34", "phoenix123", "musgraves",]

    Returns
    -------
    p: aurora.config.metadata.processing.Processing
        The processing config object


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
    Intended to test that processing config can be stored as a json, then
    reloaded from json and is equal.

    WORK IN PROGRESS -- see mt_metadata Issue #222

    Development Notes:
    TODO: This test should be completed and moved into tests.
    The json does not load into an mt_metadata object.
    The problem seems to be that at the run-level of the processing config there is an
    intention to allow for multiple time-periods. This is reasonable, consider a station
    running for several months, we may want to only process data from certain chunks
    of the time series.
    However, the time period reader does not seem to work as expected.
    A partial fix is on fix_issue_222 branch of mt_metadata

    Related to issue #172


    Returns
    -------

    """
    # import pandas as pd
    from mt_metadata.transfer_functions.processing.aurora import Processing
    from mtpy.processing import RunSummary, KernelDataset

    # Specify path to mth5
    data_path = MTH5_PATH.joinpath("test1.h5")
    if not data_path.exists():
        logger.error("You need to run make_mth5_from_asc.py")
        raise Exception
    mth5_paths = [
        data_path,
    ]
    run_summary = RunSummary()
    run_summary.from_mth5s(mth5_paths)
    tfk_dataset = KernelDataset()
    station_id = "test1"
    tfk_dataset.from_run_summary(run_summary, station_id)

    processing_config = create_test_run_config(
        station_id, tfk_dataset, save="json"
    )
    p = Processing()
    json_fn = CONFIG_PATH.joinpath(processing_config.json_fn())
    p.from_json(json_fn)
    logger.info("Assert equal needed here")
    # This fails (July 22, 2024)
    # assert p == processing_config

    # This should be true, but its false
    # p.stations.local.runs == processing_config.stations.local.runs
    # p.stations.local.runs[0] == processing_config.stations.local.runs[0]

    """
    Debugging Notes:
    Once the updated parsing from #222 is applied, the next problem is that the object that was
    read-back from json has two dicts in its time periods:
    p.stations.local.runs[0].time_periods
    [{'time_period': {'end': '1980-01-01T11:06:39+00:00', 'start': '1980-01-01T00:00:00+00:00'}},
     {'time_period': {'end': '1980-01-01T11:06:39+00:00', 'start': '1980-01-01T00:00:00+00:00'}}]
    processing_config.stations.local.runs[0].time_periods
    [{
        "time_period": {
            "end": "1980-01-01T11:06:39+00:00",
            "start": "1980-01-01T00:00:00+00:00"
        }
    }]
    """
    return


def main():
    """Allow the module to be called from the command line"""
    pass
    # TODO: fix test_to_from_json and put in tests.
    #  - see issue #222 in mt_metadata.
    test_to_from_json()
    # create_test_run_config("test1", df)
    # create_test_run_config("test2")
    # create_test_run_config("test1r2")
    # create_test_run_config("test2r1")


if __name__ == "__main__":
    main()
