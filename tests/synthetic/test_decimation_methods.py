"""
    This is a test to confirm that mth5's decimation method returns the same default values as aurora's prototype decimate.

    TODO: add tests from aurora issue #363 in this module
"""

from aurora.pipelines.time_series_helpers import prototype_decimate
from aurora.test_utils.synthetic.make_processing_configs import (
    create_test_run_config,
)
from loguru import logger
from mth5.data.make_mth5_from_asc import create_test1_h5
from mth5.mth5 import MTH5
from mth5.helpers import close_open_files
from mth5.processing import RunSummary, KernelDataset

import numpy as np


def test_decimation_methods_agree():
    """
    Get some synthetic time series and check that the decimation results are
    equal to calling the mth5 built-in run_xrts.sps_filters.decimate.

    TODO: More testing could be added for downsamplings that are not integer factors.

    """
    close_open_files()
    mth5_path = create_test1_h5()
    mth5_paths = [
        mth5_path,
    ]

    run_summary = RunSummary()
    run_summary.from_mth5s(mth5_paths)
    tfk_dataset = KernelDataset()
    station_id = "test1"
    run_id = "001"
    tfk_dataset.from_run_summary(run_summary, station_id)

    processing_config = create_test_run_config(station_id, tfk_dataset)

    mth5_obj = MTH5(file_version="0.1.0")
    mth5_obj.open_mth5(mth5_path, mode="a")
    decimated_ts = {}

    for dec_level_id, dec_config in enumerate(processing_config.decimations):
        decimated_ts[dec_level_id] = {}
        if dec_level_id == 0:
            run_obj = mth5_obj.get_run(station_id, run_id, survey=None)
            run_ts = run_obj.to_runts(start=None, end=None)
            run_xrds = run_ts.dataset
            decimated_ts[dec_level_id]["run_xrds"] = run_xrds
            current_sample_rate = run_obj.metadata.sample_rate

        if dec_level_id > 0:
            run_xrds = decimated_ts[dec_level_id - 1]["run_xrds"]
            target_sample_rate = current_sample_rate / (dec_config.decimation.factor)

            decimated_1 = prototype_decimate(dec_config.decimation, run_xrds)
            decimated_2 = run_xrds.sps_filters.decimate(
                target_sample_rate=target_sample_rate
            )

            difference = decimated_2 - decimated_1
            logger.info(len(difference.time))
            assert np.isclose(difference.to_array(), 0).all()

            logger.info("prototype decimate aurora method agrees with mth5 decimate")
            decimated_ts[dec_level_id]["run_xrds"] = decimated_1
            current_sample_rate = target_sample_rate
    return


def main():
    test_decimation_methods_agree()


if __name__ == "__main__":
    main()
