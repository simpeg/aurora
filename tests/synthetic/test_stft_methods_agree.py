"""
See aurora issue #3.  This test confirms that the internal aurora stft
method returns the same array as scipy.signal.spectrogram
"""

import numpy as np

from aurora.pipelines.time_series_helpers import prototype_decimate
from aurora.pipelines.time_series_helpers import run_ts_to_stft
from aurora.pipelines.time_series_helpers import run_ts_to_stft_scipy
from aurora.test_utils.synthetic.make_processing_configs import (
    create_test_run_config,
)

# from mtpy-v2
from mtpy.processing import RunSummary, KernelDataset

from loguru import logger
from mth5.data.make_mth5_from_asc import create_test1_h5
from mth5.mth5 import MTH5
from mth5.helpers import close_open_files


def test_stft_methods_agree():
    """
    The purpose of this method was to check if we could reasonably replace Gary's
    fft with scipy.signal.spectrogram.
    The answer is "mostly yes", under two conditons:
    1. scipy.signal.spectrogram does not inately support an extra linear detrending
    to be applied _after_ tapering.
    2. We do not wish to apply "per-segment" prewhitening as is done in some
    variations of EMTF.
    excluding this, we get numerically identical results, with basically
    zero-maintenance by using scipy.

    As of 30 Jun 2023, run_ts_to_stft_scipy is never actually used in aurora, except in
    this test.  That will change with the introduction of the FC layer in mth5 which
    will use that method.

    Because run_ts_to_stft_scipy will be used in mth5, we can port the aurora
    processing config to a mth5 FC processing config.  I.e. the dec_config argument to
    run_ts_to_stft can be reformatted so that it is an instance of
    mt_metadata.transfer_functions.processing.fourier_coefficients.decimation.Decimation

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

    for dec_level_id, dec_config in enumerate(processing_config.decimations):

        if dec_level_id == 0:
            run_obj = mth5_obj.get_run(station_id, run_id, survey=None)
            run_ts = run_obj.to_runts(start=None, end=None)
            local_run_xrts = run_ts.dataset
        else:
            local_run_xrts = prototype_decimate(
                dec_config.decimation, local_run_xrts
            )

        dec_config.extra_pre_fft_detrend_type = "constant"
        local_stft_obj = run_ts_to_stft(dec_config, local_run_xrts)
        local_stft_obj2 = run_ts_to_stft_scipy(dec_config, local_run_xrts)
        stft_difference = local_stft_obj - local_stft_obj2
        stft_difference = stft_difference.to_array()

        # drop dc term
        stft_difference = stft_difference.where(
            stft_difference.frequency > 0, drop=True
        )

        assert np.isclose(stft_difference, 0).all()

        logger.info("stft aurora method agrees with scipy.signal.spectrogram")
    return


def main():
    test_stft_methods_agree()


if __name__ == "__main__":
    main()
