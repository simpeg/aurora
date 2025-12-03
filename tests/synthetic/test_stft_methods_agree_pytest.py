"""Pytest translation of test_stft_methods_agree.py

This test confirms that the internal aurora stft method returns the same
array as scipy.signal.spectrogram
"""

import numpy as np
from mth5.helpers import close_open_files
from mth5.mth5 import MTH5
from mth5.processing import KernelDataset, RunSummary
from mth5.processing.spectre.stft import run_ts_to_stft_scipy

from aurora.pipelines.time_series_helpers import prototype_decimate
from aurora.test_utils.synthetic.make_processing_configs import create_test_run_config
from aurora.time_series.spectrogram_helpers import run_ts_to_stft


def test_stft_methods_agree(worker_safe_test1_h5):
    """Test that aurora STFT and scipy STFT produce identical results.

    The answer is "mostly yes", under two conditions:
    1. scipy.signal.spectrogram does not innately support an extra linear
       detrending to be applied _after_ tapering.
    2. We do not wish to apply "per-segment" prewhitening as is done in some
       variations of EMTF.

    Excluding these, we get numerically identical results.
    """
    close_open_files()
    mth5_path = worker_safe_test1_h5

    run_summary = RunSummary()
    run_summary.from_mth5s([mth5_path])
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
            local_run_xrts = prototype_decimate(dec_config.decimation, local_run_xrts)

        dec_config.stft.per_window_detrend_type = "constant"
        local_spectrogram = run_ts_to_stft(dec_config, local_run_xrts)
        local_spectrogram2 = run_ts_to_stft_scipy(dec_config, local_run_xrts)
        stft_difference = (
            local_spectrogram.dataset - local_spectrogram2.dataset
        ).to_array()

        # drop dc term
        stft_difference = stft_difference.where(
            stft_difference.frequency > 0, drop=True
        )

        assert np.isclose(stft_difference, 0).all()
