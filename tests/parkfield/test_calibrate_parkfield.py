from aurora.time_series.windowing_scheme import WindowingScheme
from mth5.mth5 import MTH5
from aurora.test_utils.parkfield.calibration_helpers import (
    parkfield_sanity_check,
)
from aurora.test_utils.parkfield.make_parkfield_mth5 import ensure_h5_exists
from aurora.test_utils.parkfield.path_helpers import PARKFIELD_PATHS


def validate_bulk_spectra_have_correct_units(run_obj, run_ts_obj, show_spectra=False):
    """

    Parameters
    ----------
    run_obj: mth5.groups.master_station_run_channel.RunGroup
        /Survey/Stations/PKD/001:
        ====================
            --> Dataset: ex
            .................
            --> Dataset: ey
            .................
            --> Dataset: hx
            .................
            --> Dataset: hy
            .................
            --> Dataset: hz
            .................
    run_ts_obj: mth5.timeseries.run_ts.RunTS
        RunTS Summary:
            Station:     PKD
            Run:         001
            Start:       2004-09-28T00:00:00+00:00
            End:         2004-09-28T01:59:59.950000+00:00
            Sample Rate: 40.0
            Components:  ['ex', 'ey', 'hx', 'hy', 'hz']
    show_spectra: bool
        controls whether plots flash to screen in parkfield_sanity_check

    Returns
    -------

    """
    windowing_scheme = WindowingScheme(
        taper_family="hamming",
        num_samples_window=run_ts_obj.dataset.time.shape[0],  # 288000
        num_samples_overlap=0,
        sample_rate=run_ts_obj.sample_rate,  # 40.0 sps
    )
    windowed_obj = windowing_scheme.apply_sliding_window(
        run_ts_obj.dataset, dt=1.0 / run_ts_obj.sample_rate
    )
    tapered_obj = windowing_scheme.apply_taper(windowed_obj)

    fft_obj = windowing_scheme.apply_fft(tapered_obj)
    show_response_curves = False

    parkfield_sanity_check(
        fft_obj,
        run_obj,
        figures_path=PARKFIELD_PATHS["aurora_results"],
        show_response_curves=show_response_curves,
        show_spectra=show_spectra,
        include_decimation=False,
    )
    return


def test():
    import logging

    logging.getLogger("matplotlib.font_manager").disabled = True
    logging.getLogger("matplotlib.ticker").disabled = True

    run_id = "001"
    station_id = "PKD"
    h5_path = ensure_h5_exists()
    m = MTH5(file_version="0.1.0")
    m.open_mth5(h5_path, mode="r")
    run_obj = m.get_run(station_id, run_id)
    run_ts_obj = run_obj.to_runts()
    validate_bulk_spectra_have_correct_units(run_obj, run_ts_obj, show_spectra=True)
    m.close_mth5()


def main():
    test()


if __name__ == "__main__":
    main()
