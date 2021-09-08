from aurora.general_helper_functions import FIGURES_PATH
from aurora.general_helper_functions import TEST_PATH
from aurora.time_series.windowing_scheme import WindowingScheme
from mth5.mth5 import MTH5
from calibration_helpers import parkfield_sanity_check


def validate_bulk_spectra_have_correct_units(run_obj, run_ts_obj):
    windowing_scheme = WindowingScheme(
        taper_family="hamming",
        num_samples_window=288000,
        num_samples_overlap=0,
        sampling_rate=40.0,
    )
    windowed_obj = windowing_scheme.apply_sliding_window(run_ts_obj.dataset)
    tapered_obj = windowing_scheme.apply_taper(windowed_obj)

    fft_obj = windowing_scheme.apply_fft(tapered_obj)
    show_response_curves = True
    show_spectra = False

    parkfield_sanity_check(
        fft_obj,
        run_obj,
        figures_path=FIGURES_PATH,
        show_response_curves=show_response_curves,
        show_spectra=show_spectra,
        include_decimation=True,
    )


def test():
    run_id = "001"
    station_id = "PKD"
    parkfield_h5_path = TEST_PATH.joinpath("parkfield", "data", "pkd_test_00.h5")
    if not parkfield_h5_path.exists():
        from make_parkfield_mth5 import test_make_parkfield_mth5

        test_make_parkfield_mth5()
    m = MTH5()
    m.open_mth5(parkfield_h5_path, mode="r")
    run_obj = m.get_run(station_id, run_id)
    run_ts_obj = run_obj.to_runts()
    validate_bulk_spectra_have_correct_units(run_obj, run_ts_obj)


def main():
    test()


if __name__ == "__main__":
    main()
