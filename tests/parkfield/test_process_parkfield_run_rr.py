from pathlib import Path

from aurora.config.processing_config import RunConfig
from aurora.pipelines.process_mth5 import process_mth5_run
from aurora.transfer_function.plot.comparison_plots import compare_two_z_files

from make_parkfield_mth5 import test_make_parkfield_hollister_mth5
from make_processing_configs import create_run_test_config_remote_reference

from aurora.test_utils.parkfield.path_helpers import AURORA_RESULTS_PATH
from aurora.test_utils.parkfield.path_helpers import EMTF_RESULTS_PATH


@deprecated(version="0.0.3", reason="new mt_metadata based config")
def test_processing(z_file_path=None):
    # processing_run_cfg = create_run_test_config()

    # processing_run_cfg = create_run_test_config()
    # processing_run_cfg = Path("config", "PKD-RR_SAO_run_config.json")
    # processing_run_cfg = Path("config", "pkd_test-RR_SAO_run_config.json")
    processing_run_cfg = create_run_test_config_remote_reference()
    config = RunConfig()
    config.from_json(processing_run_cfg)
    mth5_path = Path(config.mth5_path)

    # Ensure there is an mth5 to process
    if not mth5_path.exists():
        test_make_parkfield_hollister_mth5()

    run_id = "001"
    show_plot = False
    tf_collection = process_mth5_run(
        processing_run_cfg,
        run_id,
        mth5_path=mth5_path,
        units="MT",
        show_plot=show_plot,
        z_file_path=z_file_path,
    )
    return tf_collection


@deprecated(version="0.0.3", reason="new mt_metadata based config")
def main():
    z_file_path = AURORA_RESULTS_PATH.joinpath("pkd.zrr")
    test_processing(z_file_path)

    # COMPARE WITH ARCHIVED Z-FILE
    auxilliary_z_file = EMTF_RESULTS_PATH.joinpath("PKD_272_00.zrr")
    compare_two_z_files(
        z_file_path,
        auxilliary_z_file,
        label1="aurora",
        label2="emtf",
        scale_factor1=1,
        out_file="RR.png",
        markersize=3,
        rho_ylims=[1e0, 1e3],
        xlims=[0.05, 500],
    )


if __name__ == "__main__":
    main()
