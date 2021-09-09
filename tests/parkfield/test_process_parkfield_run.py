from pathlib import Path

from aurora.config.processing_config import RunConfig
from aurora.pipelines.process_mth5 import process_mth5_run
from aurora.transfer_function.plot.comparison_plots import compare_two_z_files

# from aurora.sandbox.io_helpers.zfile_murphy import read_z_file

from make_parkfield_mth5 import test_make_parkfield_mth5
from make_processing_configs import create_run_test_config

from helpers import AURORA_RESULTS_PATH
from helpers import EMTF_RESULTS_PATH


def test_processing(z_file_path=None):
    processing_run_cfg = create_run_test_config()

    config = RunConfig()
    config.from_json(processing_run_cfg)
    mth5_path = Path(config.mth5_path)

    # Ensure there is an mth5 to process
    if not mth5_path.exists():
        test_make_parkfield_mth5()
    z_file_path = AURORA_RESULTS_PATH.joinpath("pkd.zss")
    run_id = "001"
    show_plot = False
    tf_collection = process_mth5_run(
        processing_run_cfg,
        run_id,
        mth5_path=mth5_path,
        units="SI",
        show_plot=show_plot,
        z_file_path=z_file_path,
    )
    # TODO: SYNCH PERIOD AXES
    # from aurora.transfer_function.emtf_z_file_helpers import
    # merge_tf_collection_to_match_z_file
    # # for i_dec in tf_collection.tf_dict.keys():
    # #     tf = tf_collection.tf_dict[i_dec]
    # #
    # #     aurora_rho = tf.rho[:, 0]
    # #     plot_rho(axs[0], tf.periods, aurora_rho, marker="o", color=color_cyc[
    # #         i_dec], linestyle="None", label=f"aurora {i_dec} xy",
    # #              markersize=markersize)
    # #     aurora_rho = tf.rho[:, 1]
    # #     plot_rho(axs[0], tf.periods, aurora_rho, marker="o", color=color_cyc[
    # #         i_dec], linestyle="None", label=f"aurora {i_dec} yx",
    # #              markersize=markersize)
    # # plt.show()
    # plt.show()
    return tf_collection


def main():
    z_file_path = AURORA_RESULTS_PATH.joinpath("pkd.zss")
    test_processing(z_file_path)

    # COMPARE WITH ARCHIVED Z-FILE
    auxilliary_z_file = EMTF_RESULTS_PATH.joinpath("PKD_272_00.zrr")
    compare_two_z_files(
        z_file_path,
        auxilliary_z_file,
        label1="aurora",
        label2="emtf",
        scale_factor1=1e-6,
        out_file="SS.png",
        markersize=3,
    )


if __name__ == "__main__":
    main()
