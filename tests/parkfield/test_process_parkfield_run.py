from aurora.config import BANDS_DEFAULT_FILE
from aurora.config.config_creator import ConfigCreator
from aurora.pipelines.process_mth5 import process_mth5
from aurora.pipelines.run_summary import RunSummary
from aurora.test_utils.parkfield.make_parkfield_mth5 import ensure_h5_exists
from aurora.test_utils.parkfield.path_helpers import AURORA_RESULTS_PATH

# from aurora.test_utils.parkfield.path_helpers import CONFIG_PATH
from aurora.test_utils.parkfield.path_helpers import DATA_PATH
from aurora.test_utils.parkfield.path_helpers import EMTF_RESULTS_PATH
from aurora.transfer_function.kernel_dataset import KernelDataset
from aurora.transfer_function.plot.comparison_plots import compare_two_z_files

from mth5.helpers import close_open_files


def test_processing(z_file_path=None, test_clock_zero=False):
    """
    Parameters
    ----------
    z_file_path: str or Path or None
        Where to store zfile

    Returns
    -------
    tf_cls: mt_metadata.transfer_functions.core.TF
        The TF object,

    """
    close_open_files()
    h5_path = ensure_h5_exists()

    run_summary = RunSummary()
    h5s_list = [
        h5_path,
    ]
    run_summary.from_mth5s(h5s_list)
    tfk_dataset = KernelDataset()
    tfk_dataset.from_run_summary(run_summary, "PKD")

    cc = ConfigCreator()
    config = cc.create_from_kernel_dataset(
        tfk_dataset,
        estimator={"engine": "RME"},
        output_channels=["ex", "ey"],
    )

    if test_clock_zero:
        for dec_lvl_cfg in config.decimations:
            dec_lvl_cfg.window.clock_zero_type = test_clock_zero
            if test_clock_zero == "user specified":
                dec_lvl_cfg.window.clock_zero = "2004-09-28 00:00:10+00:00"

    show_plot = False
    tf_cls = process_mth5(
        config,
        tfk_dataset,
        units="MT",
        show_plot=show_plot,
        z_file_path=z_file_path,
    )

    tf_cls.write(fn="emtfxml_test.xml", file_type="emtfxml")
    return tf_cls


def test():
    import logging

    logging.getLogger("matplotlib.font_manager").disabled = True
    logging.getLogger("matplotlib.ticker").disabled = True

    z_file_path = AURORA_RESULTS_PATH.joinpath("pkd.zss")
    tf_cls = test_processing(z_file_path=z_file_path)
    tf_cls.write("pkd_mt_metadata.zss", file_type="zss")
    test_processing(
        z_file_path=z_file_path,
        test_clock_zero="user specified",
    )
    test_processing(z_file_path=z_file_path, test_clock_zero="data start")

    # COMPARE WITH ARCHIVED Z-FILE
    auxilliary_z_file = EMTF_RESULTS_PATH.joinpath("PKD_272_00.zrr")
    if z_file_path.exists():
        compare_two_z_files(
            z_file_path,
            auxilliary_z_file,
            label1="aurora",
            label2="emtf",
            scale_factor1=1,
            out_file="SS.png",
            markersize=3,
            rho_ylims=[1e0, 1e3],
            xlims=[0.05, 500],
        )
    else:
        print("Z-File not found - Parkfield tests failed to generate output")
        print("NCEDC probably not returning data")


if __name__ == "__main__":
    test()
