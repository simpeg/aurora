from aurora.config import BANDS_DEFAULT_FILE
from aurora.config.config_creator import ConfigCreator

from aurora.pipelines.process_mth5 import process_mth5
from aurora.pipelines.run_summary import RunSummary
from aurora.test_utils.parkfield.path_helpers import AURORA_RESULTS_PATH
from aurora.test_utils.parkfield.path_helpers import CONFIG_PATH
from aurora.test_utils.parkfield.path_helpers import DATA_PATH
from aurora.test_utils.parkfield.path_helpers import EMTF_RESULTS_PATH
from aurora.transfer_function.kernel_dataset import KernelDataset
from aurora.transfer_function.plot.comparison_plots import compare_two_z_files

from make_parkfield_mth5 import test_make_parkfield_mth5


def test_processing(
    return_collection=False, z_file_path=None, test_clock_zero=False
):
    """
    Parameters
    ----------
    return_collection: bool
        Controls dtype of returned object
    z_file_path: str or Path or None
        Where to store zfile

    Returns
    -------
    tf_cls: TF object,
        if  return_collection is True:
        aurora.transfer_function.transfer_function_collection.TransferFunctionCollection
        if  return_collection is False:
        mt_metadata.transfer_functions.core.TF
    """

    mth5_path = DATA_PATH.joinpath("pkd_test_00.h5")

    # Ensure there is an mth5 to process
    if not mth5_path.exists():
        test_make_parkfield_mth5()

    run_summary = RunSummary()
    run_summary.from_mth5s(
        [
            mth5_path,
        ]
    )
    tfk_dataset = KernelDataset()
    tfk_dataset.from_run_summary(run_summary, "PKD")

    cc = ConfigCreator(config_path=CONFIG_PATH)
    config = cc.create_from_kernel_dataset(
        tfk_dataset, estimator={"engine": "RME"}
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
        return_collection=return_collection,
    )

    if return_collection:
        tf_collection = tf_cls
        return tf_collection
    else:
        tf_cls.write_tf_file(fn="emtfxml_test.xml", file_type="emtfxml")
    return tf_cls


def test():
    import logging

    logging.getLogger("matplotlib.font_manager").disabled = True
    logging.getLogger("matplotlib.ticker").disabled = True

    z_file_path = AURORA_RESULTS_PATH.joinpath("pkd.zss")
    test_processing(return_collection=True, z_file_path=z_file_path)
    test_processing(
        z_file_path=z_file_path,
        test_clock_zero="user specified",
    )
    test_processing(z_file_path=z_file_path, test_clock_zero="data start")
    test_processing(z_file_path=z_file_path)

    # COMPARE WITH ARCHIVED Z-FILE
    auxilliary_z_file = EMTF_RESULTS_PATH.joinpath("PKD_272_00.zrr")
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


if __name__ == "__main__":
    test()
