import mth5.mth5
from aurora.config import BANDS_DEFAULT_FILE
from aurora.config.config_creator import ConfigCreator

from aurora.pipelines.process_mth5 import process_mth5
from aurora.pipelines.run_summary import RunSummary
from aurora.sandbox.mth5_channel_summary_helpers import (
    channel_summary_to_make_mth5,
)
from aurora.test_utils.parkfield.path_helpers import AURORA_RESULTS_PATH
from aurora.test_utils.parkfield.path_helpers import CONFIG_PATH
from aurora.test_utils.parkfield.path_helpers import DATA_PATH
from aurora.test_utils.parkfield.path_helpers import EMTF_RESULTS_PATH
from aurora.transfer_function.kernel_dataset import KernelDataset
from aurora.transfer_function.plot.comparison_plots import compare_two_z_files

from make_parkfield_mth5 import make_parkfield_hollister_mth5

from mth5.mth5 import MTH5
from mth5.helpers import close_open_files


def test_stuff_that_belongs_elsewhere():
    """
    ping the mth5, extract the summary and pass it

    This test was created so that codecov would see channel_summary_to_make_mth5().
    ToDo: channel_summary_to_make_mth5() method should be moved into mth5 and removed
    from aurora, including this test.

    Returns
    -------

    """
    close_open_files()
    parkfield_h5_path = DATA_PATH.joinpath("pkd_sao_test_00.h5")

    # Ensure there is an mth5 to process
    if not parkfield_h5_path.exists():
        try:
            make_parkfield_hollister_mth5()
        except ValueError:
            print("NCEDC Likley Down")
            print("Skipping this test")
            return

    mth5_obj = mth5.mth5.MTH5()
    mth5_obj = MTH5(file_version="0.1.0")
    mth5_obj.open_mth5(parkfield_h5_path, mode="a")
    df = mth5_obj.channel_summary.to_dataframe()
    make_mth5_df = channel_summary_to_make_mth5(df)
    mth5_obj.close_mth5()
    return make_mth5_df


def test_processing(z_file_path=None):
    """
    Parameters
    ----------
    z_file_path: str or Path or None
        Where to store zfile

    Returns
    -------
    tf_cls: TF object mt_metadata.transfer_functions.core.TF
    """

    close_open_files()
    parkfield_h5_path = DATA_PATH.joinpath("pkd_sao_test_00.h5")

    # Ensure there is an mth5 to process
    if not parkfield_h5_path.exists():
        try:
            make_parkfield_hollister_mth5()
        except ValueError:
            print("NCEDC Likley Down")
            print("Skipping this test")
            return

    run_summary = RunSummary()
    run_summary.from_mth5s(
        [
            parkfield_h5_path,
        ]
    )
    tfk_dataset = KernelDataset()
    tfk_dataset.from_run_summary(run_summary, "PKD", "SAO")

    cc = ConfigCreator(config_path=CONFIG_PATH)
    config = cc.create_from_kernel_dataset(
        tfk_dataset,
        output_channels=["ex", "ey"],
    )

    show_plot = False
    tf_cls = process_mth5(
        config,
        tfk_dataset,
        units="MT",
        show_plot=show_plot,
        z_file_path=z_file_path,
    )

    # tf_cls.write_tf_file(fn="emtfxml_test.xml", file_type="emtfxml")
    return tf_cls


def test():

    import logging
    from mth5.helpers import close_open_files

    logging.getLogger("matplotlib.font_manager").disabled = True
    logging.getLogger("matplotlib.ticker").disabled = True

    test_stuff_that_belongs_elsewhere()
    z_file_path = AURORA_RESULTS_PATH.joinpath("pkd.zrr")
    test_processing(z_file_path=z_file_path)

    # COMPARE WITH ARCHIVED Z-FILE
    auxilliary_z_file = EMTF_RESULTS_PATH.joinpath("PKD_272_00.zrr")
    if z_file_path.exists():
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
    else:
        print("Z-File not found - Parkfield tests failed to generate output")
        print("NCEDC probably not returning data")
    close_open_files()


if __name__ == "__main__":
    test()
