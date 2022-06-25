from pathlib import Path

from aurora.config import BANDS_DEFAULT_FILE
from aurora.config.config_creator import ConfigCreator
from aurora.config.metadata import Processing

from aurora.pipelines.process_mth5 import process_mth5
from aurora.pipelines.run_summary import RunSummary
from aurora.test_utils.parkfield.path_helpers import AURORA_RESULTS_PATH
from aurora.test_utils.parkfield.path_helpers import CONFIG_PATH
from aurora.test_utils.parkfield.path_helpers import DATA_PATH
from aurora.test_utils.parkfield.path_helpers import EMTF_RESULTS_PATH
from aurora.transfer_function.kernel_dataset import KernelDataset
from aurora.transfer_function.plot.comparison_plots import compare_two_z_files

from make_parkfield_mth5 import test_make_parkfield_hollister_mth5

def test_processing(z_file_path=None):
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

    mth5_path = DATA_PATH.joinpath("pkd_sao_test_00.h5")

    # Ensure there is an mth5 to process
    if not mth5_path.exists():
        test_make_parkfield_hollister_mth5()

    run_summary = RunSummary()
    run_summary.from_mth5s([mth5_path,])
    tfk_dataset = KernelDataset()
    tfk_dataset.from_run_summary(run_summary, "PKD", "SAO")

    cc = ConfigCreator(config_path=CONFIG_PATH)
    config = cc.create_run_processing_object(emtf_band_file=BANDS_DEFAULT_FILE,
                                        sample_rate=40.0,
                                        output_channels=["ex", "ey"],
                                        )
    config.stations.from_dataset_dataframe(tfk_dataset.df)

    show_plot = False
    tf_cls = process_mth5(config,
                          tfk_dataset,
                          units="MT",
                          show_plot=show_plot,
                          z_file_path=z_file_path,
                          return_collection=True
                          )

    #tf_cls.write_tf_file(fn="emtfxml_test.xml", file_type="emtfxml")
    return tf_cls





def main():
    z_file_path = AURORA_RESULTS_PATH.joinpath("pkd.zrr")
    test_processing(z_file_path=z_file_path)

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
