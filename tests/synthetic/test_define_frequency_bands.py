import pathlib
import pdb

from aurora.config.config_creator import ConfigCreator
from aurora.pipelines.process_mth5 import process_mth5
from aurora.pipelines.run_summary import RunSummary
from aurora.test_utils.synthetic.make_mth5_from_asc import create_test1_h5
from aurora.test_utils.synthetic.paths import DATA_PATH
from aurora.transfer_function.kernel_dataset import KernelDataset
from mth5.helpers import close_open_files


def test_can_declare_frequencies_directly_in_config():
    """

    Returns
    -------

    """
    decimation_factors = [1, 4, 4, 4]

    file_base = "test1.h5"

    mth5_path = DATA_PATH.joinpath(file_base)
    close_open_files()

    if not mth5_path.exists():
        create_test1_h5()

    run_summary = RunSummary()
    run_summary.from_mth5s(
        [
            mth5_path,
        ]
    )

    kernel_dataset = KernelDataset()
    kernel_dataset.from_run_summary(run_summary, "test1")

    cc = ConfigCreator()
    cfg1 = cc.create_from_kernel_dataset(kernel_dataset, estimator={"engine": "RME"})

    # Default Band edges, corresponds to DEFAULT_BANDS_FILE
    band_edges = cfg1.band_edges_dict
    cfg2 = cc.create_from_kernel_dataset(
        kernel_dataset,
        estimator={"engine": "RME"},
        band_edges=band_edges,
        decimation_factors=decimation_factors,
        num_samples_window=len(band_edges) * [128],
    )

    tf_cls1 = process_mth5(cfg1, kernel_dataset)
    tf_cls1.write(fn="cfg1.xml", file_type="emtfxml")
    tf_cls2 = process_mth5(cfg2, kernel_dataset)
    tf_cls2.write(fn="cfg2.xml", file_type="emtfxml")
    assert tf_cls2 == tf_cls1


def test():
    test_can_declare_frequencies_directly_in_config()


if __name__ == "__main__":
    test()
