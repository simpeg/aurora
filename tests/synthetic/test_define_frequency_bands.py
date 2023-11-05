import pathlib
import pdb

from aurora.config.config_creator import ConfigCreator
from aurora.pipelines.process_mth5 import process_mth5
from aurora.pipelines.run_summary import RunSummary
from aurora.test_utils.synthetic.make_mth5_from_asc import create_test1_h5
from aurora.test_utils.synthetic.paths import DATA_PATH
from aurora.test_utils.synthetic.processing_helpers import get_example_kernel_dataset
from aurora.transfer_function.kernel_dataset import KernelDataset
from mth5.helpers import close_open_files


def test_can_declare_frequencies_directly_in_config():
    """

    Returns
    -------

    """
    kernel_dataset = get_example_kernel_dataset()
    cc = ConfigCreator()
    cfg1 = cc.create_from_kernel_dataset(kernel_dataset, estimator={"engine": "RME"})
    decimation_factors = list(cfg1.decimation_info().values())  # [1, 4, 4, 4]
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
