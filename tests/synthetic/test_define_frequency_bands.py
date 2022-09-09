import pathlib
import pdb

from aurora.config.config_creator import ConfigCreator
from aurora.pipelines.process_mth5 import process_mth5
from aurora.pipelines.run_summary import RunSummary
from aurora.transfer_function.kernel_dataset import KernelDataset


def test_can_declare_frequencies_directly_in_config():
    # Default Band edges, corresponds to DEFAULT_BANDS_FILE
    band_edges = {}
    band_edges[0] = [
        [0.03515625, 0.04296875],
        [0.04296875, 0.05859375],
        [0.05859375, 0.07421875],
        [0.07421875, 0.09765625],
        [0.09765625, 0.12109375],
        [0.12109375, 0.15234375],
        [0.15234375, 0.19140625],
        [0.19140625, 0.23828125],
    ]

    band_edges[1] = [
        [0.00878906, 0.01074219],
        [0.01074219, 0.01269531],
        [0.01269531, 0.01660156],
        [0.01660156, 0.02050781],
        [0.02050781, 0.02636719],
        [0.02636719, 0.03417969],
    ]

    band_edges[2] = [
        [0.00219727, 0.00268555],
        [0.00268555, 0.00317383],
        [0.00317383, 0.00415039],
        [0.00415039, 0.00512695],
        [0.00512695, 0.0065918],
        [0.0065918, 0.00854492],
    ]

    band_edges[3] = [
        [0.00054932, 0.00079346],
        [0.00079346, 0.00115967],
        [0.00115967, 0.00164795],
        [0.00164795, 0.00213623],
        [0.00213623, 0.00274658],
    ]
    decimation_factors = [1, 4, 4, 4]

    DATA_DIR = pathlib.Path("data")
    file_base = "test1.h5"

    mth5_path = DATA_DIR.joinpath(file_base)
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

    cfg2 = cc.create_from_kernel_dataset(
        kernel_dataset,
        estimator={"engine": "RME"},
        band_edges=band_edges,
        decimation_factors=decimation_factors,
        num_samples_window=len(band_edges) * [128],
    )

    tf_cls1 = process_mth5(cfg1, kernel_dataset)
    tf_cls1.write_tf_file(fn="cfg1.xml", file_type="emtfxml")
    tf_cls2 = process_mth5(cfg2, kernel_dataset)
    tf_cls2.write_tf_file(fn="cfg2.xml", file_type="emtfxml")
    # OK, Now we need a way to assert equality between these tfs
    assert tf_cls2 == tf_cls1


def test():
    test_can_declare_frequencies_directly_in_config()


if __name__ == "__main__":
    test()
