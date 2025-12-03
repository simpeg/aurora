import unittest

from aurora.config.config_creator import ConfigCreator
from aurora.pipelines.process_mth5 import process_mth5
from aurora.test_utils.synthetic.paths import SyntheticTestPaths
from aurora.test_utils.synthetic.processing_helpers import get_example_kernel_dataset
from aurora.test_utils.synthetic.triage import tfs_nearly_equal


synthetic_test_paths = SyntheticTestPaths()


class TestDefineBandsFromDict(unittest.TestCase):
    def test_can_declare_frequencies_directly_in_config(self):
        """
        Test that manually declared frequency bands produce same results as defaults.

        This test verifies that explicitly passing band_edges to create_from_kernel_dataset
        produces the same transfer function as using the default band setup. The key is to
        use the same num_samples_window in both configs, since band edges are calculated
        based on FFT harmonics which depend on the window size.
        """
        kernel_dataset = get_example_kernel_dataset()
        cc = ConfigCreator()
        cfg1 = cc.create_from_kernel_dataset(
            kernel_dataset, estimator={"engine": "RME"}
        )
        decimation_factors = list(cfg1.decimation_info.values())  # [1, 4, 4, 4]
        # Default Band edges, corresponds to DEFAULT_BANDS_FILE
        band_edges = cfg1.band_edges_dict

        # Use the same num_samples_window as cfg1 (default is 256)
        # to ensure band_edges align with FFT harmonics
        num_samples_window = cfg1.decimations[0].stft.window.num_samples

        cfg2 = cc.create_from_kernel_dataset(
            kernel_dataset,
            estimator={"engine": "RME"},
            band_edges=band_edges,
            decimation_factors=decimation_factors,
            num_samples_window=len(band_edges) * [num_samples_window],
        )

        cfg1_path = synthetic_test_paths.aurora_results_path.joinpath("cfg1.xml")
        cfg2_path = synthetic_test_paths.aurora_results_path.joinpath("cfg2.xml")

        tf_cls1 = process_mth5(cfg1, kernel_dataset)
        tf_cls1.write(fn=cfg1_path, file_type="emtfxml")
        tf_cls2 = process_mth5(cfg2, kernel_dataset)
        tf_cls2.write(fn=cfg2_path, file_type="emtfxml")
        assert tfs_nearly_equal(tf_cls2, tf_cls1)


if __name__ == "__main__":
    unittest.main()
