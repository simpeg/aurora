# import logging
import pandas as pd
import unittest

from aurora.config.config_creator import ConfigCreator
from aurora.config.config_creator import SUPPORTED_BAND_SPECIFICATION_STYLES
from aurora.test_utils.synthetic.processing_helpers import get_example_kernel_dataset


class TestConfigCreator(unittest.TestCase):
    """ """

    @classmethod
    def setUpClass(self):
        from aurora.config.config_creator import SUPPORTED_BAND_SPECIFICATION_STYLES

        pass

    def setUp(self):
        pass

    def test_supported_band_specification_styles(self):
        cc = ConfigCreator()
        # confirm that we are restricting scope of styles
        with self.assertRaises(NotImplementedError):
            cc.band_specification_style = "some unsupported style"
        # confirm that supported styles ok
        for supported_style in SUPPORTED_BAND_SPECIFICATION_STYLES:
            cc.band_specification_style = supported_style
            cc.determine_band_specification_style()

    def test_exception_for_non_unique_band_specification(self):
        """
        tests that bands shouldn't be defined in two different ways

        """
        from aurora.config.emtf_band_setup import BANDS_DEFAULT_FILE

        kernel_dataset = get_example_kernel_dataset()
        cc = ConfigCreator()
        cfg1 = cc.create_from_kernel_dataset(
            kernel_dataset, estimator={"engine": "RME"}
        )

        with self.assertRaises(ValueError):
            cc2 = ConfigCreator(
                emtf_band_file=BANDS_DEFAULT_FILE, band_edges=cfg1.band_edges_dict
            )
            cc2.determine_band_specification_style()

    def test_frequency_bands(self):
        """
        Tests the frequency_bands method of AuroraDecimationLevel
        TODO: Move this into mt_metadata.
         - Requires a test in mt_metadata that creates a fully populated AuroraDecimationLevel

        """
        import numpy as np

        kernel_dataset = get_example_kernel_dataset()
        cc = ConfigCreator()
        cfg1 = cc.create_from_kernel_dataset(
            kernel_dataset, estimator={"engine": "RME"}
        )
        dec_level_0 = cfg1.decimations[0]
        band_edges_a = dec_level_0.frequency_bands_obj().band_edges

        # compare with another way to get band edges
        delta_f = dec_level_0.frequency_sample_interval
        lower_edges = (dec_level_0.lower_bounds * delta_f) - delta_f / 2.0
        upper_edges = (dec_level_0.upper_bounds * delta_f) + delta_f / 2.0
        band_edges_b = pd.DataFrame(
            data={
                "lower_bound": lower_edges,
                "upper_bound": upper_edges,
            }
        )
        assert (band_edges_b - band_edges_a == 0).all().all()


def main():
    # tmp = TestConfigCreator()
    # tmp.setUpClass()
    # tmp.test_exception_for_non_unique_band_specification()
    unittest.main()


if __name__ == "__main__":
    main()
