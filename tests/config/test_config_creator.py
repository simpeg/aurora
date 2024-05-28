# import logging
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


def main():
    # tmp = TestConfigCreator()
    # tmp.setUpClass()
    # tmp.test_exception_for_non_unique_band_specification()
    unittest.main()


if __name__ == "__main__":
    main()
