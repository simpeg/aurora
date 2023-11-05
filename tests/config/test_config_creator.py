# import logging
import unittest

from aurora.config.config_creator import ConfigCreator
from aurora.config.config_creator import SUPPORTED_BAND_SPECIFICATION_STYLES


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
        with self.assertRaises(NotImplementedError):
            cc.band_specification_style = "some unsupported style"
        for supported_style in SUPPORTED_BAND_SPECIFICATION_STYLES:
            cc.band_specification_style = supported_style
            cc.determine_band_specification_style()


def main():
    tmp = TestConfigCreator()
    tmp.test_exceptions()
    tmp.setUpClass()


#    unittest.main()


if __name__ == "__main__":
    main()
