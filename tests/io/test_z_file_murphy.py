import unittest

from loguru import logger

from aurora.test_utils.synthetic.paths import SyntheticTestPaths
from aurora.sandbox.io_helpers.zfile_murphy import read_z_file


class test_z_file_murphy(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.synthetic_test_paths = SyntheticTestPaths()

    def test_reader(self, z_file_path=None):

        if z_file_path is None:
            logger.info("Default z-file from emtf results being loaded")
            zss_path = self.synthetic_test_paths.emtf_results_path
            z_file_path = zss_path.joinpath("test1.zss")
        z_obj = read_z_file(z_file_path)
        assert "Hx" in z_obj.channels
        return


def main():
    unittest.main()


if __name__ == "__main__":
    main()
