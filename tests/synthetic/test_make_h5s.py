import unittest

# from mth5.data.make_mth5_from_asc import create_test1_h5
# from mth5.data.make_mth5_from_asc import create_test1_h5_with_nan
# from mth5.data.make_mth5_from_asc import create_test12rr_h5
# from mth5.data.make_mth5_from_asc import create_test2_h5
# from mth5.data.make_mth5_from_asc import create_test3_h5
from loguru import logger
from mth5.data.make_mth5_from_asc import create_test4_h5
from aurora.test_utils.synthetic.paths import SyntheticTestPaths
from aurora.test_utils.synthetic.paths import _get_mth5_ascii_data_path

synthetic_test_paths = SyntheticTestPaths()
synthetic_test_paths.mkdirs()
SOURCE_PATH = synthetic_test_paths.ascii_data_path


class TestMakeSyntheticMTH5(unittest.TestCase):
    """
    create_test1_h5(file_version=file_version)
    create_test1_h5_with_nan(file_version=file_version)
    create_test2_h5(file_version=file_version)
    create_test12rr_h5(file_version=file_version)
    create_test3_h5(file_version=file_version)
    """

    def test_get_mth5_ascii_data_path(self):
        """
        Make sure that the ascii data are where we think they are.
        Returns
        -------

        """
        mth5_data_path = _get_mth5_ascii_data_path()
        ascii_file_paths = list(mth5_data_path.glob("*asc"))
        file_names = [x.name for x in ascii_file_paths]
        logger.info(f"mth5_data_path = {mth5_data_path}")
        logger.info(f"file_names = {file_names}")

        assert "test1.asc" in file_names
        assert "test2.asc" in file_names

    def test_make_upsampled_mth5(self):
        file_version = "0.2.0"
        create_test4_h5(file_version=file_version, source_folder=SOURCE_PATH)


if __name__ == "__main__":
    unittest.main()
