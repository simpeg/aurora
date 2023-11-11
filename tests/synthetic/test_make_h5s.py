import unittest

# from aurora.test_utils.synthetic.make_mth5_from_asc import create_test1_h5
# from aurora.test_utils.synthetic.make_mth5_from_asc import create_test1_h5_with_nan
# from aurora.test_utils.synthetic.make_mth5_from_asc import create_test12rr_h5
# from aurora.test_utils.synthetic.make_mth5_from_asc import create_test2_h5
# from aurora.test_utils.synthetic.make_mth5_from_asc import create_test3_h5
from aurora.test_utils.synthetic.make_mth5_from_asc import create_test4_h5
from aurora.test_utils.synthetic.paths import SyntheticTestPaths

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

    def test_make_upsampled_mth5(self):
        file_version = "0.2.0"
        create_test4_h5(file_version=file_version, source_folder=SOURCE_PATH)


if __name__ == "__main__":
    unittest.main()
