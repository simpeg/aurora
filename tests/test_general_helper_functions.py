import logging
import pathlib
import unittest

from aurora.general_helper_functions import count_lines
from aurora.general_helper_functions import DotDict
from aurora.general_helper_functions import get_mth5_ascii_data_path
from aurora.general_helper_functions import get_test_path

TEST_PATH = get_test_path()


class TestGeneralHelperFunctions(unittest.TestCase):
    """ """

    def setUp(self):
        logging.getLogger("matplotlib.font_manager").disabled = True
        logging.getLogger("matplotlib.ticker").disabled = True

    def test_count_lines(self):
        tmp_file = TEST_PATH.joinpath("tmp.txt")
        n_lines_in = 42
        lines = n_lines_in * ["test\n"]
        f = open(tmp_file, "w")
        f.writelines(lines)
        f.close()
        n_lines_out = count_lines(tmp_file)
        assert n_lines_out == n_lines_in
        tmp_file.unlink()
        return

    def test_dot_dict(self):
        tmp = {}
        tmp["a"] = "aa"
        tmp["b"] = "bb"
        dot_dict = DotDict(tmp)
        assert dot_dict.a == tmp["a"]
        assert dot_dict.b == "bb"

    def test_get_mth5_ascii_data_path(self):
        """
        Make sure that the ascii data are where we think they are.
        Returns
        -------

        """
        mth5_data_path = get_mth5_ascii_data_path()
        ascii_file_paths = list(mth5_data_path.glob("*asc"))
        file_names = [x.name for x in ascii_file_paths]
        assert "test1.asc" in file_names
        assert "test2.asc" in file_names


def main():
    # tmp = TestMetadataValuesSetCorrect()
    # tmp.setUp()
    # tmp.test_start_times_correct()
    unittest.main()


if __name__ == "__main__":
    main()
