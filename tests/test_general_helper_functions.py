import logging
import unittest

from aurora.general_helper_functions import count_lines
from aurora.general_helper_functions import DotDict
from aurora.general_helper_functions import get_test_path
from loguru import logger

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


def main():
    # tmp = TestMetadataValuesSetCorrect()
    # tmp.setUp()
    # tmp.test_start_times_correct()
    unittest.main()


if __name__ == "__main__":
    main()
