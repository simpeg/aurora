import pathlib

from aurora.general_helper_functions import count_lines
from aurora.general_helper_functions import TEST_PATH


def test_count_lines():
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


def test():
    """ """
    test_count_lines()
    print("finito")


if __name__ == "__main__":
    test()
