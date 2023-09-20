from aurora.sandbox.io_helpers.garys_matlab_zfiles.matlab_z_file_reader import (
    test_matlab_zfile_reader,
)


def test():
    test_matlab_zfile_reader(case_id="IAK34ss")
    # test_matlab_zfile_reader(case_id="synthetic")


if __name__ == "__main__":
    test()
