"""Pytest translation of test_make_h5s.py"""

from loguru import logger
from mth5.data.make_mth5_from_asc import create_test4_h5

from aurora.test_utils.synthetic.paths import _get_mth5_ascii_data_path


def test_get_mth5_ascii_data_path():
    """Make sure that the ascii data are where we think they are."""
    mth5_data_path = _get_mth5_ascii_data_path()
    ascii_file_paths = list(mth5_data_path.glob("*asc"))
    file_names = [x.name for x in ascii_file_paths]
    logger.info(f"mth5_data_path = {mth5_data_path}")
    logger.info(f"file_names = {file_names}")

    assert "test1.asc" in file_names
    assert "test2.asc" in file_names


def test_make_upsampled_mth5(synthetic_test_paths):
    """Test creating upsampled mth5 file using synthetic_test_paths fixture."""
    file_version = "0.2.0"
    create_test4_h5(
        file_version=file_version, source_folder=synthetic_test_paths.ascii_data_path
    )
