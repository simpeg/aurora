"""Pytest translation of the unittest-based `test_issue_139.py`.

Uses mth5-provided fixtures where available to be xdist-safe and fast.

This test writes a TF z-file (zrr) from an in-memory TF object generated
from a synthetic MTH5 file, reads it back, and asserts numeric equality
of primary arrays.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from mt_metadata.transfer_functions.core import TF

from aurora.test_utils.synthetic.processing_helpers import tf_obj_from_synthetic_data


@pytest.fixture
def tf_obj_from_mth5(fresh_test12rr_mth5: Path):
    """Create a TF object from the provided fresh `test12rr` MTH5 file.

    Uses the `fresh_test12rr_mth5` fixture (created by the mth5 `conftest.py`).
    """
    return tf_obj_from_synthetic_data(fresh_test12rr_mth5)


def write_and_read_zrr(tf_obj: TF, zrr_path: Path) -> TF:
    """Write `tf_obj` to `zrr_path` as a zrr file and read it back as TF."""
    tf_obj.run_metadata.channels["hy"].measurement_azimuth = 90
    tf_obj.run_metadata.channels["ey"].measurement_azimuth = 90
    # write expects a filename; TF.write will create the zrr
    tf_obj.write(fn=str(zrr_path), file_type="zrr")

    tf_z = TF()
    tf_z.read(str(zrr_path))
    return tf_z


def _register_cleanup(cleanup_test_files, p: Path):
    try:
        cleanup_test_files(p)
    except Exception:
        # Best-effort: if the helper isn't available, ignore
        pass


def test_write_and_read_zrr(
    tf_obj_from_mth5,
    make_worker_safe_path,
    cleanup_test_files,
    tmp_path: Path,
    subtests,
):
    """Round-trip a TF through a `.zrr` write/read and validate arrays.

    This test uses `make_worker_safe_path` to generate a worker-unique
    filename so it is safe to run under `pytest-xdist`.
    """

    # Create a worker-safe path in the tmp directory
    zrr_path = make_worker_safe_path("synthetic_test1.zrr", tmp_path)

    # register cleanup so sessions don't leak files
    _register_cleanup(cleanup_test_files, zrr_path)

    # Write and read back
    tf_z = write_and_read_zrr(tf_obj_from_mth5, zrr_path)

    # Use subtests to make multiple assertions clearer in pytest output
    with subtests.test("transfer_function_data"):
        assert (
            np.isclose(
                tf_z.transfer_function.data,
                tf_obj_from_mth5.transfer_function.data,
                atol=1e-4,
            )
        ).all()

    with subtests.test("period_arrays"):
        assert np.allclose(tf_z.period, tf_obj_from_mth5.period)

    with subtests.test("shape_checks"):
        assert (
            tf_z.transfer_function.data.shape
            == tf_obj_from_mth5.transfer_function.data.shape
        )
