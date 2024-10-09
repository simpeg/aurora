"""
This is being used to diagnose Aurora issue #139, which is concerned with using the
mt_metadata TF class to write z-files.

While investigation this issue, I have encountered another potential issue:
I would expect that I can read-in an emtf_xml and then push the same data structure
back to an xml, but this does not work as expected.

ToDo: consider adding zss and zmm checks
        # zss_file_base = f"synthetic_test1.zss"
        # tf_cls.write(fn=zss_file_base, file_type="zss")
"""

import numpy as np
import pathlib
import unittest
import warnings

from aurora.test_utils.synthetic.paths import SyntheticTestPaths
from aurora.test_utils.synthetic.processing_helpers import (
    tf_obj_from_synthetic_data,
)
from mt_metadata.transfer_functions.core import TF
from mth5.data.make_mth5_from_asc import create_test12rr_h5

warnings.filterwarnings("ignore")

synthetic_test_paths = SyntheticTestPaths()


def write_zrr(tf_obj, zrr_file_base):
    tf_obj.write(fn=zrr_file_base, file_type="zrr")


class TestZFileReadWrite(unittest.TestCase):
    """ """

    @classmethod
    def setUpClass(self):
        self.xml_file_base = pathlib.Path("synthetic_test1.xml")
        self.mth5_path = synthetic_test_paths.mth5_path.joinpath("test12rr.h5")
        self.zrr_file_base = pathlib.Path("synthetic_test1.zrr")

        if not self.mth5_path.exists():
            create_test12rr_h5(target_folder=self.mth5_path.parent)

        self._tf_obj = tf_obj_from_synthetic_data(self.mth5_path)
        write_zrr(self._tf_obj, self.zrr_file_base)
        self._tf_z_obj = TF()
        self._tf_z_obj.read(self.zrr_file_base)

    @property
    def tf_obj(self):
        return self._tf_obj

    @property
    def tf_z_obj(self):
        return self._tf_z_obj

    def test_tf_obj_from_zrr(self):
        tf_z = self.tf_z_obj
        tf = self.tf_obj
        # check numeric values
        assert (
            np.isclose(tf_z.transfer_function.data, tf.transfer_function.data, 1e-4)
        ).all()
        return tf

    # An equivalent to this test in in mt_metadata on fix_issue_190
    # def test_tf_read_and_write(self):
    #     """Checks that an ingested z-file is written back out the same"""
    #     import filecmp
    #
    #     tf_z = self._tf_z_obj
    #     out_file_name = str(self.zrr_file_base).replace(".zrr", "_rewrite.zrr")
    #     out_file_path = pathlib.Path(out_file_name)
    #     tf_z.write(out_file_path)
    #     assert filecmp.cmp(self.zrr_file_base, out_file_path)


def main():
    # tmp = TestZFileReadWrite()
    # tmp.setUp()
    # tmp.test_tf_obj_from_zrr()
    unittest.main()


if __name__ == "__main__":
    main()
