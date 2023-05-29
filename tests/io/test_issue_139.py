"""
This is being used to diagnose Aurora issue #139, which is concerned with using the
mt_metadata TF class to write z-files.

While investigation this issue, I have encountered another potential issue:
I would expect that I can read-in an emtf_xml and then push the same data structure
back to an xml, but this does not work as expected.

ToDo: consider adding zss and zmm checks
        # print(type(tf_cls))
        # zss_file_base = f"synthetic_test1.zss"
        # tf_cls.write(fn=zss_file_base, file_type="zss")
"""

import logging
import numpy as np
import pathlib
import unittest
import warnings

from aurora.config.config_creator import ConfigCreator
from aurora.pipelines.process_mth5 import process_mth5
from aurora.pipelines.run_summary import RunSummary
from aurora.test_utils.synthetic.make_mth5_from_asc import create_test12rr_h5
from aurora.test_utils.synthetic.paths import DATA_PATH
from aurora.transfer_function.kernel_dataset import KernelDataset
from mt_metadata.transfer_functions.core import TF

warnings.filterwarnings("ignore")


def get_tf_obj_from_processing_synthetic_data(mth5_path):
    run_summary = RunSummary()
    run_summary.from_mth5s(list((mth5_path,)))

    kernel_dataset = KernelDataset()
    kernel_dataset.from_run_summary(run_summary, "test1", "test2")

    # Define the processing Configuration
    cc = ConfigCreator()
    config = cc.create_from_kernel_dataset(kernel_dataset)

    tf_cls = process_mth5(
        config,
        kernel_dataset,
        units="MT",
        z_file_path="zzz.zz",
    )
    return tf_cls


class TestZFileReadWrite(unittest.TestCase):
    """ """

    def setUp(self):
        """
        ivar mth5_path: The synthetic mth5 file used for testing
        Returns
        -------

        """
        logging.getLogger("matplotlib.font_manager").disabled = True
        logging.getLogger("matplotlib.ticker").disabled = True
        self.xml_file_base = pathlib.Path("synthetic_test1.xml")
        self.mth5_path = DATA_PATH.joinpath("test12rr.h5")
        self.zrr_file_base = pathlib.Path("synthetic_test1.zrr")

        if not self.mth5_path.exists():
            create_test12rr_h5()

        self._tf_obj = get_tf_obj_from_processing_synthetic_data(self.mth5_path)

    def tf_obj(self):
        return self._tf_obj

    def tf_z_obj(self):
        if not self.zrr_file_base.exists():
            self.test_tf_obj_from_zrr()
        tf_z = TF()
        tf_z.read(self.zrr_file_base)
        return tf_z

    def test_zrr_from_tf_obj(self):
        tf_obj = self.tf_obj()
        tf_obj.write(fn=self.zrr_file_base, file_type="zrr")

    def test_tf_obj_from_zrr(self):
        tf_z = self.tf_z_obj()
        tf = self.tf_obj()
        # check numeric values
        assert (
            np.isclose(tf_z.transfer_function.data, tf.transfer_function.data, 1e-4)
        ).all()
        # check metadata
        print("add metadata checks for station name, azimuths and tilts")
        return tf

    def test_tf_read_and_write(self):
        tf_z = self.tf_z_obj()
        out_file_name = str(self.zrr_file_base).replace(".zrr", "_rewrite.zrr")
        tf_z.write(out_file_name)
        print("Add assert statement that the zrr are the same")

    def test_tf_write_and_read(self):
        tf_obj = self.tf_obj()
        tf_obj.write(fn=self.xml_file_base, file_type="emtfxml")

        tf_obj2 = TF()
        tf_obj2.read(fn=self.xml_file_base)
        print("ASSERT tfobj==tfob2 everywhere it hsould")


def main():
    # tmp = TestZFileReadWrite()
    # tmp.setUp()
    # tmp.test_tf_obj_from_zrr()
    unittest.main()


if __name__ == "__main__":
    main()
