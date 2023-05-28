"""
This is an executable dump of aurora/tutorials/synthetic_data_processing.ipynb

This is being used to diagnose Aurora issue #139, which is concerned with using the
mt_metadata TF class to write z-files.

While investigation this issue, I have encountered another potential issue:
I would expect that I can read-in an emtf_xml and then push the same data structure
back to an xml, but this does not work as expected.
"""
# ## Process Synthetic Data with Aurora

# This notebook shows how to process MTH5 data from a synthetic dataset.

# Steps
# 1. Create the synthetic mth5
# 2. Get a Run Summary from the mth5
# 3. Select the station to process and optionally the remote reference station
# 4. Create a processing config
# 5. Generate TFs
# 6. Archive the TFs (in emtf_xml or z-file)

# ### Here are the modules we will need to import

# In[1]:


import pathlib
import warnings

from aurora.config.config_creator import ConfigCreator
from aurora.pipelines.process_mth5 import process_mth5
from aurora.pipelines.run_summary import RunSummary
from aurora.test_utils.synthetic.make_mth5_from_asc import create_test12rr_h5
from aurora.test_utils.synthetic.paths import DATA_PATH
from aurora.transfer_function.kernel_dataset import KernelDataset

warnings.filterwarnings("ignore")

XML_FILE_BASE = "synthetic_test1.xml"


# def test_can_read_and_write_xml():
#     """
#
#     Parameters
#     ----------
#     tf_cls: mt_metadata.transfer_functions.core import TF
#
#
#     """
#
#     from mt_metadata.transfer_functions.core import TF
#
#     tf0 = TF()
#     tf0.read(XML_FILE_BASE)
#
#     xml_file_base = XML_FILE_BASE.replace("test1", "test1_rewrite")
#     tf0.write(fn=xml_file_base, file_type="emtfxml")
#
#     tf1 = TF()
#     tf1.read(xml_file_base)
#
#     # First issue:
#     try:
#         assert tf0 == tf1
#     except AssertionError:
#         print("Fail expected condition")
#         print("the pre-export and post-read objects are different")
#
#     return


def test_can_write_and_read_xml(tf_cls):
    """

    Parameters
    ----------
    tf_cls: mt_metadata.transfer_functions.core import TF


    """

    from mt_metadata.transfer_functions.core import TF

    xml_file_base = "synthetic_test1.xml"
    tf_cls.write(fn=xml_file_base, file_type="emtfxml")

    new_tf = TF()
    new_tf.read_tf_file(xml_file_base)

    # First issue:
    try:
        assert new_tf == tf_cls
    except AssertionError:
        print("Fail expected condition")
        print("the pre-export and post-read objects are different")

    # Second Issue:
    xml_file_base2 = xml_file_base.replace(".xml", "_a.xml")
    try:
        new_tf.write(fn=xml_file_base2, file_type="emtfxml")
    except AttributeError:
        print("Failed to write TF back to xml")

    return


def test_issue_139():
    # ## Define mth5 file
    # The synthetic mth5 file is used for testing in `aurora/tests/synthetic/`

    mth5_path = DATA_PATH.joinpath("test12rr.h5")

    # If it doesn't exist, or you want to re-make it, call `create_test12rr_h5()`

    if not mth5_path.exists():
        create_test12rr_h5()

    # ## Get a Run Summary

    mth5_run_summary = RunSummary()
    mth5_run_summary.from_mth5s(
        [
            mth5_path,
        ]
    )
    run_summary = mth5_run_summary.clone()
    run_summary.mini_summary

    # ## Define a Kernel Dataset

    kernel_dataset = KernelDataset()
    kernel_dataset.from_run_summary(run_summary, "test1", "test2")

    # ## Define the processing Configuration
    cc = ConfigCreator()
    config = cc.create_from_kernel_dataset(kernel_dataset)

    # ## Call process_mth5

    show_plot = False
    tf_cls = process_mth5(
        config,
        kernel_dataset,
        units="MT",
        show_plot=show_plot,
        z_file_path="zzz.zz",
    )

    tf_cls.write(fn=XML_FILE_BASE, file_type="emtfxml")

    # print(type(tf_cls))
    # zss_file_base = f"synthetic_test1.zss"
    # tf_cls.write(fn=zss_file_base, file_type="zss")

    zrr_file_base = "synthetic_test1.zrr"
    tf_cls.write(fn=zrr_file_base, file_type="zrr")

    #
    # zmm_file_base = f"synthetic_test1.zmm"
    # tf_cls.write(fn=zmm_file_base, file_type="zmm")


def main():
    test_issue_139()
    # test_can_read_and_write_xml()
    # test_can_write_and_read_xml()


if __name__ == "__main__":
    main()
