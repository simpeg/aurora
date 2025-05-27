"""

Integrated test of the functionality of feature weights.

"""
import numpy as np

from aurora.config.metadata import Processing
from aurora.config.metadata.processing import _processing_obj_from_json_file
from aurora.general_helper_functions import PROCESSING_TEMPLATES_PATH
from aurora.pipelines.process_mth5 import process_mth5
from aurora.test_utils.synthetic.paths import SyntheticTestPaths
import mt_metadata.transfer_functions
from mth5.data.make_mth5_from_asc import create_test1_h5
from mth5.data.make_mth5_from_asc import create_test12rr_h5

import pathlib
import unittest

# class TestFeatureWeighting(unittest.TestCase):
#     """ """
#
#     @classmethod
#     def setUpClass(cls):
#         synthetic_test_paths = SyntheticTestPaths()
#         self.mth5_path = synthetic_test_paths.mth5_path.joinpath("test12rr.h5")
#
#
#         pass
#
#     def setUp(self):
#         pass
#
#     def test_supported_band_specification_styles(self):
#         cc = ConfigCreator()
#         # confirm that we are restricting scope of styles
#         with self.assertRaises(NotImplementedError):


# Set UP Class stuffs


def tst_feature_weights(
    mth5_path: pathlib.Path,
    processing_obj: Processing,
) -> mt_metadata.transfer_functions.TF:
    """
    Executes aurora processing on mth5_path, and returns mt_metadata TF object.
    - Helper function for test_issue_139

    """
    from aurora.general_helper_functions import PROCESSING_TEMPLATES_PATH
    from aurora.config.config_creator import ConfigCreator
    from mth5.processing import RunSummary, KernelDataset

    run_summary = RunSummary()
    run_summary.from_mth5s(list((mth5_path,)))

    kernel_dataset = KernelDataset()
    kernel_dataset.from_run_summary(run_summary, "test1", "test2")

    # Define the processing Configuration
    config = processing_obj
    # cc = ConfigCreator()
    # config = cc.create_from_kernel_dataset(kernel_dataset)

    tf_cls = process_mth5(
        config,
        kernel_dataset,
        units="MT",
        z_file_path="test1_RRtest2.zrr",
    )
    return tf_cls


def load_processing_objects_from_file() -> dict:
    """
    Place to test reading in the processing jsons and check that their structures are as expected.

    Returns
    -------

    """
    processing_params_jsons = {}
    processing_params_jsons["default"] = PROCESSING_TEMPLATES_PATH.joinpath(
        "processing_configuration_template.json"
    )
    processing_params_jsons["new"] = PROCESSING_TEMPLATES_PATH.joinpath(
        "test_processing_config_with_weights_block.json"
    )
    processing_objects = {}

    # processing_objects["default"] = _processing_obj_from_json_file(
    #     processing_params_jsons["default"]
    # )
    processing_objects["new"] = _processing_obj_from_json_file(
        processing_params_jsons["new"]
    )

    # Walk the weights and confirm they can all be evaluated
    po_dec0 = processing_objects["new"].decimations[0]
    for chws in po_dec0.channel_weight_specs:
        for fws in chws.feature_weight_specs:
            print(fws.feature.name)
            for wk in fws.weight_kernels:
                qq = wk.evaluate(np.arange(10) / 10.0)
                print(qq)
    return processing_objects


def main():
    synthetic_test_paths = SyntheticTestPaths()
    mth5_path = synthetic_test_paths.mth5_path.joinpath("test12rr.h5")
    processing_objects = load_processing_objects_from_file()

    # tst_feature_weights(mth5_path,  processing_objects["default"])
    # print("OK-1")
    tst_feature_weights(mth5_path, processing_objects["new"])
    print("OK-2")


if __name__ == "__main__":
    main()
    print("OK-OK")
