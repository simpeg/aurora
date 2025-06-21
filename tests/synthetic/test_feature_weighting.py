"""

Integrated test of the functionality of feature weights.

TODO:
1. Add ability to degrade the sythetic data to test the feature weighting.
this might be done by adding a lot of noise to say, 75% of the data.  This
should be enough to confuse the robust tf estimation.  To verify this, we
can process the data with the default processing, and then with the
feature weighting processing, and confirm that the feature weighting
processing produces a better tf estimate.
...
to do this, start by plotting the striding_window_cohernece in aurora
to make sure that we have a baseline coherence to compare against when
we provide degraded data.

"""

from aurora.config.metadata import Processing
from aurora.config.metadata.processing import _processing_obj_from_json_file
from aurora.general_helper_functions import PROCESSING_TEMPLATES_PATH
from aurora.general_helper_functions import MT_METADATA_FEATURES_TEST_HELPERS_PATH
from aurora.pipelines.process_mth5 import process_mth5
from aurora.test_utils.synthetic.paths import SyntheticTestPaths
from mth5.data.make_mth5_from_asc import create_test1_h5
from mth5.data.make_mth5_from_asc import create_test12rr_h5
from mt_metadata.features.weights.channel_weight_spec import ChannelWeightSpec

import json
import numpy as np
import pathlib
import unittest

import mt_metadata.transfer_functions


def _load_example_channel_weight_specs(
    keep_only=[
        "striding_window_coherence",
    ]
) -> list:
    """

    Loads example channel weight specifications from a JSON file.

    Modifies it for this test so that the feature_weight_specs are only striding_window_coherence.

    Parameters
    ----------
    keep_only: list
        List of feature names to keep in the feature_weight_specs.
        Default is ["striding_window_coherence"].
    Returns
    -------
    output: list
        List of ChannelWeightSpec objects with modified feature_weight_specs.

    """
    feature_weight_json = MT_METADATA_FEATURES_TEST_HELPERS_PATH.joinpath(
        "channel_weight_specs_example.json"
    )
    assert (
        feature_weight_json.exists()
    ), f"Could not find feature weighting block json at {feature_weight_json}"

    with open(feature_weight_json, "r") as f:
        data = json.load(f)

    output = []
    channel_weight_specs = data.get("channel_weight_specs", data)
    for cws_dict in channel_weight_specs:
        cws = ChannelWeightSpec()
        cws.from_dict(cws_dict)

        # Modify the feature_weight_specs to only include striding_window_coherence
        if keep_only:
            cws.feature_weight_specs = [
                fws for fws in cws.feature_weight_specs if fws.feature.name in keep_only
            ]

        # Ensure that the feature_weight_specs is not empty
        if not cws.feature_weight_specs:
            raise ValueError(
                "No valid feature_weight_specs found in channel weight spec."
            )

        output.append(cws)

    return output


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
        "processing_configuration_with_weights_block.json"
    )

    processing_objects = {}
    processing_objects["default"] = _processing_obj_from_json_file(
        processing_params_jsons["default"]
    )
    processing_objects["new"] = _processing_obj_from_json_file(
        processing_params_jsons["new"]
    )
    processing_objects["on_the_fly"] = _processing_obj_from_json_file(
        processing_params_jsons["default"]
    )
    cws_list = _load_example_channel_weight_specs(keep_only=None)
    processing_objects["on_the_fly"].decimations[0].channel_weight_specs = cws_list
    # Confirm that the processing objects are created correctly

    assert (
        processing_objects["new"] == processing_objects["on_the_fly"]
    ), "Processing object created on the fly does not match the one loaded from file."

    cws_list = _load_example_channel_weight_specs(
        keep_only=[
            "striding_window_coherence",
        ]
    )
    processing_objects["use_this"] = _processing_obj_from_json_file(
        processing_params_jsons["default"]
    )
    processing_objects["use_this"].decimations[0].channel_weight_specs = cws_list
    # Confirm that the processing objects are created correctly

    # Walk the weights and confirm they can all be evaluated
    po_dec0 = processing_objects["new"].decimations[0]
    for chws in po_dec0.channel_weight_specs:
        for fws in chws.feature_weight_specs:
            # print(fws.feature.name)
            for wk in fws.weight_kernels:
                weight_values = wk.evaluate(np.arange(10) / 10.0)
                assert (weight_values > 0).all()  # print(weight_values)
    return processing_objects


def main():
    synthetic_test_paths = SyntheticTestPaths()
    mth5_path = synthetic_test_paths.mth5_path.joinpath("test12rr.h5")
    processing_objects = load_processing_objects_from_file()

    # print(processing_objects["default"])
    # tst_feature_weights(mth5_path,  processing_objects["default"])
    # print("OK-1")
    tst_feature_weights(mth5_path, processing_objects["use_this"])
    print("OK-2")


if __name__ == "__main__":
    main()
    print("OK-OK")
