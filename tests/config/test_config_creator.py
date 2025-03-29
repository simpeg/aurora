# import logging

# import filecmp
import json
import pandas as pd
import unittest

from aurora.config.config_creator import ConfigCreator
from aurora.config.config_creator import SUPPORTED_BAND_SPECIFICATION_STYLES

from aurora.config.metadata import Processing
from aurora.general_helper_functions import AURORA_PATH
from aurora.test_utils.synthetic.processing_helpers import get_example_kernel_dataset


class TestConfigCreator(unittest.TestCase):
    """ """

    @classmethod
    def setUpClass(self):
        from aurora.config.config_creator import SUPPORTED_BAND_SPECIFICATION_STYLES

        pass

    def setUp(self):
        pass

    def test_supported_band_specification_styles(self):
        cc = ConfigCreator()
        # confirm that we are restricting scope of styles
        with self.assertRaises(NotImplementedError):
            cc.band_specification_style = "some unsupported style"
        # confirm that supported styles ok
        for supported_style in SUPPORTED_BAND_SPECIFICATION_STYLES:
            cc.band_specification_style = supported_style
            cc.determine_band_specification_style()

    def test_exception_for_non_unique_band_specification(self):
        """
        tests that bands shouldn't be defined in two different ways

        """
        from aurora.config.emtf_band_setup import BANDS_DEFAULT_FILE

        kernel_dataset = get_example_kernel_dataset()
        cc = ConfigCreator()
        cfg1 = cc.create_from_kernel_dataset(
            kernel_dataset, estimator={"engine": "RME"}
        )

        with self.assertRaises(ValueError):
            cc2 = ConfigCreator(
                emtf_band_file=BANDS_DEFAULT_FILE, band_edges=cfg1.band_edges_dict
            )
            cc2.determine_band_specification_style()

    def test_frequency_bands(self):
        """
        Tests the frequency_bands method of AuroraDecimationLevel
        TODO: Move this into mt_metadata.
         - Requires a test in mt_metadata that creates a fully populated AuroraDecimationLevel

        """

        kernel_dataset = get_example_kernel_dataset(num_stations=1)
        cc = ConfigCreator()
        cfg1 = cc.create_from_kernel_dataset(
            kernel_dataset, estimator={"engine": "RME"}
        )
        dec_level_0 = cfg1.decimations[0]
        band_edges_a = dec_level_0.frequency_bands_obj().band_edges

        # compare with another way to get band edges
        delta_f = dec_level_0.frequency_sample_interval
        lower_edges = (dec_level_0.lower_bounds * delta_f) - delta_f / 2.0
        upper_edges = (dec_level_0.upper_bounds * delta_f) + delta_f / 2.0
        band_edges_b = pd.DataFrame(
            data={
                "lower_bound": lower_edges,
                "upper_bound": upper_edges,
            }
        )
        assert (band_edges_b - band_edges_a == 0).all().all()

    def test_default_synthetic_processing_parameters(self):
        """
            Test that the config can be
        Returns
        -------

        """
        kernel_dataset = get_example_kernel_dataset(num_stations=2)
        cc = ConfigCreator()
        processing_config = cc.create_from_kernel_dataset(
            kernel_dataset,
            estimator={"engine": "RME_RR"},
        )
        target_file = AURORA_PATH.joinpath(
            "aurora", "config", "tmp_processing_config.json"
        )
        processing_config.save_as_json(target_file)
        assert target_file.exists()

        # get mt_processing_parameters template
        reference_file = AURORA_PATH.joinpath(
            "aurora", "config", "processing_configuration_template.json"
        )
        assert reference_file.exists()
        with open(reference_file, "r") as f_ref:
            ref_json_str = f_ref.read()
        p2j = json.loads(ref_json_str)
        reference_processing_obj = Processing()
        reference_processing_obj.from_dict(p2j)
        # Now we want to assert that the config creators output is same as reference.
        # This may fail on github since the mth5_paths will be /runner/some_github_stuff whereas locally /home/kkappler
        # in that case, may need to create something to check only non-system specfic things , like:
        reference_processing_obj.id == processing_config.id
        # etc.
        # for local testing we can:
        assert reference_processing_obj == processing_config
        # TODO: mt_metadata #222 is still live and this nearly addresses it!


def main():
    # tmp = TestConfigCreator()
    # tmp.setUpClass()
    # tmp.test_exception_for_non_unique_band_specification()
    unittest.main()


if __name__ == "__main__":
    main()
