# import logging

# import filecmp
import json
import pandas as pd
import unittest

from aurora.config.config_creator import ConfigCreator
from aurora.config.config_creator import SUPPORTED_BAND_SPECIFICATION_STYLES

from aurora.config.metadata import Processing
from aurora.config.metadata.processing import _processing_obj_from_json_file
from aurora.general_helper_functions import PROCESSING_TEMPLATES_PATH
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
        target_file = PROCESSING_TEMPLATES_PATH.joinpath("tmp_processing_config.json")
        processing_config.save_as_json(target_file)

        # get mt_processing_parameters template
        reference_file = PROCESSING_TEMPLATES_PATH.joinpath(
            "processing_configuration_template.json"
        )
        assert reference_file.exists()
        reference_processing_obj = _processing_obj_from_json_file(reference_file)

        # Now we want to assert that the config creator output matches the reference file.
        # This may fail on github since the mth5_paths will be /home/runner/some_git_path whereas locally /home/kkappler
        # in that case, may need to create something to check only non-system specific things , like:
        assert reference_processing_obj.id == processing_config.id
        assert (
            reference_processing_obj.band_specification_style
            == processing_config.band_specification_style
        )
        assert (
            reference_processing_obj.channel_nomenclature
            == processing_config.channel_nomenclature
        )
        assert reference_processing_obj.decimations == processing_config.decimations
        assert reference_processing_obj.decimations == processing_config.decimations
        assert (
            reference_processing_obj.stations.local.id
            == processing_config.stations.local.id
        )
        assert (
            reference_processing_obj.stations.local.remote
            == processing_config.stations.local.remote
        )
        assert (
            reference_processing_obj.stations.local.runs
            == processing_config.stations.local.runs
        )
        assert (
            reference_processing_obj.stations.remote[0].id
            == processing_config.stations.remote[0].id
        )
        assert (
            reference_processing_obj.stations.remote[0].remote
            == processing_config.stations.remote[0].remote
        )
        assert (
            reference_processing_obj.stations.remote[0].runs
            == processing_config.stations.remote[0].runs
        )

        # keys to ignore, because they have paths from local machine:..
        # 'band_setup_file',
        # 'stations.local.mth5_path',
        # 'stations.remote[0].mth5_path (if it exists)
        # TODO: consider simply updating these paths during the test and simply:
        # assert reference_processing_obj == processing_config
        # OR
        # assert reference_processing_obj.__eq__(
        #     processing_config,
        #     ignore_keys=['band_setup_file',
        #                  'stations.local.mth5_path',
        #                  # 'stations.remote',
        #                  ])
        # but this will still have the remote reference path ... ugh:/
        target_file.unlink()


def main():
    # tmp = TestConfigCreator()
    # tmp.setUpClass()
    # tmp.test_exception_for_non_unique_band_specification()
    unittest.main()


if __name__ == "__main__":
    main()
