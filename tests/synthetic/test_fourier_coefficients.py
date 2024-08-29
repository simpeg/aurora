import unittest

from aurora.config.config_creator import ConfigCreator
from aurora.pipelines.fourier_coefficients import add_fcs_to_mth5
from aurora.pipelines.fourier_coefficients import fc_decimations_creator
from aurora.pipelines.fourier_coefficients import read_back_fcs
from aurora.pipelines.process_mth5 import process_mth5
from aurora.test_utils.synthetic.make_processing_configs import (
    create_test_run_config,
)
from aurora.test_utils.synthetic.paths import SyntheticTestPaths
from mth5.data.make_mth5_from_asc import create_test1_h5
from mth5.data.make_mth5_from_asc import create_test2_h5
from mth5.data.make_mth5_from_asc import create_test3_h5
from mth5.data.make_mth5_from_asc import create_test12rr_h5

# from mtpy-v2
from mtpy.processing import RunSummary, KernelDataset

from loguru import logger
from mth5.helpers import close_open_files

synthetic_test_paths = SyntheticTestPaths()
synthetic_test_paths.mkdirs()
AURORA_RESULTS_PATH = synthetic_test_paths.aurora_results_path


class TestAddFourierCoefficientsToSyntheticData(unittest.TestCase):
    """
    Runs several synthetic processing tests from config creation to tf_cls.

    There are two ways to prepare the FC-schema
      a) use the mt_metadata.FCDecimation class explictly
      b) mt_metadata.transfer_functions.processing.aurora.decimation_level.DecimationLevel has
      a to_fc_decimation() method that returns mt_metadata.FCDecimation

    Flow is to make some mth5 files from synthetic data, then loop over those files adding fcs.
    Finally, process the mth5s to make TFs.

    Synthetic files for which this is currently passing tests:
    [PosixPath('/home/kkappler/software/irismt/aurora/tests/synthetic/data/test1.h5'),
     PosixPath('/home/kkappler/software/irismt/aurora/tests/synthetic/data/test2.h5'),
     PosixPath('/home/kkappler/software/irismt/aurora/tests/synthetic/data/test3.h5'),
     PosixPath('/home/kkappler/software/irismt/aurora/tests/synthetic/data/test12rr.h5')]

      TODO: review test_123 to see if it can be shortened.
    """

    @classmethod
    def setUpClass(self):
        """
        Makes some synthetic h5 files for testing.

        """
        logger.info("Making synthetic data")
        close_open_files()
        self.file_version = "0.1.0"
        mth5_path_1 = create_test1_h5(file_version=self.file_version)
        mth5_path_2 = create_test2_h5(file_version=self.file_version)
        mth5_path_3 = create_test3_h5(file_version=self.file_version)
        mth5_path_12rr = create_test12rr_h5(file_version=self.file_version)
        self.mth5_paths = [
            mth5_path_1,
            mth5_path_2,
            mth5_path_3,
            mth5_path_12rr,
        ]

    def test_123(self):
        """
        This test adds FCs to each of the synthetic files that get built in setUpClass method.
        - This could probably be shortened, it isn't clear that all the h5 files need to have fc added
        and be processed too.

        uses the to_fc_decimation() method of
        mt_metadata.transfer_functions.processing.aurora.decimation_level.DecimationLevel.

        Returns
        -------

        """
        for mth5_path in self.mth5_paths:
            mth5_paths = [
                mth5_path,
            ]
            run_summary = RunSummary()
            run_summary.from_mth5s(mth5_paths)
            tfk_dataset = KernelDataset()

            # Get Processing Config
            if mth5_path.stem in [
                "test1",
                "test2",
            ]:
                station_id = mth5_path.stem
                tfk_dataset.from_run_summary(run_summary, station_id)
                processing_config = create_test_run_config(station_id, tfk_dataset)
            elif mth5_path.stem in [
                "test3",
            ]:
                station_id = "test3"
                tfk_dataset.from_run_summary(run_summary, station_id)
                cc = ConfigCreator()
                processing_config = cc.create_from_kernel_dataset(tfk_dataset)
            elif mth5_path.stem in [
                "test12rr",
            ]:
                tfk_dataset.from_run_summary(run_summary, "test1", "test2")
                cc = ConfigCreator()
                processing_config = cc.create_from_kernel_dataset(tfk_dataset)

            # Extract FC decimations from processing config and build the layer
            fc_decimations = [
                x.to_fc_decimation() for x in processing_config.decimations
            ]
            # For code coverage, have a case where fc_decimations is None
            if mth5_path.stem == "test1":
                fc_decimations = None

            add_fcs_to_mth5(mth5_path, fc_decimations=fc_decimations)
            read_back_fcs(mth5_path)

            # Confirm the file still processes fine with the fcs inside
            tfc = process_mth5(processing_config, tfk_dataset=tfk_dataset)

        return tfc

    def test_fc_decimations_creator(self):
        """"""
        cfgs = fc_decimations_creator(1.0)

        # test time period must of of type
        with self.assertRaises(NotImplementedError):
            time_period = ["2023-01-01T17:48:59", "2023-01-09T08:54:08"]
            fc_decimations_creator(1.0, time_period=time_period)
        return cfgs

    def test_create_then_use_stored_fcs_for_processing(self):
        """"""
        from test_processing import process_synthetic_2

        z_file_path_1 = AURORA_RESULTS_PATH.joinpath("test2.zss")
        z_file_path_2 = AURORA_RESULTS_PATH.joinpath("test2_from_stored_fc.zss")
        tf1 = process_synthetic_2(
            force_make_mth5=True, z_file_path=z_file_path_1, save_fc=True
        )
        tf2 = process_synthetic_2(force_make_mth5=False, z_file_path=z_file_path_2)
        assert tf1 == tf2


def main():
    # test_case = TestAddFourierCoefficientsToSyntheticData()
    # test_case.setUpClass()
    # test_case.test_create_then_use_stored_fcs_for_processing()
    # test_case.test_123()
    # test_case.fc_decimations_creator()
    unittest.main()


if __name__ == "__main__":
    main()
