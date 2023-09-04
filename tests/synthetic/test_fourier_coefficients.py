"""
Flow:
1. We need to start with a list of mth5 files.
- Assert that synthetic data exist, (and build if they dont)
- should already know what you expect here ...  test1,2,3.h5 and test12rr.h5
2. Two ways to prepare the FC-schema (processing configs)
- a) use the mt_metadata processing fourier_coefficients structures explictly
- b) use the default processing configs you already use for processing, and
extract type (a) cfgs from these (the machinery to do this should exist already)
3. Loop over files and generate FCs
4. Compare fc values against some archived values

ToDo:
1. Make one test generate the decimation_and_stft_config with default values from
the decimation_and_stft_config_creator method here
2. Make another test take the existing aurora processing config and transform it to
decimation_and_stft_config


Here are the synthetic files for which this is currently passing tests
    [PosixPath('/home/kkappler/software/irismt/aurora/tests/synthetic/data/test1.h5'),
     PosixPath('/home/kkappler/software/irismt/aurora/tests/synthetic/data/test2.h5'),
     PosixPath('/home/kkappler/software/irismt/aurora/tests/synthetic/data/test3.h5'),
     PosixPath('/home/kkappler/software/irismt/aurora/tests/synthetic/data/test12rr.h5')]

"""
import unittest

from aurora.config.config_creator import ConfigCreator
from aurora.pipelines.fourier_coefficients import add_fcs_to_mth5
from aurora.pipelines.fourier_coefficients import decimation_and_stft_config_creator
from aurora.pipelines.fourier_coefficients import read_back_fcs
from aurora.pipelines.process_mth5 import process_mth5
from aurora.pipelines.run_summary import RunSummary
from aurora.test_utils.synthetic.make_mth5_from_asc import create_test1_h5
from aurora.test_utils.synthetic.make_mth5_from_asc import create_test2_h5
from aurora.test_utils.synthetic.make_mth5_from_asc import create_test3_h5
from aurora.test_utils.synthetic.make_mth5_from_asc import create_test12rr_h5
# from aurora.test_utils.synthetic.make_mth5_from_asc import main as make_all_h5
from aurora.test_utils.synthetic.make_processing_configs import create_test_run_config

from aurora.transfer_function.kernel_dataset import KernelDataset
from mth5.helpers import close_open_files

class TestAddFourierCoefficientsToSyntheticData(unittest.TestCase):
    """
    Runs several synthetic processing tests from config creation to tf_cls.

    """
    @classmethod
    def setUpClass(self):
        print("make synthetic data")
        close_open_files()
        self.file_version = "0.1.0"
        mth5_path_1 = create_test1_h5(file_version=self.file_version)
        mth5_path_2 = create_test2_h5(file_version=self.file_version)
        mth5_path_3 = create_test3_h5(file_version=self.file_version)
        mth5_path_12rr = create_test12rr_h5(file_version=self.file_version)
        self.mth5_paths = [mth5_path_1, mth5_path_2, mth5_path_3, mth5_path_12rr]



    def test_123(self):
        for mth5_path in self.mth5_paths:
            mth5_paths = [mth5_path, ]
            run_summary = RunSummary()
            run_summary.from_mth5s(mth5_paths)
            tfk_dataset = KernelDataset()

            # Get Processing Config
            if mth5_path.stem in ["test1", "test2",]:
                station_id = mth5_path.stem
                tfk_dataset.from_run_summary(run_summary, station_id)
                processing_config = create_test_run_config(station_id, tfk_dataset)
            elif mth5_path.stem in ["test3",]:
                station_id = "test3"
                tfk_dataset.from_run_summary(run_summary, station_id)
                cc = ConfigCreator()
                processing_config = cc.create_from_kernel_dataset(tfk_dataset)
            elif mth5_path.stem in ["test12rr",]:
                tfk_dataset.from_run_summary(run_summary, "test1", "test2")
                cc = ConfigCreator()
                processing_config = cc.create_from_kernel_dataset(tfk_dataset)

            # Extract FC decimations from processing config and build the layer
            fc_decimations = [x.to_fc_decimation("local") for x in processing_config.decimations]
            add_fcs_to_mth5(mth5_path, decimation_and_stft_configs=fc_decimations)
            read_back_fcs(mth5_path)
            # Confirm the file still processes fine with the fcs inside
            # This is currently failing due to tf_kernel checking if fcs already exist, NotImplementedError
            tfc = process_mth5(processing_config, tfk_dataset=tfk_dataset)
            return tfc
        print("OK")
        print("NEXT STEP is add a Tap-Point into existing processing to create these levels")
        print("NEXT STEP AFTER THAT is to try processing data from the FC LEVEL")
        return


    def test_decimation_and_stft_config_creator(self):
        cfgs = decimation_and_stft_config_creator(1.0)
        return cfgs



def main():
    # test_case = TestAddFourierCoefficientsToSyntheticData()
    # test_case.setUpClass()
    # test_case.test_123()
    # test_case.test_decimation_and_stft_config_creator()
    # print("se funciona!")
    unittest.main()

if __name__ == "__main__":
    main()