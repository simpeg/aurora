import unittest

from aurora.config.config_creator import ConfigCreator

# from aurora.config.emtf_band_setup import BANDS_DEFAULT_FILE
from aurora.pipelines.transfer_function_kernel import station_obj_from_row
from aurora.pipelines.transfer_function_kernel import TransferFunctionKernel
from aurora.test_utils.synthetic.processing_helpers import get_example_kernel_dataset


class TestTransferFunctionKernel(unittest.TestCase):
    """ """

    @classmethod
    def setUpClass(cls) -> None:
        pass
        # kernel_dataset = get_example_kernel_dataset()
        # cc = ConfigCreator()
        # processing_config = cc.create_from_kernel_dataset(
        #     kernel_dataset, estimator={"engine": "RME"}
        # )
        # cls.tfk = TransferFunctionKernel(dataset=kernel_dataset, config=processing_config)

    def setUp(self):
        pass

    def test_init(self):
        kernel_dataset = get_example_kernel_dataset()
        cc = ConfigCreator()
        processing_config = cc.create_from_kernel_dataset(
            kernel_dataset, estimator={"engine": "RME"}
        )
        tfk = TransferFunctionKernel(dataset=kernel_dataset, config=processing_config)
        assert isinstance(tfk, TransferFunctionKernel)

    def test_cannot_init_without_processing_config(self):
        with self.assertRaises(TypeError):
            TransferFunctionKernel()

    # def test_helper_function_station_obj_from_row(self):
    #     """
    #     Need to make sure that test1.h5 exists
    #     - also need a v1 and a v2 file to make this work
    #     - consider making test1_v1.h5, test1_v2.h5
    #     - for now, this gets tested in the integrated tests
    #     """
    #     pass


def main():
    unittest.main()


if __name__ == "__main__":
    main()
