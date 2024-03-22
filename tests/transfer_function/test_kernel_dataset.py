import logging
import pathlib
import unittest

from aurora.pipelines.run_summary import RunSummary
from aurora.test_utils.synthetic.make_mth5_from_asc import create_test12rr_h5
from aurora.transfer_function.kernel_dataset import KernelDataset


class TestKernelDataset(unittest.TestCase):
    """ """

    @classmethod
    def setUpClass(self):
        self._mth5_path = create_test12rr_h5()
        self._run_summary = RunSummary()
        self._run_summary.from_mth5s(
            [
                self._mth5_path,
            ]
        )

    def setUp(self):
        self.run_summary = self._run_summary.clone()
        self.kd = KernelDataset()
        self.kd.from_run_summary(self.run_summary, "test1", "test2")

    def test_exception_from_empty_run_summary(self):
        # make the run summary df empty
        self.run_summary.df.valid = False
        self.run_summary.drop_invalid_rows()
        with self.assertRaises(ValueError):  # as context:
            self.kd.from_run_summary(self.run_summary, "test1", "test2")

    def test_clone_dataframe(self):
        cloned_df = self.kd.clone_dataframe()

        # fc column is None so this wont be true
        self.assertFalse((cloned_df == self.kd.df).all().all())

        cloned_df["fc"] = False
        self.kd.df["fc"] = False
        assert (cloned_df == self.kd.df).all().all()

    def test_clone(self):
        clone = self.kd.clone()

        # fc column is None so this wont be true
        self.assertFalse((clone.df == self.kd.df).all().all())

        clone.df["fc"] = False
        self.kd.df["fc"] = False
        assert (clone.df == self.kd.df).all().all()
        # add more checks


def main():
    unittest.main()


if __name__ == "__main__":
    main()
