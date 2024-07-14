import unittest

import numpy as np
import pandas as pd
from aurora.transfer_function.regression.base import RegressionEstimator


def make_mini_dataset(n_rows=None):
    """
    TODO: Make this a pytest fixture
    Parameters
    ----------
    n_rows

    Returns
    -------

    """
    ex_data = np.array(
        [
            4.39080123e-07 - 2.41097397e-06j,
            -2.33418464e-06 + 2.10752581e-06j,
            1.38642624e-06 - 1.87333571e-06j,
        ]
    )
    hx_data = np.array(
        [
            7.00767250e-07 - 9.18819198e-07j,
            -1.06648904e-07 + 8.19420154e-07j,
            -1.02700963e-07 - 3.73904463e-07j,
        ]
    )

    hy_data = np.array(
        [
            1.94321684e-07 + 3.71934877e-07j,
            1.15361101e-08 - 6.32581646e-07j,
            3.86095787e-08 + 4.33155345e-07j,
        ]
    )
    timestamps = pd.date_range(
        start=pd.Timestamp("1977-03-02T06:00:00"), periods=len(ex_data), freq="S"
    )
    frequency = 0.666 * np.ones(len(ex_data))

    df = pd.DataFrame(
        data={
            "time": timestamps,
            "frequency": frequency,
            "ex": ex_data,
            "hx": hx_data,
            "hy": hy_data,
        }
    )
    if n_rows:
        df = df.iloc[0:n_rows]
    df = df.set_index(["time", "frequency"])
    xr_ds = df.to_xarray()
    return xr_ds


class TestRegressionBase(unittest.TestCase):
    """ """

    @classmethod
    def setUpClass(self):
        self.dataset = make_mini_dataset(n_rows=1)
        self.expected_solution = np.array(
            [-0.04192569 - 0.36502722j, -3.65284496 - 4.05194938j]
        )

    def setUp(self):
        pass

    def test_regression(self):
        dataset = make_mini_dataset()
        X = dataset[["hx", "hy"]]
        X = X.stack(observation=("frequency", "time"))
        Y = dataset[
            [
                "ex",
            ]
        ]
        Y = Y.stack(observation=("frequency", "time"))
        re = RegressionEstimator(X=X, Y=Y)
        re.estimate_ols()
        difference = re.b - np.atleast_2d(self.expected_solution).T
        assert np.isclose(difference, 0).all()
        re.estimate()
        difference = re.b - np.atleast_2d(self.expected_solution).T
        assert np.isclose(difference, 0).all()

    def test_underdetermined_regression(self):
        """ """
        dataset = make_mini_dataset(n_rows=1)
        X = dataset[["hx", "hy"]]
        X = X.stack(observation=("frequency", "time"))
        Y = dataset[
            [
                "ex",
            ]
        ]
        Y = Y.stack(observation=("frequency", "time"))
        re = RegressionEstimator(X=X, Y=Y)
        re.solve_underdetermined()
        assert re.b is not None

    def test_can_handle_xr_dataarray(self):
        dataset = make_mini_dataset()
        X = dataset[["hx", "hy"]]
        X = X.stack(observation=("frequency", "time"))
        Y = dataset[
            [
                "ex",
            ]
        ]
        Y = Y.stack(observation=("frequency", "time"))
        X_da = X.to_array()
        Y_da = Y.to_array()
        re = RegressionEstimator(X=X_da, Y=Y_da)
        re.estimate_ols()
        difference = re.b - np.atleast_2d(self.expected_solution).T
        assert np.isclose(difference, 0).all()
        re.estimate()
        difference = re.b - np.atleast_2d(self.expected_solution).T
        assert np.isclose(difference, 0).all()

    def test_can_handle_np_ndarray(self):
        """
        While we are at it -- handle numpy arrays as well.
        Returns
        -------

        """
        dataset = make_mini_dataset()
        X = dataset[["hx", "hy"]]
        X = X.stack(observation=("frequency", "time"))
        Y = dataset[
            [
                "ex",
            ]
        ]
        Y = Y.stack(observation=("frequency", "time"))
        X_np = X.to_array().data
        Y_np = Y.to_array().data
        re = RegressionEstimator(X=X_np, Y=Y_np)
        re.estimate_ols()
        difference = re.b - np.atleast_2d(self.expected_solution).T
        assert np.isclose(difference, 0).all()
        re.estimate()
        difference = re.b - np.atleast_2d(self.expected_solution).T
        assert np.isclose(difference, 0).all()


def main():
    unittest.main()


if __name__ == "__main__":
    main()
