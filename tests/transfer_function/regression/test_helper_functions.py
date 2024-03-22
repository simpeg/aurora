import unittest

import numpy as np

from aurora.transfer_function.regression.helper_functions import direct_solve_tf
from aurora.transfer_function.regression.helper_functions import simple_solve_tf


class TestHelperFunctions(unittest.TestCase):
    """ """

    @classmethod
    def setUpClass(self):
        self.electric_data = np.array(
            [
                4.39080123e-07 - 2.41097397e-06j,
                -2.33418464e-06 + 2.10752581e-06j,
                1.38642624e-06 - 1.87333571e-06j,
            ]
        )
        self.magnetic_data = np.array(
            [
                [7.00767250e-07 - 9.18819198e-07j, 1.94321684e-07 + 3.71934877e-07j],
                [-1.06648904e-07 + 8.19420154e-07j, 1.15361101e-08 - 6.32581646e-07j],
                [-1.02700963e-07 - 3.73904463e-07j, 3.86095787e-08 + 4.33155345e-07j],
            ]
        )
        self.expected_solution = np.array(
            [-0.04192569 - 0.36502722j, -3.65284496 - 4.05194938j]
        )

    def setUp(self):
        pass

    def test_simple_solve_tf(self):
        X = self.magnetic_data
        Y = self.electric_data
        z = simple_solve_tf(Y, X)
        assert np.isclose(z, self.expected_solution, rtol=1e-8).all()
        return z

    def test_direct_solve_tf(self):
        X = self.magnetic_data
        Y = self.electric_data
        z = direct_solve_tf(Y, X)
        assert np.isclose(z, self.expected_solution, rtol=1e-8).all()
        return z


def main():
    unittest.main()


if __name__ == "__main__":
    main()
