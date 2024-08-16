from aurora.time_series.xarray_helpers import initialize_xrda_2d
from aurora.transfer_function.cross_power import tf_from_cross_powers
from aurora.transfer_function.cross_power import _channel_names
from aurora.transfer_function.cross_power import (
    _zxx,
    _zxy,
    _zyx,
    _zyy,
    _tx,
    _ty,
    _tf__x,
    _tf__y,
)
from mt_metadata.transfer_functions import (
    STANDARD_INPUT_CHANNELS,
    STANDARD_OUTPUT_CHANNELS,
)

import unittest
import numpy as np


class TestCrossPower(unittest.TestCase):
    """ """

    @classmethod
    def setUpClass(self):
        # self._mth5_path = create_test12rr_h5()  # will use this in a future version
        components = STANDARD_INPUT_CHANNELS + STANDARD_OUTPUT_CHANNELS

        self.station_ids = ["MT1", "MT2"]
        station_1_channels = [f"{self.station_ids[0]}_{x}" for x in components]
        station_2_channels = [f"{self.station_ids[1]}_{x}" for x in components]
        channels = station_1_channels + station_2_channels
        sdm = initialize_xrda_2d(
            channels=channels,
            dtype=complex,
        )
        np.random.seed(0)
        data = np.random.random((len(channels), 1000))
        sdm.data = np.cov(data)
        self.sdm = sdm

    def setUp(self):
        pass

    def test_channel_names(self):
        station = self.station_ids[0]
        remote = self.station_ids[1]
        Ex, Ey, Hx, Hy, Hz, A, B = _channel_names(
            station_id=station, remote=remote, join_char="_"
        )
        assert Ex == f"{station}_{'ex'}"
        assert Ey == f"{station}_{'ey'}"
        assert Hx == f"{station}_{'hx'}"
        assert Hy == f"{station}_{'hy'}"
        assert Hz == f"{station}_{'hz'}"
        assert A == f"{remote}_{'hx'}"
        assert B == f"{remote}_{'hy'}"

    def test_generalizing_vozoffs_equations(self):
        station = self.station_ids[0]
        remote = self.station_ids[1]
        Ex, Ey, Hx, Hy, Hz, A, B = _channel_names(
            station_id=station, remote=remote, join_char="_"
        )
        assert _zxx(self.sdm, Ex=Ex, Hx=Hx, Hy=Hy, A=A, B=B) == _tf__x(
            self.sdm, Y=Ex, Hx=Hx, Hy=Hy, A=A, B=B
        )
        assert _zxy(self.sdm, Ex=Ex, Hx=Hx, Hy=Hy, A=A, B=B) == _tf__y(
            self.sdm, Y=Ex, Hx=Hx, Hy=Hy, A=A, B=B
        )
        assert _zyx(self.sdm, Ey=Ey, Hx=Hx, Hy=Hy, A=A, B=B) == _tf__x(
            self.sdm, Y=Ey, Hx=Hx, Hy=Hy, A=A, B=B
        )
        assert _zyy(self.sdm, Ey=Ey, Hx=Hx, Hy=Hy, A=A, B=B) == _tf__y(
            self.sdm, Y=Ey, Hx=Hx, Hy=Hy, A=A, B=B
        )
        assert _tx(self.sdm, Hz=Hz, Hx=Hx, Hy=Hy, A=A, B=B) == _tf__x(
            self.sdm, Y=Hz, Hx=Hx, Hy=Hy, A=A, B=B
        )
        assert _ty(self.sdm, Hz=Hz, Hx=Hx, Hy=Hy, A=A, B=B) == _tf__y(
            self.sdm, Y=Hz, Hx=Hx, Hy=Hy, A=A, B=B
        )

    def test_tf_from_cross_powers(self):
        tf_from_cross_powers(
            self.sdm,
            station_id=self.station_ids[0],
            remote=self.station_ids[1],
        )


def main():
    unittest.main()


if __name__ == "__main__":
    main()
