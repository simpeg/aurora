# import logging
import unittest

import numpy as np
import pandas as pd
from aurora.pipelines.run_summary import RunSummary
from aurora.test_utils.synthetic.make_mth5_from_asc import create_test12rr_h5
from aurora.test_utils.synthetic.paths import DATA_PATH


class TestRunSummary(unittest.TestCase):
    """ """

    @classmethod
    def setUpClass(self):
        self._mth5_path = DATA_PATH.joinpath("test12rr.h5")
        if not self._mth5_path.exists():
            self._mth5_path = create_test12rr_h5()
        self._rs = RunSummary()
        self._rs.from_mth5s(
            [
                self._mth5_path,
            ]
        )

    def test_add_duration(self):
        rs = self._rs.clone()
        rs.add_duration()
        assert "duration" in rs.df.columns


class TestRunSummaryValidation(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.df_bad_outputs = pd.DataFrame(
            {
                "survey": {2: "LD2024", 4: "LD2024", 6: "LD2024"},
                "station_id": {2: "12", 4: "12", 6: "12"},
                "run_id": {
                    2: "sr4096_0002",
                    4: "sr4096_0004",
                    6: "sr4096_0006",
                },
                "start": {
                    2: pd.Timestamp("2024-05-09 00:59:58+0000", tz="UTC"),
                    4: pd.Timestamp("2024-05-09 06:59:58+0000", tz="UTC"),
                    6: pd.Timestamp("2024-05-09 12:59:58+0000", tz="UTC"),
                },
                "end": {
                    2: pd.Timestamp(
                        "2024-05-09 01:09:41.997070312+0000", tz="UTC"
                    ),
                    4: pd.Timestamp(
                        "2024-05-09 07:09:41.996582031+0000", tz="UTC"
                    ),
                    6: pd.Timestamp(
                        "2024-05-09 13:09:41.996338+0000", tz="UTC"
                    ),
                },
                "sample_rate": {2: 4096.0, 4: 4096.0, 6: 4096.0},
                "input_channels": {
                    2: ["hx", "hy"],
                    4: ["hx", "hy"],
                    6: ["hx", "hy"],
                },
                "output_channels": {
                    2: ["ey", "hz"],
                    4: ["ex", "ey", "hz"],
                    6: ["ex", "ey", "hz"],
                },
                "channel_scale_factors": {
                    2: {"ey": 1.0, "hx": 1.0, "hy": 1.0, "hz": 1.0},
                    4: {"ex": 1.0, "ey": 1.0, "hx": 1.0, "hy": 1.0, "hz": 1.0},
                    6: {"ex": 1.0, "ey": 1.0, "hx": 1.0, "hy": 1.0, "hz": 1.0},
                },
                "valid": {2: True, 4: True, 6: True},
                "mth5_path": {
                    2: "path1\test.h5",
                    4: "path1\test.h5",
                    6: "path1\test.h5",
                },
            }
        )
        self.df_bad_inputs = pd.DataFrame(
            {
                "survey": {2: "LD2024", 4: "LD2024", 6: "LD2024"},
                "station_id": {2: "12", 4: "12", 6: "12"},
                "run_id": {
                    2: "sr4096_0002",
                    4: "sr4096_0004",
                    6: "sr4096_0006",
                },
                "start": {
                    2: pd.Timestamp("2024-05-09 00:59:58+0000", tz="UTC"),
                    4: pd.Timestamp("2024-05-09 06:59:58+0000", tz="UTC"),
                    6: pd.Timestamp("2024-05-09 12:59:58+0000", tz="UTC"),
                },
                "end": {
                    2: pd.Timestamp(
                        "2024-05-09 01:09:41.997070312+0000", tz="UTC"
                    ),
                    4: pd.Timestamp(
                        "2024-05-09 07:09:41.996582031+0000", tz="UTC"
                    ),
                    6: pd.Timestamp(
                        "2024-05-09 13:09:41.996338+0000", tz="UTC"
                    ),
                },
                "sample_rate": {2: 4096.0, 4: 4096.0, 6: 4096.0},
                "input_channels": {
                    2: ["hx", "hy"],
                    4: ["hx"],
                    6: ["hx", "hy"],
                },
                "output_channels": {
                    2: ["ex", "ey", "hz"],
                    4: ["ex", "ey", "hz"],
                    6: ["ex", "ey", "hz"],
                },
                "channel_scale_factors": {
                    2: {"ey": 1.0, "hx": 1.0, "hy": 1.0, "hz": 1.0},
                    4: {"ex": 1.0, "ey": 1.0, "hx": 1.0, "hy": 1.0, "hz": 1.0},
                    6: {"ex": 1.0, "ey": 1.0, "hx": 1.0, "hy": 1.0, "hz": 1.0},
                },
                "valid": {2: True, 4: True, 6: True},
                "mth5_path": {
                    2: "path1\test.h5",
                    4: "path1\test.h5",
                    6: "path1\test.h5",
                },
            }
        )

    def test_bad_outputs(self):
        rs = RunSummary(df=self.df_bad_outputs)
        rs.validate_channels()

        self.assertEqual(
            True, np.all([False, True, True] == rs.df.valid.values)
        )

    def test_bad_inputs(self):
        rs = RunSummary(df=self.df_bad_inputs)
        rs.validate_channels()

        self.assertEqual(
            True, np.all([True, False, True] == rs.df.valid.values)
        )

    def test_bad_outputs_drop(self):
        rs = RunSummary(df=self.df_bad_outputs)
        rs.validate_channels(drop=True)

        self.assertEqual(True, np.all([True, True] == rs.df.valid.values))

    def test_bad_inputs_drop(self):
        rs = RunSummary(df=self.df_bad_inputs)
        rs.validate_channels(drop=True)

        self.assertEqual(True, np.all([True, True] == rs.df.valid.values))

    def test_duration(self):
        rs = RunSummary(df=self.df_bad_outputs)
        rs.add_duration()

        self.assertEqual(
            True,
            np.isclose(
                np.array([583.99707, 583.996582, 583.996338]), rs.df.duration
            ).all(),
        )


if __name__ == "__main__":
    unittest.main()
