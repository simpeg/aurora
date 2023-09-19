from mth5.utils.helpers import initialize_mth5
from aurora.test_utils.synthetic.paths import DATA_PATH
from mth5.helpers import close_open_files

import datetime


def fix_time(tstmp):
    year = tstmp.year
    month = tstmp.month
    day = tstmp.day
    hour = tstmp.hour
    minute = tstmp.minute
    second = tstmp.second
    out = datetime.datetime(year, month, day, hour, minute, second)
    return out


def test_can_slice_a_run_ts_using_timestamp():
    close_open_files()
    mth5_path = DATA_PATH.joinpath("test1.h5")
    mth5_obj = initialize_mth5(mth5_path, "r")
    df = mth5_obj.channel_summary.to_dataframe()
    run_001 = mth5_obj.get_run("test1", "001")
    run_ts_01 = run_001.to_runts()
    start = df.iloc[0].start
    end = df.iloc[0].end
    print(" Workaround for mt_metadata issue #86 - remove once that is resolved")
    start = fix_time(start)
    end = fix_time(end)
    run_ts_02 = run_001.to_runts(start=start, end=end)
    run_ts_03 = run_001.to_runts(
        start=start, end=end + datetime.timedelta(microseconds=499999)
    )

    run_ts_04 = run_001.to_runts(
        start=start, end=end + datetime.timedelta(microseconds=500000)
    )
    print(f"run_ts_01 has {len(run_ts_01.dataset.ex.data)} samples")
    print(f"run_ts_02 has {len(run_ts_02.dataset.ex.data)} samples")
    print(f"run_ts_03 has {len(run_ts_03.dataset.ex.data)} samples")
    print(f"run_ts_04 has {len(run_ts_04.dataset.ex.data)} samples")


def main():
    test_can_slice_a_run_ts_using_timestamp()


if __name__ == "__main__":
    main()
