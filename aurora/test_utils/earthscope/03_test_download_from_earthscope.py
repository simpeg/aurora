"""

Flow:
Use stage 2 output csv
For each such row,
    extract the network/station_id (if the metadata exist)
    download data



"""


import pandas as pd
import time

from aurora.test_utils.earthscope.data_availability import DataAvailability
from aurora.test_utils.earthscope.data_availability import row_to_request_df
from aurora.test_utils.earthscope.filter_repair import repair_missing_filters
from aurora.test_utils.earthscope.helpers import DATA_PATH
from aurora.test_utils.earthscope.helpers import get_most_recent_summary_filepath
#from aurora.test_utils.earthscope.helpers import get_summary_table_schema
from aurora.test_utils.earthscope.helpers import USE_CHANNEL_WILDCARDS
from aurora.test_utils.earthscope.widescale_test import WidesScaleTest
from mth5.mth5 import MTH5
from mth5.clients import FDSN


STAGE_ID = 3
MTH5_VERSION = "0.2.0"
TRY_REPAIR_MISSING_FILTERS = True
RAISE_EXCEPTION_IF_DATA_AVAILABILITY_EMPTY = False



class TestBuildMTH5(WidesScaleTest):

    def __init__(self, **kwargs):
        """
        data_availability_exception: bool
        If True, raise exception of DataAvailablty empty
        """
        #super(WidesScaleTest, self).__init__(**kwargs)
        super().__init__(**kwargs)
        # self.augment_with_existing = kwargs.get("augment_with_existing", True)
        self._data_availability = None
        data_availability_exception = kwargs.get("data_availability_exception", False)
        self.use_channel_wildcards = kwargs.get("use_channel_wildcards", False)
        self.max_number_download_attempts = kwargs.get("max_number_download_attempts", 3)
        self.try_repair_missing_filters = kwargs.get("try_repair_missing_filters", True)

    @property
    def data_availability(self):
        if self._data_availability is None:
            self._data_availability = DataAvailability()
        return self._data_availability

    def prepare_jobs_dataframe(self):
        schema = self.get_dataframe_schema()
        source_csv = get_most_recent_summary_filepath(2)
        source_df = pd.read_csv(source_csv)
        df = source_df.copy(deep=True)
        renamer_dict = {"filesize": "metadata_filesize",
                        "filename": "metadata_filename"}
        df = df.rename(columns=renamer_dict)
        for col in schema:
            default = col.default
            if col.dtype == "int64":
                default = int(default)
            if col.dtype == "bool":
                default = bool(int(default))
            if col.name not in df.columns:
                df[col.name] = default
                if col.dtype == "string":
                    df[col.name] = ""
                    df[col.name] = df[col.name].astype(str)

        to_str_cols = ["network_id", "station_id", "exception"]
        for col in to_str_cols:
            df[col] = df[col].astype(str)

        # for col in schema:
        #     if col.dtype == "string":
        #         df[col.name] = df[col.name].astype(str)
        df["num_filter_details"] = ""
        n_rows = len(df)
        info_str = f"There are {n_rows} network-station pairs"
        print(info_str)


        print("dropping impossible files")
        df = df[df.exception == "nan"]
        df.reset_index(inplace=True, drop=True)
        n_rows = len(df)
        info_str = f"There are {n_rows} network-station pairs after dropping impossible cases"
        print(info_str)
        return df

    def enrich_row(self, row):
        request_df = row_to_request_df(row, self.data_availability, verbosity=1,
                                       use_channel_wildcards=self.use_channel_wildcards)

        fdsn_object = FDSN(mth5_version=MTH5_VERSION)
        fdsn_object.client = "IRIS"

        expected_file_name = DATA_PATH.joinpath(fdsn_object.make_filename(request_df))
        if expected_file_name.exists():
            print(f"Already have data for {row.station_id}-{row.network_id}")
            row.at["data_mth5_size"] = expected_file_name.stat().st_size
            row.at["data_mth5_name"] = expected_file_name.name
            return row
        try:

            mth5_filename = fdsn_object.make_mth5_from_fdsn_client(request_df,
                                                                   interact=False,
                                                                   path=DATA_PATH)
            if self.try_repair_missing_filters:
                repair_missing_filters(mth5_filename, MTH5_VERSION, triage_units=True, add_filters_where_none=False)
            row.at["data_mth5_size"] = expected_file_name.stat().st_size
            row.at["data_mth5_name"] = expected_file_name.name
            row.at["data_mth5_exception"] = ""
            row.at["data_mth5_error_message"] = ""
        except Exception as e:
            row.at["data_mth5_size"] = 0
            row.at["data_mth5_name"] = ""
            row.at["data_mth5_exception"] = e.__class__.__name__
            row.at["data_mth5_error_message"] = e.args[0]
        return row



def review_results():
    coverage_csv = get_most_recent_summary_filepath(STAGE_ID)
    df = pd.read_csv(coverage_csv)
    exception_types = df.exception.unique()
    exception_types = [x for x in exception_types if x != "nan"]
    msg = f"Identified {len(exception_types)} exception types\n {exception_types}\n\n"
    print(msg)
    print("\n Value counts")
    print(df.exception.value_counts())


def test_repair_filters_SI_to_MT():
    from mth5.helpers import close_open_files
    close_open_files()
    mth5_paths = ["/home/kkappler/.cache/earthscope/data/EM_ORF08.h5",
                  "/home/kkappler/.cache/earthscope/data/EM_ORG08.h5"]

    for mth5_path in mth5_paths:
        repair_missing_filters(mth5_path, mth5_version=MTH5_VERSION, triage_units=True)
    #mth5_path = "/home/kkappler/.cache/earthscope/data/EM_ORF08.h5"
    # mth5_path = "/home/kkappler/.cache/earthscope/data/EM_ORG08.h5"
    # repair_missing_filters(mth5_path, mth5_version="0.2.0", triage_units=True)

def repair_all_filters_and_units():
    from mth5.helpers import close_open_files
    close_open_files()
    all_data_h5 = DATA_PATH.glob("*.h5")

    for i, mth5_path in enumerate(all_data_h5):
        if i>14:
            print(f"repairing {i} {mth5_path.name}")
            repair_missing_filters(mth5_path, mth5_version=MTH5_VERSION, triage_units=True)
    print("ALL DONE")

def main():
    # test_repair_filters_SI_to_MT()
    # repair_all_filters_and_units()

    t0 = time.time()
    tester = TestBuildMTH5(stage_id=STAGE_ID,
                              save_csv=True,
                              use_channel_wildcards=USE_CHANNEL_WILDCARDS,
                              data_availability_exception=RAISE_EXCEPTION_IF_DATA_AVAILABILITY_EMPTY,
                              try_repair_missing_filters=TRY_REPAIR_MISSING_FILTERS,
                              )
    # tester.startrow = 1679
    # tester.endrow = 1680
    tester.run_test()
    review_results()
    total_time_elapsed = time.time() - t0
    print(f"Total scraping & review time {total_time_elapsed:.2f}s using {tester.n_partitions} partitions")

if __name__ == "__main__":
    main()
