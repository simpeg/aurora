"""
Iterate over rows of stage 1 output csv, selecting only rows where the name is of the form:
18057859_EM_MH010.xml
uid_NETWORK_STATION.xml

For each such row, we make a list of stations that were identified
as self or RR

For every station in list:
    get metadata
    show number of channels
    any other pertinent information
"""



import numpy as np
import pandas as pd
import requests
import time

from aurora.sandbox.mth5_helpers import get_experiment_from_obspy_inventory
from aurora.sandbox.mth5_helpers import mth5_from_experiment
from aurora.sandbox.mth5_helpers import enrich_channel_summary

from aurora.test_utils.earthscope.data_availability import DataAvailability
from aurora.test_utils.earthscope.data_availability import DataAvailabilityException
from aurora.test_utils.earthscope.data_availability import row_to_request_df
from aurora.test_utils.earthscope.data_availability import url_maker
from aurora.test_utils.earthscope.helpers import EXPERIMENT_PATH
from aurora.test_utils.earthscope.helpers import get_most_recent_summary_filepath
from aurora.test_utils.earthscope.helpers import restrict_to_mda
from aurora.test_utils.earthscope.helpers import SUMMARY_TABLES_PATH
from aurora.test_utils.earthscope.helpers import timestamp_now
from aurora.test_utils.earthscope.helpers import USE_CHANNEL_WILDCARDS
from aurora.test_utils.earthscope.widescale import WidesScaleTest
from mth5.mth5 import MTH5
from mth5.clients import FDSN

STAGE_ID = 2
KNOWN_NON_EARTHCSCOPE_STATIONS = ["FRD", ]


# CONFIG
MTH5_VERSION = "0.2.0"


class TestDatalessMTH5(WidesScaleTest):

    def __init__(self, **kwargs):
        """
        data_availability_exception: bool
        If True, raise exception of DataAvailablty empty
        """
        #super(WidesScaleTest, self).__init__(**kwargs)
        super().__init__(**kwargs)
        self.augment_with_existing = kwargs.get("augment_with_existing", True)
        self.use_skeleton = kwargs.get("use_skeleton", True) # speeds up preparation of dataframe
        self.skeleton_file = f"skeleton_{str(STAGE_ID).zfill(2)}.csv"
        self.xml_source = "data" # "data" or "emtf"
        self._data_availability = None
        self.use_channel_wildcards = kwargs.get("use_channel_wildcards", False)
        self.data_availability_exception = kwargs.get("data_availability_exception", True)
        self.max_number_download_attempts = kwargs.get("max_number_download_attempts", 3)


    @property
    def data_availability(self):
        if self._data_availability is None:
            self._data_availability = DataAvailability()
        return self._data_availability

    def prepare_jobs_dataframe(self, source_csv=None):
        """
        Define the data structure that is output from this stage of processing
        """
        schema = self.get_dataframe_schema()
        def initialize_metadata_df():
            """ """
            schema = self.get_dataframe_schema()
            column_names = [x.name for x in schema]
            df = pd.DataFrame(columns=column_names)
            return df

        if self.use_skeleton:
            df = pd.read_csv(self.skeleton_file)
            return df

        # ? Allow to start from a previous, partially  filled csv?
        # try:
        #     coverage_csv = get_most_recent_summary_filepath(STAGE_ID)
        #     coverage_df = pd.read_csv(coverage_csv)
        # except FileNotFoundError:
        #     coverage_df = initialize_metadata_df()
        coverage_df = initialize_metadata_df()

        if not source_csv:
            source_csv = get_most_recent_summary_filepath(1)
        spud_df = pd.read_csv(source_csv)
        spud_df = restrict_to_mda(spud_df)
        print(f"Restricting spud_df to mda (Earthscope) entries: {len(spud_df)} rows")

        for i_row, row in spud_df.iterrows():
            # Ignore XML that cannot be read
            if row[f"{self.xml_source}_error"] is True:
                print(f"Skipping {row.emtf_id} for now, tf not reading in")
                continue

            # Sort out remotes
            remotes = row.data_remotes.split(",")
            if len(remotes) == 1:
                if remotes[0] == "nan":
                    remotes = []
            if remotes:
                print(f"remotes: {remotes} ")

            all_stations = [row.station_id, ] + remotes
            network_id = row.network_id
            for original_station_id in all_stations:
                # handle wonky station_ids, ignore ambigous or incorrect labels
                station_id = analyse_station_id(original_station_id)
                if not station_id:
                    continue

                #only assign known values, set other columns en masse later
                new_row = {"station_id": station_id,
                           "network_id": network_id,
                           "emtf_id": row.emtf_id,
                           "data_id": row.data_id,
                           "data_xml_filebase": row.data_xml_filebase}
                # of course we should collect all the dictionaries first and then build the df,
                # this is inefficient, but tis a work in progress.
                coverage_df = coverage_df.append(new_row, ignore_index=True)

        # Now have coverage df, but need to uniquify it
        print(len(coverage_df))
        subset = ['network_id', 'station_id']
        ucdf = coverage_df.drop_duplicates(subset=subset, keep='first')
        print(len(ucdf))
        # Assign default values to columns
        for col in schema:
            if not ucdf[col.name].any():
                ucdf[col.name] = col.default
                if col.dtype == "string":
                    ucdf[col.name] = ""
        ucdf.to_csv(self.skeleton_file, index=False)
        return ucdf


    def enrich_row(self, row):
        """
        This will eventually get used by dask, but as a step we need to make this a method
        that works with df.apply()
        Returns:

        """
        try:
            request_df = row_to_request_df(row, self.data_availability, verbosity=1,
                                           use_channel_wildcards=self.use_channel_wildcards,
                                           raise_exception_if_data_availability_empty=self.data_availability_exception)

        except Exception as e:
            print(f"{e}")
            row["num_channels_inventory"] = 0
            row["num_channels_h5"] = 0
            row["exception"] = e.__class__.__name__
            row["error_message"] = e.args[0]
            return row

        fdsn_object = FDSN(mth5_version=MTH5_VERSION)
        fdsn_object.client = "IRIS"
        expected_file_name = EXPERIMENT_PATH.joinpath(fdsn_object.make_filename(request_df))

        if expected_file_name.exists():
            print(f"Already have data for {row.network_id}-{row.station_id}")

            if self.augment_with_existing:
                m = MTH5()
                m.open_mth5(expected_file_name)
                channel_summary_df = get_augmented_channel_summary(m)
                m.close_mth5()
                add_row_properties(expected_file_name, channel_summary_df, row)
        else:
            n_tries = 0
            while n_tries < self.max_number_download_attempts:
                try:
                    inventory, data = fdsn_object.get_inventory_from_df(request_df, data=False)
                    n_ch_inventory = len(inventory.networks[0].stations[0].channels)
                    row["num_channels_inventory"] = n_ch_inventory
                    experiment = get_experiment_from_obspy_inventory(inventory)
                    m = mth5_from_experiment(experiment, expected_file_name)
                    m.channel_summary.summarize()
                    channel_summary_df = get_augmented_channel_summary(m)
                    m.close_mth5()
                    add_row_properties(expected_file_name, channel_summary_df, row)
                    n_tries = self.max_number_download_attempts
                except Exception as e:
                    print(f"{e}")
                    row["num_channels_inventory"] = 0
                    row["num_channels_h5"] = 0
                    row["exception"] = e.__class__.__name__
                    row["error_message"] = e.args[0]
                    n_tries += 1
                    if e.__class__.__name__ == "DataAvailabilityException":
                        n_tries = self.max_number_download_attempts
        return row




def analyse_station_id(station_id):
    """
    Helper function that is not very robust, but specific to handling station_ids in the 2023 earthscope tests,
    in particular, parsing non-schema defined reference stations from SPUD TF XML.

    Most of the station ids at IRIS/Earthscope have the following format:
    [ST][L][ZZ] where ST is the standard two-digit abbreviation for state, L is a letter [A-Z], and ZZ is an integer,
    zero-padded to two chars [01-99].

    In the initial run of widescale tests the were not very many exceptions to this rule and they were handled here.
    However, not that the XML reader has matured, there are many new cases.

    Some exceptions:
    CAM01 has reference K1x.
    It maybe reasonable to assume this means CAK01

    IDA11 has reference A10x.
    It maybe reasonable to assume this means IDA10

    IDJ10 has reference L6x.
    It maybe reasonable to assume this means IDL06

    ORH02 has reference M1x.
    It maybe reasonable to assume this means ORM01

    There are a lot of exceptions however, and it looks as though
    we may want to add an extra function or stage that validates the station codes make sense.

    Three character codes
    Four character codes




    Parameters
    ----------
    station_id

    Returns
    -------

    """
    if station_id in KNOWN_NON_EARTHCSCOPE_STATIONS:
        print(f"skipping {station_id} -- it's not an earthscope station")
        return None
    if len(station_id) == 0:
        print("NO Station ID")
        return None
    if len(station_id) in [1,2]:
        print(f"Unexpected station_id {station_id} of length {len(station_id)}")
        return None
    elif station_id[-1] == "x":
        print(f"Expected number at end of station_id {station_id}, but found a lowercase x")
    elif len(station_id) == 3:
        print(f"Maybe forgot to archive the TWO-CHAR STATE CODE onto station_id {station_id}")
    elif len(station_id) == 4:
        print(f"?? Typo in station_id {station_id}??")
    elif len(station_id) > 5:
        print(f"run labels probably tacked onto station_id {station_id}")
        print("Can we confirm that FDSN has a max 5CHAR station ID code???")
        station_id = station_id[0:5]
        print(f"Setting to first 5 chars: {station_id}")
    elif len(station_id)==5:
        pass # looks normal
    else:
        print(f"Havent encountered case len(station_id)={len(station_id)}")
        raise NotImplementedError
    return station_id



def get_augmented_channel_summary(m):
    channel_summary_df = m.channel_summary.to_dataframe()
    channel_summary_df = enrich_channel_summary(m, channel_summary_df, "num_filters")
    channel_summary_df = enrich_channel_summary(m, channel_summary_df, "filter_units_in")
    channel_summary_df = enrich_channel_summary(m, channel_summary_df, "filter_units_out")
    return channel_summary_df


def add_row_properties(expected_file_name, channel_summary_df, row):
    num_filterless_channels = len(channel_summary_df[channel_summary_df.num_filters == 0])
    n_ch_h5 = len(channel_summary_df)
    row["filename"] = expected_file_name.name
    row["filesize"] = expected_file_name.stat().st_size
    row["num_filterless_channels"] = num_filterless_channels
    aa = channel_summary_df.component.to_list()
    bb = channel_summary_df.num_filters.to_list()
    row["num_filter_details"] = str(dict(zip(aa, bb)))

    cc = channel_summary_df.filter_units_in.to_list()
    row["filter_units_in_details"] = str(dict(zip(aa, cc)))
    dd = channel_summary_df.filter_units_out.to_list()
    row["filter_units_out_details"] = str(dict(zip(aa, dd)))

    # new_row["num_channels_inventory"] = n_ch_inventory
    row["num_channels_h5"] = n_ch_h5
    row["exception"] = ""
    row["error_message"] = ""
    #return row



def scan_data_availability_exceptions():
    """


    -------

    """
    coverage_csv = get_most_recent_summary_filepath(STAGE_ID)
    df = pd.read_csv(coverage_csv)
    sub_df = df[df["exception"]=="DataAvailabilityException"]
    #sub_df = df
    print(len(sub_df))
    for i, row in sub_df.iterrows():
        print(i)
        url = url_maker(row.network_id, row.station_id)
        response = requests.get(url)
        if response.status_code == 200:
            print('Web site exists')
            raise NotImplementedError
        else:
            print(f'Web site does not exist {response.status_code}')
    return

def review_results():
    now_str = timestamp_now()
    exceptions_summary_filebase = f"02_exceptions_summary_{now_str}.txt"
    exceptions_summary_filepath = SUMMARY_TABLES_PATH.joinpath(exceptions_summary_filebase)


    coverage_csv = get_most_recent_summary_filepath(STAGE_ID)
    df = pd.read_csv(coverage_csv)

    to_str_cols = ["network_id", "station_id", "exception"]
    for str_col in to_str_cols:
        df[str_col] = df[str_col].astype(str)

    with open(exceptions_summary_filepath, 'w') as f:
        msg = "*** EXCEPTIONS SUMMARY *** \n\n"
        print(msg)
        f.write(msg)

        exception_types = df.exception.unique()
        exception_types = [x for x in exception_types if x!="nan"]
        msg = f"Identified {len(exception_types)} exception types\n {exception_types}\n\n"
        print(msg)
        f.write(msg)


        exception_counts = {}
        for exception_type in exception_types:
            exception_df = df[df.exception == exception_type]
            n_exceptions = len(exception_df)
            unique_errors = exception_df.error_message.unique()
            n_unique_errors = len(unique_errors)
            msg = f"{n_exceptions} instances of {exception_type}, with {n_unique_errors} unique error(s)\n"
            print(msg)
            f.write(msg)
            print(unique_errors, "\n\n")
            msg = [f"{x}\n" for x in unique_errors]
            f.write("".join(msg) + "\n\n")
            exception_counts[exception_type] = len(exception_df)
            if exception_type == "IndexError":
                out_csv_filebase = f"02_index_error_exceptions__questionable_data_availability.csv"
                out_csv = SUMMARY_TABLES_PATH.joinpath(out_csv_filebase)
                exception_df.to_csv(out_csv, index=False)

        grouper = df.groupby(["network_id", "station_id"])
        msg = f"\n\nThere were {len(grouper)} unique network-station pairs in {len(df)} rows\n\n"
        print(msg)
        f.write(msg)
        print(exception_counts)
        f.write(str(exception_counts))
        msg = f"TOTAL #Exceptions {np.array(list(exception_counts.values())).sum()} of {len(df)} Cases"
        print(msg)
        f.write(msg)
    return

def exception_analyser():
    """like batch_download, but will only try to pull selected row ids"""
    # batch_download_metadata_v2(row_start=853, row_end=854) #EM AB718 FDSNNoDataException
    #batch_download_metadata_v2(row_start=1399, row_end=1400) # ZU COR22 NotImplementedError
    # batch_download_metadata_v2(row_start=1337, row_end=1338) #
    #batch_download_metadata_v2(row_start=1784, row_end=1785) # ZU Y30 TypeError
    # batch_download_metadata_v2(row_start=613, row_end=614)  # EM OHM52 FDSNTimeoutException
    #batch_download_metadata_v2(row_start=1443, row_end=1444)  # 8P REU09 TypeError
    batch_download_metadata_v2(row_start=1487, row_end=1488)  # 8P REX11 IndexError
    #Not all channels are present in the archive for the duration of the stations existence.



def main():
    t0 = time.time()
    tester = TestDatalessMTH5(stage_id=STAGE_ID,
                              save_csv=True,
                              data_availability_exception=True,
                              use_channel_wildcards=USE_CHANNEL_WILDCARDS,
                              use_skeleton=False,
                              augment_with_existing=True,
                              )
    # tester.endrow = 5
    tester.run_test()
    # exception_analyser()
    # scan_data_availability_exceptions()
    review_results()
    total_time_elapsed = time.time() - t0
    print(f"Total scraping & review time {total_time_elapsed:.2f}s using {tester.n_partitions} partitions")

if __name__ == "__main__":
    main()
