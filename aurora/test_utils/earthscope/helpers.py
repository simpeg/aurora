"""
CACHE_PATH: This is a place where all the downloads will land, and summaray csvs will be kept
DATA_AVAILABILITY_PATH: This is a place where information about data availability will be staged
These are txt files generated by Laura's ipynb
DATA_PATH: This is where the mth5 files are archived locally

SPUD_XML_PATH
"""
import copy
import datetime
import pandas as pd
import pathlib
import re
import socket
import subprocess

from aurora.sandbox.mth5_helpers import build_request_df

## PLACEHOLDER FOR CONFIG
USE_CHANNEL_WILDCARDS = False
HOSTNAME = socket.gethostname()
HOME = pathlib.Path().home()

if "gadi" in HOSTNAME:
    CACHE_PATH = pathlib.Path("/scratch/tq84/kk9397/earthscope")
else:
    CACHE_PATH = HOME.joinpath(".cache").joinpath("earthscope")
CACHE_PATH.mkdir(parents=True, exist_ok=True)
## PLACEHOLDER FOR CONFIG

# Data Availability
DATA_AVAILABILITY_PATH = CACHE_PATH.joinpath("data_availability")
DATA_AVAILABILITY_PATH.mkdir(parents=True, exist_ok=True)
PUBLIC_DATA_AVAILABILITY_PATH = DATA_AVAILABILITY_PATH.joinpath("public")
PUBLIC_DATA_AVAILABILITY_PATH.mkdir(parents=True, exist_ok=True)
RESTRICTED_DATA_AVAILABILITY_PATH = DATA_AVAILABILITY_PATH.joinpath("restricted")
RESTRICTED_DATA_AVAILABILITY_PATH.mkdir(parents=True, exist_ok=True)
DATA_AVAILABILITY_CSV = DATA_AVAILABILITY_PATH.joinpath("MT_acquisitions.csv")

# Data (mth5s)
DATA_PATH = CACHE_PATH.joinpath("data")
DATA_PATH.mkdir(parents=True, exist_ok=True)

# MetaData (mth5s)
EXPERIMENT_PATH = CACHE_PATH.joinpath("experiments")
EXPERIMENT_PATH.mkdir(parents=True, exist_ok=True)

# Transfer Functions
AURORA_TF_PATH = CACHE_PATH.joinpath("aurora_transfer_functions")
AURORA_TF_PATH.mkdir(parents=True, exist_ok=True)

# Summary tables
SUMMARY_TABLES_PATH = CACHE_PATH.joinpath("summary_tables")
SUMMARY_TABLES_PATH.mkdir(parents=True, exist_ok=True)

SPUD_XML_PATHS = {}
SPUD_XML_PATHS["base"] = CACHE_PATH.joinpath("spud_xml")
SPUD_XML_PATHS["base"].mkdir(parents=True, exist_ok=True)
SPUD_XML_PATHS["data"] = SPUD_XML_PATHS["base"].joinpath("data")
SPUD_XML_PATHS["data"].mkdir(parents=True, exist_ok=True)
SPUD_XML_PATHS["emtf"] = SPUD_XML_PATHS["base"].joinpath("emtf")
SPUD_XML_PATHS["emtf"].mkdir(parents=True, exist_ok=True)

def strip_xml_tags(some_string):
    """
    Allows simplification of less intuitive (albeit faster) commands such as:
    cmd = f"grep 'SourceData id' {emtf_filepath} | awk -F'"'"'"' '{print $2}'"
    qq = subprocess.check_output([cmd], shell=True)
    data_id = int(qq.decode().strip())
    with
    cmd = f"grep 'SourceData id' {emtf_filepath}"
    qq = subprocess.check_output([cmd], shell=True)
    qq = strip_xml_tags(qq)
    data_id = int(qq.decode().strip())
    :param some_string:
    :return:
    """
    stripped = re.sub('<[^>]*>', '', some_string)
    return stripped

def get_via_curl(source, target):
    """
    If exit_status of 127 is returned you may need to install curl in your environment

    Note 1: EMTF spuds come as HTML, to get XML, needed to edit the curl command, adding
    -H 'Accept: application/xml'
    https://stackoverflow.com/questions/22924993/getting-webpage-data-in-xml-format-using-curl

    Parameters
    ----------
    source
    target

    Returns
    -------
    """
    cmd = f"curl -s -H 'Accept: application/xml' {source} -o {target}"
    print(cmd)
    exit_status = subprocess.call([cmd], shell=True)
    if exit_status == 0:
        return
    else:
        print(f"Failed to {cmd}")
        print(f"exit_status {exit_status}")
        if exit_status==127:
            print("you may need to install curl in your environment")
            raise Exception

def load_xml_tf(file_path):
    """
    using emtf_xml path will fail with KeyError: 'field_notes'
    :param file_path:
    :return:
    """
    from mt_metadata.transfer_functions.core import TF
    # if "15029445_EM_PAM57" in str(file_path):
    #     print("debug")
    print(f"reading {file_path}")
    spud_tf = TF(file_path)
    spud_tf.read()
    return spud_tf


def get_remotes_from_tf_not(tf_obj):
    remote_references = tf_obj.station_metadata.get_attr_from_name('transfer_function.remote_references')
    remotes = list()
    for remote_station in remote_references:
        if not len(remote_station.split('-')) > 1:
            if remote_station != station:
                remotes.append(remote_station)

    return remotes

def get_rr_info(tf_obj):
    rr_info_list = tf_obj.station_metadata.transfer_function.processing_parameters
    # this may have more than one entry .. why?
    assert len(rr_info_list) == 1
    rr_info_instance = rr_info_list[0]
    return rr_info_instance

def get_rr_type(tf_obj):
    rr_info_instance = get_rr_info(tf_obj)
    rr_type = rr_info_instance["remote_ref.type"]
    return rr_type


def get_remotes_from_tf_2(tf_obj):
    """
    A second way to get remotes
    :param tf_obj:
    :return:
    """
    attr_name = "transfer_function.remote_references"
    remote_references = tf_obj.station_metadata.get_attr_from_name(attr_name)
    remotes = list()
    for remote_station in remote_references:
        if not len(remote_station.split('-')) > 1:
            # if remote_station != station:
            remotes.append(remote_station)
    print(remote_references)
    return remotes

def get_remotes_from_tf(tf_obj):
    """
    There were 5 cases of RemoteRef type encountered when reviewing SPUD TFs
    These were:
    1. Robust Remote Reference           1452
    2. Robust Multi-Station Reference     328
    3. Robust Multi-station Reference      15
    4. Multi-Station Reference              1
    5. Merged Transfer Functions           52
    where the number of instaces of each is listed on the right.


    :param tf_obj:
    :return:
    """
    rr_info_instance = get_rr_info(tf_obj)
    if rr_info_instance["remote_ref.type"] == "Robust Remote Reference":
        # then there is only one
        try:
            remotes = [rr_info_instance["remote_info.site.id"], ]
        except KeyError:
            print(" No remote listed in xml at expected location")
            # here an an example: https: // ds.iris.edu / spudservice / data / 14862696
            return []
    else:
        remotes = get_remotes_from_tf_2(tf_obj)
    return remotes


def get_summary_table_schema(stage_number):
    """
    A place where the columns of the various summary tables are defined.
    Stages 0 and 1 are related in the sense that the summary_table of stage 1 simply involves adding columns to the
    summmary from stage 0.  The same relationship exists between stages 2 and 3.

    If we were going to properly formalize this flow, it would be good to make a json of the schema, where each column
    was associated with a dtype, a description, and a default_value.  In that way, the same script could run to prepare
    a table at any stage, taking only the schema as input.

    This
    Stage 0
    Args:
        stage_number:

    Returns:

    """
    # Stages 0 and 1
    schemata = {}
    schemata[0] = {'emtf_id': "int64", 'data_id': 'int64', 'fail': 'bool',
                'emtf_file_size': 'int64', 'emtf_xml_filebase': 'string',
                'data_file_size': 'int64', 'data_xml_filebase': 'string'}

    new_01 = {'emtf_error': 'bool',
              'emtf_exception': 'string',
              'emtf_error_message': 'string',
              'emtf_remote_ref_type': 'string',
              'emtf_remotes': 'string',
              'data_error': 'bool',
              'data_exception': 'string',
              'data_error_message': 'string',
              'data_remote_ref_type': 'string',
              'data_remotes': 'string',
              }
    schemata[1] = {**schemata[0], **new_01 }

    # Stages 2 and 3
    # Note emtf_id, data_id, data_xml_filebase are duplicated from schema 0
    schemata[2] = {"network_id":"string",
                   "station_id":"string",
                   "filename":"string",
                   "filesize":"int64",
                   "num_channels_inventory": 'int64',
                   "num_channels_h5":'int64',
                   "num_filterless_channels":"int64",
                   "num_filter_details": "string",
                   "exception":'string',
                   "error_message":'string',
                   'emtf_id': "int64", 'data_id': 'int64','data_xml_filebase': 'string'
                   }

    schemata[3] = copy.deepcopy(schemata[2])
    # rename filename,filesise to metadata_filename, metadata_filesize
    schemata[3]["metadata_filename"] = schemata[3].pop("filename")
    schemata[3]["metadata_filesize"] = schemata[3].pop("filesize")
    schemata[3]["data_mth5_size"] = "int64"
    schemata[3]["data_mth5_name"] = "string"
    schemata[3]["data_mth5_exception"] = "string"
    schemata[3]["data_mth5_error_message"] = "string"

    schemata[4] = {}
    schemata[4]["data_id"] = 'int64'
    schemata[4]["network_id"] = "string"
    schemata[4]["station_id"] = "string"
    schemata[4]["remote_id"] = "string"
    schemata[4]["filename"] = "string"
    schemata[4]["exception"] = "string"
    schemata[4]["error_message"] = "string"
    schemata[4]["data_xml_filebase"] = "string"

    try:
        return schemata[stage_number]
    except KeyError:
        print(f"Schema for stage number {stage_number} is not defined")
        return None

def timestamp_now():
    """
    helper function to make a timestamp for file labelling.
    Default behaviour is to strip milliseconds/microseconds,
    replace spaces with underscores, and colons with empty string.
    Returns:

    """
    now = datetime.datetime.now().__str__().split(".")[0].replace(" ", "_")
    now_str = now.replace(":", "")
    return now_str

def get_summary_table_filename(stage_number):
    base_names = {}
    base_names["00"] = "spud_xml_scrape"
    base_names["01"] = "spud_xml_review"
    base_names["02"] = "local_metadata_coverage"
    base_names["03"] = "local_mth5_coverage"
    base_names["04"] = "processing_review"
    base_names["05"] = "tf_comparison_review"
    stage_number_str = str(stage_number).zfill(2)

    csv_name = f"{stage_number_str}_{base_names[stage_number_str]}.csv"
    if stage_number in [1,]:
        now_str = timestamp_now()
        csv_name = csv_name.replace(".csv", f"_{now_str}.csv")

    csv_path = SUMMARY_TABLES_PATH.joinpath(csv_name)

    return csv_path


def get_most_recent_summary_filepath(stage_number):
    """
    For each stage of task 1, there is a summary table produced, and that summary table is used
    as input for the next stage of the process.  These tables are timestamped.
    Normally we want the most recent one, and we don't want to be pasting filenames all over the place
    This returns the path to the most recent table.

    :param stage_number:
    :return:
    """
    stage_number_str = str(stage_number).zfill(2)
    globby = SUMMARY_TABLES_PATH.glob(f"{stage_number_str}*")
    globby = list(globby)
    globby.sort()
    if len(globby) == 0:
        print(f"glob operation returned an empty list when looking for data from stage {stage_number_str}")
        print("Expect an IndexError")
    return globby[-1]


def load_most_recent_summary(stage_number):
    review_csv = get_most_recent_summary_filepath(stage_number)
    print(f"loading {review_csv}")
    results_df = pd.read_csv(review_csv)
    return results_df







KEEP_COLUMNS = ['emtf_id', 'data_id','data_file_size','data_xml_filebase',
                'data_error', 'data_remote_ref_type', 'data_remotes',]

def restrict_to_mda(df, RR=None, keep_columns=KEEP_COLUMNS):
    """
    Takes as input the summary from xml ingest (process 01) and restricts to rows where
    data at IRIS/Earthscope are expected.
    :param df:
    :param RR:
    :param keep_columns:
    :return:
    """
    n_xml = len(df)
    is_not_mda = df.data_xml_filebase.str.contains("__")
    n_non_mda = is_not_mda.sum()
    n_mda = len(df) - n_non_mda
    print(f"There are {n_mda} / {n_xml} files with mda string ")
    print(f"There are {n_non_mda} / {n_xml} files without mda string ")
    mda_df = df[~is_not_mda]
    mda_df.reset_index(drop=True, inplace=True)


    if RR:
        is_rrr = mda_df.data_remote_ref_type == RR
        mda_df = mda_df[is_rrr]
        mda_df.reset_index(drop=True, inplace=True)

    mda_df = mda_df[keep_columns]

    fix_nans_in_columns = ["data_remotes",]
    for col in fix_nans_in_columns:
        if col in mda_df.columns:
            mda_df[col] = mda_df[col].astype(str)
            mda_df[mda_df[col]=="nan"][col] = ""

    print("ADD NETWORK/STATION COLUMNS for convenience")
    print("Consdier PUSH THIS BACK TO TASK 01 once all XML are reading successfully")
    # Get station/Networks
    xml_source = 'data'
    n_rows = len(mda_df)
    networks = n_rows * [""]
    stations = n_rows * [""]
    for i, row in mda_df.iterrows():
        #xml_path = SPUD_XML_PATHS[xml_source].joinpath(row[f"{xml_source}_xml_filebase"])
        xml_filebase = row[f"{xml_source}_xml_filebase"]
        xml_filestem = xml_filebase.split(".")[0]
        [xml_uid, network_id, station_id] = xml_filestem.split("_")
        networks[i] = network_id
        stations[i] = station_id
    mda_df["station_id"] = stations
    mda_df["network_id"] = networks

    return mda_df


def row_to_request_df(row, data_availability_obj, verbosity=1, use_channel_wildcards=False,
                      raise_exception_if_data_availability_empty=True):
    """

    Parameters
    ----------
    row: pandas.core.series.Series
        Row of a custom dataframe used in widescale earthscope tests.
        The only information we currently take from this row is the network_id and station_id
    data_availability: This is an instance of DataAvailability object.
        the data_availability object is a global varaible in 02 and 03, and so I anticipate there
        could be issues running this in parallel ...
        This could be handled in future with webcalls
        Also, if I passed instead row.network_id, row.station_id, I could just pass
        the
    verbosity: int
        Print request df to screen, to be deprecated
    use_channel_wildcards: bool
        If True look for ["*Q*", "*F*", ]

    Returns
    -------

    """
    time_period_dict = {}
    network_id = row.network_id
    station_id = row.station_id
    if use_channel_wildcards:
        availabile_channels = ["*Q*", "*F*", ]
    else:
        availabile_channels = data_availability_obj.get_available_channels(network_id, station_id)
        for ch in availabile_channels:
            tp = data_availability_obj.get_available_time_period(network_id, station_id, ch)
            time_period_dict[ch] = tp

    if len(availabile_channels) == 0:
        if raise_exception_if_data_availability_empty:
            msg = f"No data from {network_id}_{station_id}"
            raise DataAvailabilityException(msg)
        else:
            print("Setting channels to wildcards because local data_availabilty query returned empty list")
            availabile_channels = ["*Q*", "*F*", ]

    request_df = build_request_df(network_id, station_id,
                                  channels=availabile_channels,
                                  start=None, end=None,
                                  time_period_dict=time_period_dict)
    if verbosity > 1:
        print(f"request_df: \n {request_df}")
    return request_df

def test_summary_table_schema():
    get_summary_table_schema(0)
    get_summary_table_schema(1)
    print("OK")

if __name__ == "__main__":
    try:
        assert(1==0)
    except Exception as e:
        raise DataAvailabilityException("put message here")
    test_summary_table_schema()
