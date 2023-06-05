"""
CACHE_PATH: This is a place where all the downloads will land, and summaray csvs will be kept
DATA_AVAILABILITY_PATH: This is a place where information about data availability will be staged
These are txt files generated by Laura's ipynb
DATA_PATH: This is where the mth5 files are archived locally

SPUD_XML_PATH
"""
import datetime
import pathlib

import pandas as pd

HOME = pathlib.Path().home()
CACHE_PATH = HOME.joinpath(".cache").joinpath("earthscope")
CACHE_PATH.mkdir(parents=True, exist_ok=True)

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

# Summary tables
SUMMARY_TABLES_PATH = CACHE_PATH.joinpath("summary_tables")
SUMMARY_TABLES_PATH.mkdir(parents=True, exist_ok=True)

SPUD_XML_PATH = CACHE_PATH.joinpath("spud_xml")
SPUD_XML_CSV = SPUD_XML_PATH.joinpath("spud_summary.csv")
SPUD_EMTF_PATH = SPUD_XML_PATH.joinpath("emtf")
SPUD_DATA_PATH = SPUD_XML_PATH.joinpath("data")
SPUD_EMTF_PATH.mkdir(parents=True, exist_ok=True)
SPUD_DATA_PATH.mkdir(parents=True, exist_ok=True)

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

def build_request_df(station_id, network_id, channels=None, start=None, end=None):
    if channels is None:
        channels = "*"
        print("this doesn't work")

    # need this for columns
    from mth5.clients import FDSN
    fdsn_object = FDSN(mth5_version='0.2.0')
    fdsn_object.client = "IRIS"
    if start is None:
        start = '1970-01-01 00:00:00'
    if end is None:
        end = datetime.datetime.now()
        end = end.replace(hour=0, minute=0, second=0, microsecond=0)

    request_list = []
    for channel in channels:
        request_list.append([network_id, station_id, '', channel, start, end])

    print(request_list)

    request_df = pd.DataFrame(request_list, columns=fdsn_object.request_columns)
    return request_df


def get_summary_table_filename(stage_number):
    base_names = {}
    base_names["01"] = "spud_xml_review"
    base_names["02"] = "local_metadata_coverage"
    base_names["03"] = "local_mth5_coverage"
    stage_number_str = str(stage_number).zfill(2)

    now = datetime.datetime.now().__str__().split(".")[0].replace(" ", "_")
    now_str = now.replace(":", "")
    csv_name = f"{stage_number_str}_{base_names[stage_number_str]}.csv"
    if stage_number in [1,]:
        now = datetime.datetime.now().__str__().split(".")[0].replace(" ", "_")
        now_str = now.replace(":", "")
        csv_name = csv_name.replace(".csv", f"_{now_str}.csv")

    csv_path = SUMMARY_TABLES_PATH.joinpath(csv_name)

    return csv_path


def get_most_recent_review(stage_number):
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
    return globby[-1]


def load_data_availability_dfs():
    output = {}
    globby = PUBLIC_DATA_AVAILABILITY_PATH.glob("*txt")
    for txt_file in globby:
        print(txt_file)
        network_id = txt_file.name.split("_")[-1].split(".txt")[0]
        df = pd.read_csv(txt_file, parse_dates=['Earliest', 'Latest', ])
        output[network_id] = df
        print(f"loaded {network_id}")
    return output
