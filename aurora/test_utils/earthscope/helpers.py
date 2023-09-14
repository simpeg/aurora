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

# PLACEHOLDER FOR CONFIG
USE_CHANNEL_WILDCARDS = False
HOSTNAME = socket.gethostname()
HOME = pathlib.Path().home()

if "gadi" in HOSTNAME:
    CACHE_PATH = pathlib.Path("/scratch/tq84/kk9397/earthscope")
else:
    CACHE_PATH = HOME.joinpath(".cache").joinpath("earthscope")
CACHE_PATH.mkdir(parents=True, exist_ok=True)


# Data Availability
DATA_AVAILABILITY_PATHS = {}
DATA_AVAILABILITY_PATHS["base"] = CACHE_PATH.joinpath("data_availability")
DATA_AVAILABILITY_PATHS["base"].mkdir(parents=True, exist_ok=True)
DATA_AVAILABILITY_PATHS["public"] = DATA_AVAILABILITY_PATHS["base"].joinpath("public")
DATA_AVAILABILITY_PATHS["public"].mkdir(parents=True, exist_ok=True)
DATA_AVAILABILITY_PATHS["restricted"] = DATA_AVAILABILITY_PATHS["base"].joinpath("restricted")
DATA_AVAILABILITY_PATHS["restricted"].mkdir(parents=True, exist_ok=True)
DATA_AVAILABILITY_CSV = DATA_AVAILABILITY_PATHS["base"].joinpath("MT_acquisitions.csv")

# Data (mth5s)
DATA_PATH = CACHE_PATH.joinpath("data")
DATA_PATH.mkdir(parents=True, exist_ok=True)

# MetaData (mth5s)
EXPERIMENT_PATH = CACHE_PATH.joinpath("dataless_mth5")
EXPERIMENT_PATH.mkdir(parents=True, exist_ok=True)

# Transfer Functions
AURORA_TF_PATH = CACHE_PATH.joinpath("aurora_transfer_functions")
AURORA_TF_PATH.mkdir(parents=True, exist_ok=True)
AURORA_Z_PATH = AURORA_TF_PATH.joinpath("Z")
AURORA_Z_PATH.mkdir(parents=True, exist_ok=True)

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



def get_summary_table_schema_v2(stage_number):
    """
    Update the summary_table_schema to use mt_metadata style standards and csv schema defns

    Rather than providing dicts with key:dtype its better to have:
    col.name = 'emtf_id'
    col.dtype = "int64"
    col.default = -1
    etc.
    
    Parameters
    ----------
    stage_number

    Returns
    -------

    """
    if stage_number in [0, 1, 2, 3, 4]:
        from aurora.test_utils.earthscope.metadata import make_schema_list
        schema = make_schema_list(stage_number)
        return schema
    else:
        msg = f"Schema not defined for stage_id {stage_number}"
        print(msg)
        raise NotImplementedError


def get_summary_table_schema(stage_number):
    """
    A place where the columns of the various summary tables are defined.
    Stages 0 and 1 are related in the sense that the summary_table of stage 1 simply involves adding columns to the
    summmary from stage 0.  The same relationship exists between stages 2 and 3.

    If we were going to properly formalize this flow, it would be good to make a json of the schema, where each column
    was associated with a dtype, a description, and a default_value.  In that way, the same script could run to prepare
    a table at any stage, taking only the schema as input.

    """
    print("TO BE DEPRECATED and replaced by CSV SCHEMA")
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
                   "filter_units_in_details": "string",
                   "filter_units_out_details": "string",
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


def load_most_recent_summary(stage_number, dtypes=None):
    review_csv = get_most_recent_summary_filepath(stage_number)
    print(f"loading {review_csv}")
    results_df = pd.read_csv(review_csv, dtype=dtypes)
    return results_df







KEEP_COLUMNS = ['emtf_id', 'data_id','data_file_size','data_xml_filebase',
                'data_error', 'data_processing_type', 'data_remotes',]

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
        is_rrr = mda_df.data_processing_type == RR
        mda_df = mda_df[is_rrr]
        mda_df.reset_index(drop=True, inplace=True)

    mda_df = mda_df[keep_columns]

    fix_nans_in_columns = ["data_remotes",]
    for col in fix_nans_in_columns:
        if col in mda_df.columns:
            mda_df[col] = mda_df[col].astype(str)
            # OLD
            # mda_df[mda_df[col]=="nan"][col] = ""
            # NEW
            mda_df[mda_df[col].isin(["<NA>", "nan"])][col] = ""

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


def none_or_str(value):
    """
    argparse helper that allows us to cast to None, see FraggaMuffin's response in this link:
    https://stackoverflow.com/questions/48295246/how-to-pass-none-keyword-as-command-line-argument
    """
    if value == 'None':
        return None
    return value

def test_summary_table_schema():
    get_summary_table_schema(0)
    get_summary_table_schema(1)
    get_summary_table_schema(2)
    get_summary_table_schema(3)
    print("OK")

if __name__ == "__main__":
    test_summary_table_schema()
