import datetime
import pathlib

import pandas as pd

HOME = pathlib.Path().home()
CACHE_PATH = HOME.joinpath(".cache").joinpath("earthscope")
CACHE_PATH.mkdir(parents=True, exist_ok=True)

DATA_AVAILABILITY_PATH = CACHE_PATH.joinpath("data_availability")
DATA_AVAILABILITY_PATH.mkdir(parents=True, exist_ok=True)
PUBLIC_DATA_AVAILABILITY_PATH = DATA_AVAILABILITY_PATH.joinpath("public")
PUBLIC_DATA_AVAILABILITY_PATH.mkdir(parents=True, exist_ok=True)
RESTRICTED_DATA_AVAILABILITY_PATH = DATA_AVAILABILITY_PATH.joinpath("restricted")
RESTRICTED_DATA_AVAILABILITY_PATH.mkdir(parents=True, exist_ok=True)
DATA_AVAILABILITY_CSV = DATA_AVAILABILITY_PATH.joinpath("MT_acquisitions.csv")

DATA_PATH = CACHE_PATH.joinpath("data")
DATA_PATH.mkdir(parents=True, exist_ok=True)

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
            print(" No remote listed in xml")
            # here an an example: https: // ds.iris.edu / spudservice / data / 14862696
            return []
        #return remotes
    else:
        print("hmm")
        attr_name = "transfer_function.remote_references"
        remote_references = tf_obj.station_metadata.get_attr_from_name(attr_name)
        remotes = list()
        for remote_station in remote_references:
            if not len(remote_station.split('-')) > 1:
                #if remote_station != station:
                remotes.append(remote_station)
        print(remote_references)
    print(remotes)
    return remotes

def build_request_df(station_ids, network_id, start=None, end=None):
    from mth5.clients import FDSN
    fdsn_object = FDSN(mth5_version='0.2.0')
    fdsn_object.client = "IRIS"
    if start is None:
        start = '1970-01-01 00:00:00'
    if end is None:
        end = datetime.datetime.now()

    print(station_ids)
    request_list = [[network_id, station_ids.pop(0), '', '*', start, end]]

    # Handle remotes
    try:
        for station_id in station_ids:
            request_list.append([network_id, station_ids, '', '*', startdate, enddate])

    except Exception as e:
        print(e)
        print("kwahhat? ")

    print(request_list)

    request_df = pd.DataFrame(request_list, columns=fdsn_object.request_columns)
    return request_df


