import pathlib

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


def get_remotes_from_tf(tf_obj):
    remote_references = tf_obj.station_metadata.get_attr_from_name('transfer_function.remote_references')
    remotes = list()
    for remote_station in remote_references:
        if not len(remote_station.split('-')) > 1:
            if remote_station != station:
                remotes.append(remote_station)

    return remotes

