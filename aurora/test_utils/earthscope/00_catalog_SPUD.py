"""
Python version of Laura's bash script to scrape SPUD emtf xml

Stripping the xml tags after grepping:
https://stackoverflow.com/questions/3662142/how-to-remove-tags-from-a-string-in-python-using-regular-expressions-not-in-ht

"""


import pandas as pd
import subprocess
import time

from aurora.general_helper_functions import AURORA_PATH
from aurora.test_utils.earthscope.helpers import SPUD_XML_PATHS
from aurora.test_utils.earthscope.helpers import get_via_curl
from aurora.test_utils.earthscope.helpers import strip_xml_tags
from aurora.test_utils.earthscope.widescale import WidesScaleTest


input_spud_ids_file = AURORA_PATH.joinpath(
    "aurora", "test_utils", "earthscope", "0_spud_ids.list"
)
target_dir_data = SPUD_XML_PATHS["data"]
target_dir_emtf = SPUD_XML_PATHS["emtf"]

# There are two potential sources for SPUD XML sheets
EMTF_URL = "https://ds.iris.edu/spudservice/emtf"
DATA_URL = "https://ds.iris.edu/spudservice/data"

STAGE_ID = 0


# HELPER FUNCTIONS
def extract_network_and_station_from_mda_info(emtf_filepath):
    """based on part of a bash script recievd from Laura, could use a desciption of the expected mda line format"""
    # cmd = f"grep 'mda' {emtf_file} | awk -F'"'"'"' '{print $2}'"
    cmd = f"grep 'mda' {emtf_filepath}"
    try:
        qq = subprocess.check_output([cmd], shell=True)
    except subprocess.CalledProcessError as e:
        print(f"{e} grep found no mda string -- assuming data are archived elsewhere")
        qq = None
    network = ""
    station = ""
    if qq:
        xml_url = qq.decode().strip()
        url = strip_xml_tags(xml_url)
        url_parts = url.split("/")
        if "mda" in url_parts:
            idx = url_parts.index("mda")
            network = url_parts[idx + 1]
            station = url_parts[idx + 2]
    return network, station


def extract_data_id_from_emtf(emtf_filepath):
    """
    modified to check if grep returns empty
    Parameters
    ----------
    emtf_filepath: str or pathlib.Path
            Location of a TF XML file from SPUD EMTF archive

    Returns
    -------
    data_id: str
            String (that looks like an int) that tells us how ot access the data XML
    """
    cmd = f"grep 'SourceData id' {emtf_filepath} | awk -F'" '"' "' '{print $2}'"
    qq = subprocess.check_output([cmd], shell=True)
    if qq:
        data_id = int(qq.decode().strip())

        cmd = f"grep 'SourceData id' {emtf_filepath}"
        qq = subprocess.check_output([cmd], shell=True)
        data_id2 = int(qq.decode().strip().split('"')[1])
        assert data_id2 == data_id
        return data_id
    else:
        return False


def to_download_or_not_to_download(filepath, force_download, emtf_or_data=""):
    if filepath.exists():
        download = False
        print(f"XML {emtf_or_data} file {filepath} already exists")
        if force_download:
            download = True
            print(f"Forcing download of {emtf_or_data} file")
    else:
        download = True
    return download


# MAIN CLASS
class TestScrapeSPUD(WidesScaleTest):
    def __init__(self, **kwargs):
        """
        some notews
        """
        super().__init__(**kwargs)
        self.force_download_data = kwargs.get("force_download_data", False)
        self.force_download_emtf = kwargs.get("force_download_emtf", False)
        self.somthing = kwargs.get("somthing", -11)

    def prepare_jobs_dataframe(self):
        """
        Define the data structure that is output from this stage of processing
        Returns
        -------

        """
        schema = self.df_schema
        df = pd.read_csv(
            input_spud_ids_file,
            names=[
                "emtf_id",
            ],
        )
        schema.pop(0)  # emtf_id already defined
        for col in schema:
            default = col.default
            if col.dtype == "int64":
                default = int(default)
            if col.dtype == "bool":
                default = bool(int(default))
            df[col.name] = default
            if col.dtype == "string":
                df[col.name] = ""
        return df

    def enrich_row(self, row):
        """
        This will eventually get used by dask, but as a step we need to make this a method
        that works with df.apply()
        Returns:

        """
        print(f"Getting {row.emtf_id}")
        spud_emtf_url = f"{EMTF_URL}/{row.emtf_id}"
        emtf_filebase = f"{row.emtf_id}.xml"
        emtf_filepath = target_dir_emtf.joinpath(emtf_filebase)

        download_emtf = to_download_or_not_to_download(
            emtf_filepath, self.force_download_emtf, emtf_or_data="EMTF"
        )
        if download_emtf:
            try:
                get_via_curl(spud_emtf_url, emtf_filepath)
            except:
                row["fail"] = True
                return row

        file_size = emtf_filepath.lstat().st_size
        row["emtf_file_size"] = file_size
        row["emtf_xml_filebase"] = emtf_filebase

        # Extract source ID from DATA_URL, and add to df
        data_id = extract_data_id_from_emtf(emtf_filepath)
        row["data_id"] = data_id

        # Extract Station Name info if IRIS provides it
        network, station = extract_network_and_station_from_mda_info(emtf_filepath)

        spud_data_url = f"{DATA_URL}/{data_id}"
        data_filebase = "_".join([str(row.emtf_id), network, station]) + ".xml"
        data_filepath = target_dir_data.joinpath(data_filebase)

        download_data = to_download_or_not_to_download(
            data_filepath, self.force_download_data, emtf_or_data="DATA"
        )
        if download_data:
            try:
                get_via_curl(spud_data_url, data_filepath)
            except:
                row["fail"] = True
                return row

        if data_filepath.exists():
            file_size = data_filepath.lstat().st_size
            row.at["data_file_size"] = file_size
            row.at["data_xml_filebase"] = data_filebase
        return row


def main():
    """ """
    tester = TestScrapeSPUD(stage_id=0)

    # expect to find data_id, and Earthscope mda
    # tester.startrow = 25
    # tester.endrow = 26

    # tester.endrow = 5
    tester.run_test()

    # row_start = 0; row_end = 1 # expect to find data_id, but not an Earthscope mda
    # df= scrape_spud(force_download_emtf=False,  row_start=25, row_end=26,
    # 				save_final=False, npartitions=0)

    # re-scrape emtf
    # scrape_spud(force_download_emtf=True, save_final=False)

    print("Success")


if __name__ == "__main__":
    main()
