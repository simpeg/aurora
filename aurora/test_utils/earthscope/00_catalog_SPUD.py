"""
Python version of Laura's bash script to scrape SPUD emtf xml

Stripping the xml tags after grepping:
https://stackoverflow.com/questions/3662142/how-to-remove-tags-from-a-string-in-python-using-regular-expressions-not-in-ht

"""

import argparse
import numpy as np
import pandas as pd
import pathlib
import re
import subprocess
import time

from aurora.general_helper_functions import AURORA_PATH
from aurora.test_utils.earthscope.helpers import SPUD_XML_PATHS
from aurora.test_utils.earthscope.helpers import get_summary_table_filename
from aurora.test_utils.earthscope.helpers import get_summary_table_schema
from aurora.test_utils.earthscope.helpers import get_via_curl
from aurora.test_utils.earthscope.helpers import strip_xml_tags

force_download_data = False
force_download_emtf = False
input_spud_ids_file = AURORA_PATH.joinpath("aurora", "test_utils", "earthscope", "0_spud_ids.list")
target_dir_data = SPUD_XML_PATHS["data"]
target_dir_emtf = SPUD_XML_PATHS["emtf"]

# There are two potential sources for SPUD XML sheets
EMTF_URL = "https://ds.iris.edu/spudservice/emtf"
DATA_URL = "https://ds.iris.edu/spudservice/data"

# class EMTFXML(object):
# 	def __init__(self, **kwargs):
# 		self.filepath = kwargs.get("filepath", "")

def extract_network_and_station_from_mda_info(emtf_filepath):
	# cmd = f"grep 'mda' {emtf_file} | awk -F'"'"'"' '{print $2}'"
	cmd = f"grep 'mda' {emtf_filepath}"
	try:
		qq = subprocess.check_output([cmd], shell=True)
	except subprocess.CalledProcessError as e:
		print("GREP found no mda string-- assuming data are archived elsewhere")
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

def prepare_dataframe_for_scraping(restrict_to_first_n_rows=False):
	"""
	Define columns and default values
	Args:
		restrict_to_first_n_rows:

	Returns:

	"""
	# Read in list of spud emtf_ids and initialize a dataframe
	df = pd.read_csv(input_spud_ids_file, names=["emtf_id", ])
	df["data_id"] = 0
	df["fail"] = False
	df["emtf_file_size"] = 0
	df["emtf_xml_filebase"] = ""
	df["data_file_size"] = 0
	df["data_xml_filebase"] = ""
	n_rows = len(df)
	info_str = f"There are {n_rows} spud files"
	print(f"There are {n_rows} spud files")
	if restrict_to_first_n_rows:
		df = df.iloc[:restrict_to_first_n_rows]
		info_str += f"\n restricting to first {restrict_to_first_n_rows} rows for testing"
		n_rows = len(df)
	print(info_str)
	return df

def enrich_row(row):
	#print("row",row)
	print(f"Getting {row.emtf_id}")
	source_url = f"{EMTF_URL}/{row.emtf_id}"
	emtf_filebase = f"{row.emtf_id}.xml"
	emtf_filepath = target_dir_emtf.joinpath(emtf_filebase)

	# To download on not to download - that is the question:
	if emtf_filepath.exists():
		download_emtf = False
		print(f"XML emtf_file {emtf_filepath} already exists")
		if force_download_emtf:
			download_emtf = True
			print("Forcing download of EMTF file")
	else:
		download_emtf = True

	if download_emtf:
		try:
			get_via_curl(source_url, emtf_filepath)
		except:
			row["fail"] = True

	file_size = emtf_filepath.lstat().st_size
	row["emtf_file_size"] = file_size
	row["emtf_xml_filebase"] = emtf_filebase

	# Extract source ID from DATA_URL, and add to df
	print(emtf_filepath)
	cmd = f"grep 'SourceData id' {emtf_filepath} | awk -F'"'"'"' '{print $2}'"
	qq = subprocess.check_output([cmd], shell=True)
	data_id = int(qq.decode().strip())

	cmd = f"grep 'SourceData id' {emtf_filepath}"
	qq = subprocess.check_output([cmd], shell=True)
	data_id2 = int(qq.decode().strip().split('"')[1])

	assert data_id2==data_id

	print(f"source_data_id = {data_id}")
	row["data_id"] = data_id

	# Extract Station Name info if IRIS provides it
	network, station = extract_network_and_station_from_mda_info(emtf_filepath)

	data_filebase = "_".join([str(row.emtf_id), network, station]) + ".xml"
	source_url = f"{DATA_URL}/{data_id}"
	data_filepath = target_dir_data.joinpath(data_filebase)
	if data_filepath.exists():
		if force_download_data:
			print("Forcing download of DATA file")
			get_via_curl(source_url, data_filepath)
		else:
			print(f"XML data_file {data_filepath} already exists - skipping")
	else:
		get_via_curl(source_url, data_filepath)

	if data_filepath.exists():
		file_size = data_filepath.lstat().st_size
		row.at["data_file_size"] = file_size
		row.at["data_xml_filebase"] = data_filebase
	return row

def scrape_spud(force_download_data=False,
				force_download_emtf=False,
				restrict_to_first_n_rows=False,
				save_final=True,
				npartitions=0):
	"""
	Notes:
		Gets xml from web location, and makes a local copy

	Parameters
	----------
	force_download_data
	force_download_emtf
	restrict_to_first_n_rows: integer or None
		If an integer is provided, we will only operate of restrict_to_first_n_rows
		of the dataframe.  Used for testing only
	save_final

	Returns
	-------

	"""
	df = prepare_dataframe_for_scraping(restrict_to_first_n_rows=restrict_to_first_n_rows)
	if not npartitions:
		enriched_df = df.apply(enrich_row, axis=1)
	else:
		import dask.dataframe as dd
		ddf = dd.from_pandas(df, npartitions=npartitions)
		n_rows = len(df)
		df_schema = get_summary_table_schema(0)
		enriched_df = ddf.apply(enrich_row, axis=1, meta=df_schema).compute()

	if save_final:
		spud_xml_csv = get_summary_table_filename(0)
		enriched_df.to_csv(spud_xml_csv, index=False)
	return enriched_df

def main():
	"""
	Follows this great argparse tutorial: https://docs.python.org/3/howto/argparse.html
	:return:
	"""
	parser = argparse.ArgumentParser(description="Scrape XML files from SPUD")
	parser.add_argument("--nrows", help="process only the first n rows of the df", type=int, default=0)
	parser.add_argument("--npart", help="how many partitions to use (triggers dask dataframe if > 0", type=int,  default=0)
	args = parser.parse_args()
	print(f"nrows = {args.nrows}")
	print(f"npartitions = {args.npart}")

	t0 = time.time()

	# normal usage
	scrape_spud(restrict_to_first_n_rows=args.nrows, save_final=True, npartitions=args.npart)

	# debugging
	#df= scrape_spud(force_download_emtf=False, restrict_to_first_n_rows=5,
   # 					save_final=False, npartitions=0)

	# re-scrape emtf
	# scrape_spud(force_download_emtf=True, save_final=False)

	delta_t = time.time() - t0
	print(f"Success {delta_t}s")

if __name__ == "__main__":
	main()
