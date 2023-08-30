"""
Python version of Laura's bash script to scrape SPUD emtf xml

Stripping the xml tags after grepping:
https://stackoverflow.com/questions/3662142/how-to-remove-tags-from-a-string-in-python-using-regular-expressions-not-in-ht

Argparse tutorial: https://docs.python.org/3/howto/argparse.html
"""

import argparse
import pandas as pd
import pathlib
import re
import subprocess
import time

from aurora.general_helper_functions import AURORA_PATH
from aurora.test_utils.earthscope.helpers import SPUD_XML_PATHS
from aurora.test_utils.earthscope.helpers import get_summary_table_filename
from aurora.test_utils.earthscope.helpers import get_summary_table_schema
from aurora.test_utils.earthscope.helpers import get_summary_table_schema_v2
from aurora.test_utils.earthscope.helpers import get_via_curl
from aurora.test_utils.earthscope.helpers import none_or_str
from aurora.test_utils.earthscope.helpers import strip_xml_tags

force_download_data = False
force_download_emtf = False
input_spud_ids_file = AURORA_PATH.joinpath("aurora", "test_utils", "earthscope", "0_spud_ids.list")
target_dir_data = SPUD_XML_PATHS["data"]
target_dir_emtf = SPUD_XML_PATHS["emtf"]

# There are two potential sources for SPUD XML sheets
EMTF_URL = "https://ds.iris.edu/spudservice/emtf"
DATA_URL = "https://ds.iris.edu/spudservice/data"

STAGE_ID = 0
DF_SCHEMA = get_summary_table_schema_v2(STAGE_ID)

# class EMTFXML(object):
# 	def __init__(self, **kwargs):
# 		self.filepath = kwargs.get("filepath", "")

def extract_network_and_station_from_mda_info(emtf_filepath):
	# cmd = f"grep 'mda' {emtf_file} | awk -F'"'"'"' '{print $2}'"
	cmd = f"grep 'mda' {emtf_filepath}"
	try:
		qq = subprocess.check_output([cmd], shell=True)
	except subprocess.CalledProcessError as e:
		print("grep found no mda string -- assuming data are archived elsewhere")
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
	cmd = f"grep 'SourceData id' {emtf_filepath} | awk -F'"'"'"' '{print $2}'"
	qq = subprocess.check_output([cmd], shell=True)
	if qq:
		data_id = int(qq.decode().strip())

		cmd = f"grep 'SourceData id' {emtf_filepath}"
		qq = subprocess.check_output([cmd], shell=True)
		data_id2 = int(qq.decode().strip().split('"')[1])
		assert data_id2==data_id
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

def prepare_dataframe_for_scraping(restrict_to_first_n_rows=False):
	"""
	Reads in list of spud emtf_ids and initializes a dataframe
	Define columns and default values
	Args:
		restrict_to_first_n_rows:

	Returns:

	"""
	schema = get_summary_table_schema_v2(STAGE_ID)
	df = pd.read_csv(input_spud_ids_file, names=["emtf_id", ])
	schema.pop(0) #emtf_id already defined
	for col in schema:
		default = col.default
		if col.dtype == "int64":
			default = int(default)
		if col.dtype == "bool":
			default = bool(int(default))
		df[col.name] = default
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
	"""
	Downloads emtf xml and archives it, extracts info about data xml and then dowloads and archives that as well

	Parameters
	----------
	row: pandas.core.series.Series
		A row of a data frame

	Returns
	-------
	row: pandas.core.series.Series
		Same as input, but modified in-place with updated info.

	"""
	print(f"Getting {row.emtf_id}")
	spud_emtf_url = f"{EMTF_URL}/{row.emtf_id}"
	emtf_filebase = f"{row.emtf_id}.xml"
	emtf_filepath = target_dir_emtf.joinpath(emtf_filebase)

	download_emtf = to_download_or_not_to_download(emtf_filepath, force_download_emtf, emtf_or_data="EMTF")
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

	data_filebase = "_".join([str(row.emtf_id), network, station]) + ".xml"
	spud_data_url = f"{DATA_URL}/{data_id}"
	data_filepath = target_dir_data.joinpath(data_filebase)

	download_data = to_download_or_not_to_download(data_filepath, force_download_data, emtf_or_data="DATA")
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

def scrape_spud(row_start=0, row_end=None,
				force_download_data=False,
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

	if row_end is None:
		row_end = len(df)
	df = df[row_start:row_end]

	if not npartitions:
		enriched_df = df.apply(enrich_row, axis=1)
	else:
		import dask.dataframe as dd
		ddf = dd.from_pandas(df, npartitions=npartitions)
		n_rows = len(df)
		#meta = get_summary_table_schema(0)
		schema = get_summary_table_schema_v2(0)
		meta = {x.name:x.dtype for x in schema}
		enriched_df = ddf.apply(enrich_row, axis=1, meta=meta).compute()

	if save_final:
		spud_xml_csv = get_summary_table_filename(0)
		enriched_df.to_csv(spud_xml_csv, index=False)
	return enriched_df

def main():
	"""
	"""
	parser = argparse.ArgumentParser(description="Scrape XML files from SPUD")
	parser.add_argument("--nrows", help="process only the first n rows of the df", type=int, default=0)
	parser.add_argument("--npart", help="how many partitions to use (triggers dask dataframe if > 0", type=int,  default=1)
	parser.add_argument("--startrow", help="First row to process (zero-indexed)", type=int, default=0)
	# parser.add_argument('category', type=none_or_str, nargs='?', default=None,
	# 					help='the category of the stuff')
	parser.add_argument("--endrow", help="Last row to process (zero-indexed)", type=none_or_str, default=None, nargs='?',)

	args, unknown = parser.parse_known_args()
	print(f"nrows = {args.nrows}")
	print(f"npartitions = {args.npart}")
	print(f"startrow = {args.startrow}")
	print(f"endrow = {args.endrow}")
	if isinstance(args.endrow, str):
		args.endrow = int(args.endrow)
	# print(f"type(endrow) = {type(args.endrow)}")

	t0 = time.time()

	# normal usage
	scrape_spud(restrict_to_first_n_rows=args.nrows, save_final=True, npartitions=args.npart,
				row_start=args.startrow, row_end=args.endrow)

	# debugging
	#df= scrape_spud(force_download_emtf=False, restrict_to_first_n_rows=5,
   # 					save_final=False, npartitions=0)

	# re-scrape emtf
	# scrape_spud(force_download_emtf=True, save_final=False)

	delta_t = time.time() - t0
	print(f"Success {delta_t}s")

if __name__ == "__main__":
	main()
