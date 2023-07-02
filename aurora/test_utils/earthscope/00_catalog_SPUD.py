"""
Python version of Laura's bash script to scrape SPUD emtf xml



Stripping the xml tags after grepping:
https://stackoverflow.com/questions/3662142/how-to-remove-tags-from-a-string-in-python-using-regular-expressions-not-in-ht

"""


import numpy as np
import pandas as pd
import pathlib
import re
import subprocess
import time


from aurora.test_utils.earthscope.helpers import SPUD_XML_PATHS
from aurora.test_utils.earthscope.helpers import SPUD_XML_CSV


input_spud_ids_file = pathlib.Path('0_spud_ids.list')
target_dir_data = SPUD_XML_PATHS["data"]
target_dir_emtf = SPUD_XML_PATHS["emtf"]

# There are two potential sources for SPUD XML sheets
EMTF_URL = "https://ds.iris.edu/spudservice/emtf"
DATA_URL = "https://ds.iris.edu/spudservice/data"

def get_via_curl(source, target):
	"""
	If exit_status of 127 is returned you may need to install curl in your environment
	If you need a file with the IRIS mda string, i_row=6000 has one.

	Note that the EMTF spuds come as HTML, to get XML need to edit the curl command, adding
	-H 'Accept: application/xml'
	https://stackoverflow.com/questions/22924993/getting-webpage-data-in-xml-format-using-curl

	ToDo: confirm the -H option works OK for DATA_URL as well.

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
	if exit_status != 0:
		print(f"Failed to {cmd}")
		raise Exception
	return

def scrape_spud(force_download_data=False,
				force_download_emtf=False,
				restrict_to_first_n_rows=False,
				save_at_intervals=False,
				save_final=True, ):
	"""
	Notes:
	1. columns "emtf_xml_path" and "data_xml_path" should be depreacted.  A better
	solution is to store the filebase only, and use a config to control the path.


	Parameters
	----------
	force_download_data
	force_download_emtf
	restrict_to_first_n_rows: integer or None
		If an integer is provided, we will only operate of restrict_to_first_n_rows
		of the dataframe.  Used for testing only
	save_at_intervals
	save_final

	Returns
	-------

	"""
	# Read in list of spud emtf_ids and initialize a dataframe
	df = pd.read_csv(input_spud_ids_file, names=["emtf_id", ])
	df["data_id"] = 0
	df["file_size"] = 0
	df["fail"] = False
	df["emtf_xml_path"] = ""
	df["emtf_xml_filebase"] = ""
	df["data_xml_path"] = ""
	df["data_xml_filebase"] = ""

	n_rows = len(df)
	info_str = f"There are {n_rows} spud files"
	print(f"There are {n_rows} spud files")
	if restrict_to_first_n_rows:
		df = df.iloc[:restrict_to_first_n_rows]
		info_str += f"\n restricting to first {restrict_to_first_n_rows} rows for testing"
		n_rows = len(df)
	print(info_str)

	# Iterate over rows of dataframe (spud files)
	for i_row, row in df.iterrows():
		if save_at_intervals:
			if np.mod(i_row, 20) == 0:
				df.to_csv(SPUD_XML_CSV, index=False)
		# Uncomment lines below to enable fast-forward
		# cutoff = 840# 6000 #2000 # 11
		# if i_row < cutoff:
		#	continue

		print(f"Getting {i_row}/{n_rows}, {row.emtf_id}")

		# Get xml from web location, and make a local copy
		source_url = f"{EMTF_URL}/{row.emtf_id}"
		emtf_filebase = f"{row.emtf_id}.xml"

		emtf_filepath = target_dir_emtf.joinpath(emtf_filebase)
		if emtf_filepath.exists():
			download_emtf = False
			print(f"XML emtf_file {emtf_filepath} already exists - skipping")
		else:
			download_emtf = True

		if force_download_emtf:
			download_emtf = True
			print("Forcing download of EMTF file")

		if download_emtf:
			try:
				get_via_curl(source_url, emtf_filepath)
			except:
				df.at[i_row, "fail"] = True
				continue

		df.at[i_row, "emtf_xml_path"] = str(emtf_filepath)
		df.at[i_row, "emtf_xml_filebase"] = emtf_filebase

		# Extract source ID from DATA_URL, and add to df
		cmd = f"grep 'SourceData id' {emtf_filepath} | awk -F'"'"'"' '{print $2}'"

		qq = subprocess.check_output([cmd], shell=True)
		data_id = int(qq.decode().strip())
		print(f"source_data_id = {data_id}")
		df.at[i_row, "data_id" ] = data_id
		#re.sub('<[^>]*>', '', mystring)
		# Extract Station Name info if IRIS provides it
		#cmd = f"grep 'mda' {emtf_file} | awk -F'"'"'"' '{print $2}'"
		cmd = f"grep 'mda' {emtf_filepath}"
		try:
			qq = subprocess.check_output([cmd], shell=True)
		except subprocess.CalledProcessError as e:
			print("NO GREPP")
			qq = None
		network = ""
		station = ""
		if qq:
			xml_url = qq.decode().strip()
			url = re.sub('<[^>]*>', '', xml_url)
			url_parts = url.split("/")
			if "mda" in url_parts:
				idx = url_parts.index("mda")
				network = url_parts[idx + 1]
				station = url_parts[idx + 2]

		data_filebase = "_".join([str(row.emtf_id), network, station]) + ".xml"
		source_url = f"{DATA_URL}/{data_id}"
		data_filepath = target_dir_data.joinpath(data_filebase)
		if data_filepath.exists():
			if force_download_data:
				print("Forcing download of DATA file")
				get_via_curl(source_url, data_filepath)
			else:
				print(f"XML data_file {data_filepath} already exists - skipping")
				pass
		else:
			get_via_curl(source_url, data_filepath)

		if data_filepath.exists():
			file_size = data_filepath.lstat().st_size
			df.at[i_row, "file_size"] = file_size
			df.at[i_row, "data_xml_path"] = str(data_filepath)
			df.at[i_row, "data_xml_filebase"] = data_filebase
		print("OK")
	if save_final:
		df.to_csv(SPUD_XML_CSV, index=False)
	return df

def main():
	t0 = time.time()

	# normal usage
	scrape_spud(save_at_intervals=True)

	# debugging
	#df= scrape_spud(force_download_emtf=False, restrict_to_first_n_rows=11,
    #					save_final=False)

	# re-scrape emtf
	# scrape_spud(force_download_emtf=True, save_final=False)

	delta_t = time.time() - t0
	print(f"Success {delta_t}s")

if __name__ == "__main__":
	main()
