"""
Python version of Laura's bash script to scrape SPUD emtf xml


if 127 is returned you may need to install curl in your environment
If you need a file with the IRIS mda string, i_row=6000 has one.

There are two potential sources for SPUD XML data.

Note that the EMTF spuds come as HTML, to get XML need to edit the curl command, adding
-H 'Accept: application/xml'
https://stackoverflow.com/questions/22924993/getting-webpage-data-in-xml-format-using-curl

Stripping the xml tags after grepping:
https://stackoverflow.com/questions/3662142/how-to-remove-tags-from-a-string-in-python-using-regular-expressions-not-in-ht

"""


import numpy as np
import pandas as pd
import pathlib
import re
import subprocess
import time


from aurora.test_utils.earthscope.helpers import SPUD_DATA_PATH
from aurora.test_utils.earthscope.helpers import SPUD_EMTF_PATH
from aurora.test_utils.earthscope.helpers import SPUD_XML_CSV

TMP_FROM_EMTF = False  #boolean, controls whether emtf_xml is stored as tmp, or archived locally

input_spud_ids_file = pathlib.Path('0_spud_ids.list')
# output_spud_ids_file = pathlib.Path('1_spud_ids.list')
target_dir_data = SPUD_DATA_PATH
target_dir_emtf = SPUD_EMTF_PATH


EMTF_URL = "https://ds.iris.edu/spudservice/emtf"
DATA_URL = "https://ds.iris.edu/spudservice/data"

def get_via_curl(source, target):
	cmd = f"curl -s -H 'Accept: application/xml' {source} -o {target}"
	print(cmd)
	exit_status = subprocess.call([cmd], shell=True)
	if exit_status != 0:
		print(f"Failed to {cmd}")
		raise Exception
	return

def scrape_spud(force_download_data=False,
				force_download_emtf=False,
				save_at_intervals=False,
				save_final=True):
	"""

	:param force_download:
	:return:
	"""
	# Read in list of spud emtf_ids and initialize a dataframe
	df = pd.read_csv(input_spud_ids_file, names=["emtf_id", ])
	df["data_id"] = 0
	df["file_size"] = 0
	df["fail"] = False
	df["emtf_xml_path"] = ""
	df["data_xml_path"] = ""
	n_rows = len(df)

	print(f"There are {n_rows} spud files")

	# Iterate over rows of dataframe (spud files)
	for i_row, row in df.iterrows():
		if save_at_intervals:
			if np.mod(i_row, 20) == 0:
				df.to_csv(SPUD_XML_CSV, index=False)
		# Uncomment lines below to enable fast-forward
		cutoff = 840# 6000 #2000 # 11
		if i_row < cutoff:
			continue

		print(f"Getting {i_row}/{n_rows}, {row.emtf_id}")

		# Get xml from web location, and make a local copy
		source_url = f"{EMTF_URL}/{row.emtf_id}"
		if TMP_FROM_EMTF:
			out_file_base = "tmp.xml"
		else:
			out_file_base = f"{row.emtf_id}.xml"

		emtf_file = target_dir_emtf.joinpath(out_file_base)
		if TMP_FROM_EMTF:
			download_emtf = True
		else:
			if emtf_file.exists():
				download_emtf = False
				print(f"XML emtf_file {emtf_file} already exists - skipping")
			else:
				download_emtf = True

		if force_download_emtf:
			download_emtf = True
			print("Forcing download of EMTF file")

		if download_emtf:
			try:
				get_via_curl(source_url, emtf_file)
			except:
				df.at[i_row, "fail"] = True
				continue

		df.at[i_row, "emtf_xml_path"] = str(emtf_file)

		# Extract source ID from DATA_URL, and add to df
		cmd = f"grep 'SourceData id' {emtf_file} | awk -F'"'"'"' '{print $2}'"
		# print(cmd)
		qq = subprocess.check_output([cmd], shell=True)
		data_id = int(qq.decode().strip())
		print(f"source_data_id = {data_id}")
		df.at[i_row, "data_id" ] = data_id
		#re.sub('<[^>]*>', '', mystring)
		# Extract Station Name info if IRIS provides it
		#cmd = f"grep 'mda' {emtf_file} | awk -F'"'"'"' '{print $2}'"
		cmd = f"grep 'mda' {emtf_file}"
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

		out_file_base = "_".join([str(row.emtf_id), network, station]) + ".xml"
		source_url = f"{DATA_URL}/{data_id}"
		output_xml = target_dir_data.joinpath(out_file_base)
		if output_xml.exists():
			if force_download_data:
				print("Forcing download of DATA file")
				get_via_curl(source_url, output_xml)
			else:
				print(f"XML data_file {output_xml} already exists - skipping")
				pass
		else:
			get_via_curl(source_url, output_xml)

		if output_xml.exists():
			file_size = output_xml.lstat().st_size
			df.at[i_row, "file_size"] = file_size
			df.at[i_row, "data_xml_path"] = str(output_xml)
		print("OK")
	if save_final:
		df.to_csv(SPUD_XML_CSV, index=False)

def main():
	t0 = time.time()

	# normal usage
	scrape_spud(save_at_intervals=True)

	# debugging
	#scrape_spud(force_download_emtf=False, save_final=False)

	# re-scrape emtf
	# scrape_spud(force_download_emtf=True, save_final=False)

	delta_t = time.time() - t0
	print(f"Success {delta_t}s")

if __name__ == "__main__":
	main()
