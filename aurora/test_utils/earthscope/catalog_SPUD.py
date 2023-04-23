"""
Python version of Laura's bash script to scrape SPUD emtf xml


if 127 is returned you may need to install curl in your environment
If you need a file with the IRIS mda string, i_row=6000 has one.


"""
import pathlib

import numpy as np
import pandas as pd
import subprocess
import time

TMP_FROM_EMTF = False  #boolean, controls whether emtf_xml is stored as tmp, or archived locally

# def grep_text(fname, ):
# 	'"'"'"'
data_dir = pathlib.Path(".")
input_spud_ids_file = pathlib.Path('0_spud_ids.list')
output_spud_ids_file = pathlib.Path('1_spud_ids.list')
target_dir_emtf = data_dir.joinpath('spud_xml').joinpath("emtf")
target_dir_data = data_dir.joinpath('spud_xml').joinpath("data")
target_dir_emtf.mkdir(exist_ok=True, parents=True)
target_dir_data.mkdir(exist_ok=True, parents=True)

EMTF_URL = "https://ds.iris.edu/spudservice/emtf"
DATA_URL = "https://ds.iris.edu/spudservice/data"

def get_via_curl(source, target):
	cmd = f"curl -s {source} -o {target}"
	print(cmd)
	exit_status = subprocess.call([cmd], shell=True)
	if exit_status != 0:
		print(f"Failed to {cmd}")
		raise Exception
	return

def scrape_spud(force_download=False):
	df = pd.read_csv(input_spud_ids_file, names=["emtf_id", ])
	df["data_id"] = 0
	df["file_size"] = 0
	df["fail"] = False
	df["emtf_xml_path"] = ""
	df["data_xml_path"] = ""
	n_rows = len(df)

	print(f"There are {n_rows} spud files")

	for i_row, row in df.iterrows():
		if np.mod(i_row, 20) == 0:
			df.to_csv("spud_summary.csv", index=False)
		# Uncomment lines below to enable fast-forward
		# cutoff = 11# 6000 #2000 # 11
		# if i_row < cutoff:
		# 	continue

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

		# Extract Station Name info if IRIS provides it
		cmd = f"grep 'mda' {emtf_file} | awk -F'"'"'"' '{print $2}'"
		qq = subprocess.check_output([cmd], shell=True)
		network = ""
		station = ""
		if qq:
			url_parts = qq.decode().strip().split("/")
			if url_parts[-3] == "mda":
				network = url_parts[-2]
				station = url_parts[-1]

		out_file_base = "_".join([str(row.emtf_id), network, station]) + ".xml"
		source_url = f"{DATA_URL}/{data_id}"
		output_xml = target_dir_data.joinpath(out_file_base)
		if output_xml.exists():
			if force_download:
				get_via_curl(source_url, output_xml)
			else:
				print(f"XML data_file {output_xml} already exists - skipping")
				pass
		else:
			get_via_curl(source_url, output_xml)

		if output_xml.exists():
			file_size = output_xml.lstat().st_size
			df.at[i_row, "file_size"] = file_size
			df.at[i_row, "data_xml_path"] = str(emtf_file)
		print("OK")
	df.to_csv("spud_summary.csv", index=False)

def main():
	t0 = time.time()
	scrape_spud()
	delta_t = time.time() - t0
	print(f"Success {delta_t}s")

if __name__ == "__main__":
	main()
