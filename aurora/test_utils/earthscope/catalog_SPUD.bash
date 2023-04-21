#!/bin/bash
# SPUD scraper - grabs EMTF xmls from SPUD
# to be loaded into mth5

# Laura Keyson, EarthScope

spud_ids='spud_ids.list'
dir='spud_xml'


for id in `cat $spud_ids`; do
	echo $id
	tmpFile="${dir}/tmp.xml"
	curl -s https://ds.iris.edu/spudservice/emtf/${id} -o ${tmpFile}
	
	xml_id=`grep "SourceData id" ${tmpFile} | awk -F'"' '{print $2}'`
	station_name=`grep 'mda' ${tmpFile} | awk -F'"' '{print $2}' | awk -F'/' 'BEGIN{OFS="_";} {print $5,$6}'`
	echo "  $station_name"
	
	if [[ ${#station_name} -eq 0 ]]; then
		file_name=$id
	else
		file_name=${station_name}_${id}
	fi

	xml_url="http://ds.iris.edu/spudservice/data/${xml_id}"

	curl -s $xml_url -o ${dir}/${file_name}.xml


done


