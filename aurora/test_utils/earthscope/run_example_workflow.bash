#!/bin/bash

# Example usage
#  for station in `cat ../8P_stations.list`; do ./run_example_workflow.bash 8P $station 2>&1 | tee -a test.log; done

network=$1
station=$2

echo $network, $station
python3 example_workflow.py --network=$network --station=$station
echo
echo
echo

