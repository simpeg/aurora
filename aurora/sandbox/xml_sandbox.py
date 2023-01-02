# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 15:24:33 2021

@author: kkappler
This is a xml reader prototype for the filter.xml

Filter application info: they always have either "value" or "poles_zeros"


#import xml.etree.cElementTree as ET
#tree = ET.parse(xml_path)
# mt_root_element = tree.getroot()
# mt_experiment = Experiment()
# mt_experiment.from_xml(mt_root_element)
"""


import datetime


from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from obspy import read_inventory


def test_get_example_em_xml_from_iris_via_web():
    print("test_get_example_em_xml_from_iris_via_web")
    client = Client(base_url="IRIS", force_redirect=True)
    starttime = UTCDateTime("2015-01-09")
    endtime = UTCDateTime("2015-01-20")
    inventory = client.get_stations(
        network="XX", station="EMXXX", starttime=starttime, endtime=endtime
    )
    network = inventory[0]  # obspy.core.inventory.network.Network
    print(f"network {network}")


def test_get_example_xml_inventory():
    print("test_get_example_xml_inventory")
    test_file_name = "fdsn-station_2021-03-09T04_44_51.xml"
    inventory = read_inventory(test_file_name)
    iterate_through_mtml(inventory)


def iterate_through_mtml(networks):
    """
    Starting from pseudocode recommended by Tim
    20210203: So far all obspy XML encountered have had only a single network.

    Returns
    -------
    type networks: obspy.core.inventory.inventory.Inventory
    """
    for network in networks:
        for station in network:
            for channel in station:
                response = channel.response
                stages = response.response_stages
                info = "{}-{}-{} {}-stage response".format(
                    network.code, station.code, channel.code, len(stages)
                )
                print(info)

                for stage in stages:
                    # pass
                    print("stage {}".format(stage))


def main():
    """ """
    test_get_example_xml_inventory()
    test_get_example_em_xml_from_iris_via_web()
    print("finito {}".format(datetime.datetime.now()))


if __name__ == "__main__":
    main()
