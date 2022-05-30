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

from mt_metadata.timeseries import Station

from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from obspy import read_inventory




def get_response_inventory_from_server(
    network=None,
    station=None,
    channel=None,
    starttime=None,
    endtime=None,
    level="response",
    base_url="IRIS",
):
    """

    Parameters
    ----------
    network     network = "BK"
    station
    channel     channel = "LQ2,LQ3,LT1,LT2".  If None it will get all channels
    starttime
    endtime
    station_id

    Returns
    -------

    """
    client = Client(base_url=base_url, force_redirect=True)
    inventory = client.get_stations(
        network=network,
        station=station,
        channel=channel,
        starttime=starttime,
        endtime=endtime,
        level=level,
    )
    return inventory


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
    print("ok")


def describe_inventory_stages(inventory, assign_names=False):
    """
    Scans inventory looking for stages.  Has option to assign names to stages,
    these names are used as keys in MTH5. Modifies inventory in place.

    Parameters
    ----------
    inventory
    assign_names

    Returns
    -------

    """
    new_names_were_assigned = False
    networks = inventory.networks
    for network in networks:
        for station in network:
            for channel in station:
                response = channel.response
                stages = response.response_stages
                info = (
                    f"{network.code}-{station.code}-{channel.code}"
                    f" {len(stages)}-stage response"
                )
                print(info)
                for i, stage in enumerate(stages):
                    print(f"stagename {stage.name}")
                    if stage.name is None:
                        if assign_names:
                            new_names_were_assigned = True
                            new_name = f"{station.code}_{channel.code}_{i}"
                            stage.name = new_name
                            print(f"ASSIGNING stage {stage}, name {stage.name}")
                    if hasattr(stage, "symmetry"):
                        pass
                        # import matplotlib.pyplot as plt
                        # print(f"symmetry: {stage.symmetry}")
                        # plt.figure()
                        # plt.clf()
                        # plt.plot(stage.coefficients)
                        # plt.ylabel("Filter Amplitude")
                        # plt.xlabel("Filter 'Tap'")
                        # plt.title(f"{stage.name}; symmetry: {stage.symmetry}")
                        # plt.savefig(FIGURES_BUCKET.joinpath(f
                        # "{stage.name}.png"))
                        # plt.show()
    if new_names_were_assigned:
        inventory.networks = networks
        print("Inventory Networks Reassigned")
    return


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
