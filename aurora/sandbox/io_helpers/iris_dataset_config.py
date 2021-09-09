from aurora.sandbox.io_helpers.inventory_review import scan_inventory_for_nonconformity
from aurora.sandbox.xml_sandbox import describe_inventory_stages
from aurora.sandbox.xml_sandbox import get_response_inventory_from_server


class IRISDatasetConfig(object):
    """
    This class contains the information needed to uniquely specify a
    dataset that will be accessed from IRIS.
    This config will only work for single stations.

    Need:
    -iris_metadata_parameters
    -data_parameters (how to rover, or load from local)
    -a way to specify station-channel, this config will only work for single stations.

    """

    def __init__(self):
        self.network = None
        self.station = None
        self.channels = None
        self.starttime = None
        self.endtime = None
        self.description = None
        self.dataset_id = None
        self.components_list = None  #

    def get_inventory_from_iris(
        self, base_url="IRIS", ensure_inventory_stages_are_named=True
    ):

        inventory = get_response_inventory_from_server(
            network=self.network,
            station=self.station,
            channel=self.channel_codes,
            starttime=self.starttime,
            endtime=self.endtime,
            base_url=base_url,
        )
        inventory = scan_inventory_for_nonconformity(inventory)
        if ensure_inventory_stages_are_named:
            describe_inventory_stages(inventory, assign_names=True)

        return inventory

    def get_data_via_rover(self):
        """
        Need to know where does the rover-ed file end up?
        that path needs to be accessible to load the data after it is generated.
        See example in ipython notebook in ulf_geoE repo
        Returns
        -------

        """

        pass

    def get_station_xml_filename(self, tag=""):
        """
        DEPRECATED
        """
        print("get_station_xml_filename DEPRECATED")
        raise Exception
