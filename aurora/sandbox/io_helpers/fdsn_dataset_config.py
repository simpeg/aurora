from obspy.clients import fdsn

from aurora.sandbox.io_helpers.inventory_review import scan_inventory_for_nonconformity
from aurora.sandbox.xml_sandbox import describe_inventory_stages
from aurora.sandbox.xml_sandbox import get_response_inventory_from_server


class FDSNDatasetConfig(object):
    """
    This class contains the information needed to uniquely specify a
    dataset that will be accessed from IRIS, NCEDC, or other FDSN client.
    This config will only work for single stations.

    Need:
    -fdsn_metadata_parameters
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

    # @classmethod
    # def from_df_row(cls, row):
    #     qq = cls.__init__()
    #     qq.station = row.station
    #     qq.startime = row.startime
    #     etc...

    def get_inventory_from_client(
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

    def get_data_via_fdsn_client(self, data_source="IRIS"):
        client = fdsn.Client(data_source)

        streams = client.get_waveforms(
            self.network,
            self.station,
            None,
            self.channel_codes,
            self.starttime,
            self.endtime,
        )
        return streams

    def get_station_xml_filename(self, tag=""):
        """
        DEPRECATED
        """
        print("get_station_xml_filename DEPRECATED")
        raise Exception

    def describe(self):
        print(f"station_id = {self.station}")  # station_id in mth5_obj.station_list
        print(f"network_id = {self.network}")
        print(f"channel_ids = {self.channel_codes}")

    @property
    def h5_filebase(self):
        filebase = f"{self.dataset_id}.h5"
        return filebase
