from aurora.sandbox.io_helpers.inventory_review import describe_inventory_stages
from aurora.sandbox.io_helpers.inventory_review import scan_inventory_for_nonconformity
from obspy.clients.fdsn import Client


class FDSNDataset(object):
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
        self.channel_codes = None
        self.starttime = None
        self.endtime = None

        self.description = None
        self.dataset_id = None
        self.components_list = None  #
        self.data_source = "IRIS"

        self._client = None

    @property
    def client(self):
        if self._client is None:
            self.initialize_client()
        return self._client

    def initialize_client(self):
        self._client = Client(base_url=self.data_source, force_redirect=True)

    def get_inventory(self, ensure_inventory_stages_are_named=True, level="response"):

        inventory = self.client.get_stations(
            network=self.network,
            station=self.station,
            channel=self.channel_codes,
            starttime=self.starttime,
            endtime=self.endtime,
            level=level,
        )
        inventory = scan_inventory_for_nonconformity(inventory)
        if ensure_inventory_stages_are_named:
            describe_inventory_stages(inventory, assign_names=True)
        return inventory

    def get_data_via_rover(self):
        """ """
        raise NotImplementedError

    def get_data_via_fdsn_client(self):
        streams = self.client.get_waveforms(
            self.network,
            self.station,
            None,
            self.channel_codes,
            self.starttime,
            self.endtime,
        )
        return streams

    def describe(self):
        print(f"station_id = {self.station}")
        print(f"network_id = {self.network}")
        print(f"channel_ids = {self.channel_codes}")

    @property
    def h5_filebase(self):
        filebase = f"{self.dataset_id}.h5"
        return filebase
