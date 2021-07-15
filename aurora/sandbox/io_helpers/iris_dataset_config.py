from aurora.sandbox.io_helpers.inventory_review import scan_inventory_for_nonconformity
from aurora.sandbox.xml_sandbox import describe_inventory_stages
from aurora.sandbox.xml_sandbox import get_response_inventory_from_iris
from mth5_test_data.util import MTH5_TEST_DATA_DIR as DATA_DIR

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
        self.components_list = None #


    def get_inventory_from_iris(self, ensure_inventory_stages_are_named=True):

        inventory = get_response_inventory_from_iris(network=self.network,
                                                     station=self.station,
                                                     channel=self.channel_codes,
                                                     starttime=self.starttime,
                                                     endtime=self.endtime,
                                                     )
        inventory = scan_inventory_for_nonconformity(inventory)
        if ensure_inventory_stages_are_named:
            describe_inventory_stages(inventory, assign_names=True)
            # describe_inventory_stages(inventory, assign_names=False)

        return inventory


    def get_data_via_rover(self):
        """
        Need
        1. Where does the rover-ed file end up?  that path needs to be accessible to load the data
        after it is generated
        Returns
        -------

        """
        pass

    def get_station_xml_filename(self, tag=""):
        """
        Placeholder in case we need to make many of these
        TODO: Modify so the path comes from the dataset_id, not the station_id...

        """
        filebase = f"{self.dataset_id}.xml"
        if tag:
            filebase = f"{tag}_{filebase}"
        target_folder = DATA_DIR.joinpath("iris",f"{self.network}")
        target_folder.mkdir(exist_ok=True)
        xml_filepath = target_folder.joinpath(filebase)
        return xml_filepath

    def save_xml(self, experiment, tag=""):
        """
        could probably use inventory or experiement
        Maybe even add a type-checker here
        if isinstance(xml_obj, inventory):
            tag="inventory"
        elif  isinstance(xml_obj, Experiment()):
            tag="experiment"

        Parameters
        ----------
        experiement

        Returns
        -------

        """
        output_xml_path = self.get_station_xml_filename(tag=tag)
        experiment.to_xml(output_xml_path)
        print(f"saved experiement to {output_xml_path}")
        return
