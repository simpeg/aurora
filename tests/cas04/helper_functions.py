from obspy import read_inventory

from mt_metadata.timeseries.stationxml import XMLInventoryMTExperiment
from mth5.utils.helpers import initialize_mth5


def xml_to_mth5(xml_path, h5_path="tmp.h5"):
    """
    Parameters
    ----------
    xml_path

    Returns
    -------

    """
    inventory0 = read_inventory(str(xml_path))  # 8P
    translator = XMLInventoryMTExperiment()
    experiment = translator.xml_to_mt(inventory_object=inventory0)
    mth5_obj = initialize_mth5(h5_path)
    mth5_obj.from_experiment(experiment)
    return mth5_obj
