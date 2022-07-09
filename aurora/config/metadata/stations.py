# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 13:58:07 2022

@author: jpeacock
"""
# =============================================================================
# Imports
# =============================================================================
from mt_metadata.base import Base, get_schema
from .standards import SCHEMA_FN_PATHS
from .station import Station

# =============================================================================
attr_dict = get_schema("stations", SCHEMA_FN_PATHS)
attr_dict.add_dict(get_schema("station", SCHEMA_FN_PATHS), "local")

# =============================================================================
class Stations(Base):
    """
    class to hold station information

    station to process
    remote references to use

    """

    def __init__(self, **kwargs):
        self.local = Station()
        self._remote = []

        super().__init__(attr_dict=attr_dict, **kwargs)

    @property
    def remote(self):
        return_list = []
        for rr in self._remote:
            if isinstance(rr, dict):
                b = Station()
                b.from_dict(rr)
                b.remote = True
            elif isinstance(rr, Station):
                b = rr
                b.remote = True
            return_list.append(rr)
        return return_list

    @remote.setter
    def remote(self, rr_station):
        self._remote = []
        if isinstance(rr_station, list):
            for item in rr_station:
                if not isinstance(item, Station):
                    raise TypeError(
                        f"list item must be Station object not {type(item)}"
                    )
                self._remote.append(item)

        elif isinstance(rr_station, dict):
            remote = Station()
            remote.from_dict(rr_station)
            remote.remote = True
            self._remote.append(remote)

        elif isinstance(rr_station, Station):
            rr_station.remote = True
            self._remote.append(rr_station)
        else:
            raise ValueError(f"not sure to do with {type(rr_station)}")

    def add_remote(self, rr):
        """
        add a band
        """

        if not isinstance(rr, (Station, dict)):
            raise TypeError(f"List entry must be a Station object not {type(rr)}")
        if isinstance(rr, dict):
            obj = Station()
            obj.from_dict(rr)

        else:
            obj = rr

        obj.remote = True

        self._remote.append(obj)

    @property
    def remote_dict(self):
        """
        need to have a dictionary, but it can't be an attribute cause that
        gets confusing when reading in a json file

        :return: DESCRIPTION
        :rtype: TYPE

        """
        return dict([(rr.id, rr) for rr in self.remote])

    def from_dataset_dataframe(self, df):
        """
        from a dataset dataframe

        :return: DESCRIPTION
        :rtype: TYPE

        """

        station = df[df.remote == False].station_id.unique()[0]
        rr_stations = df[df.remote == True].station_id.unique()

        self.local.from_dataset_dataframe(df[df.station_id == station])

        for rr_station in rr_stations:
            rr = Station()
            rr.from_dataset_dataframe(df[df.station_id == rr_station])
            self.add_remote(rr)

    def to_dataset_dataframe(self):
        """
        output a dataframe

        :return: DESCRIPTION
        :rtype: TYPE

        """

        local_df = self.local.to_dataset_dataframe()

        for rr in self.remote:
            local_df = local_df.append(rr.to_dataset_dataframe())

        local_df = local_df.reset_index()

        return local_df

    def get_station(self, station_id):
        """
        get a station object from the id

        :param station_id: DESCRIPTION
        :type station_id: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        if self.local.id == station_id:
            return self.local

        elif station_id in self.remote_dict.keys():
            return self.remote_dict[station_id]

        raise KeyError(f"could not find {station_id}")
