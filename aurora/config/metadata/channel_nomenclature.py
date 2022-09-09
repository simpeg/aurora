# -*- coding: utf-8 -*-
"""

@author: kkappler
"""
# =============================================================================
# Imports
# =============================================================================
from mt_metadata.base.helpers import write_lines
from mt_metadata.base import get_schema, Base
from .standards import SCHEMA_FN_PATHS

# =============================================================================
attr_dict = get_schema("channel_nomenclature", SCHEMA_FN_PATHS)

CHANNEL_MAPS = {}
CHANNEL_MAPS["default"] = {
    "hx": "hx",
    "hy": "hy",
    "hz": "hz",
    "ex": "ex",
    "ey": "ey",
}

LEMI_CHANNEL_MAP_12 = {
    "hx": "bx",
    "hy": "by",
    "hz": "bz",
    "ex": "e1",
    "ey": "e2",
}

LEMI_CHANNEL_MAP_34 = {
    "hx": "bx",
    "hy": "by",
    "hz": "bz",
    "ex": "e3",
    "ey": "e4",
}

PHOENIX_CHANNEL_MAP_123 = {
    "hx": "h1",
    "hy": "h2",
    "hz": "h3",
    "ex": "e1",
    "ey": "e2",
}


# =============================================================================
class ChannelNomenclature(Base):
    __doc__ = write_lines(attr_dict)

    def __init__(self, **kwargs):

        super().__init__(attr_dict=attr_dict, **kwargs)
        self._keyword = None

    def update_by_keyword(self, keyword):
        """
        Assign the HEXY values "ex", "ey" etc based on a pre-defined dict that
        corresponds to a common use case
        Parameters
        ----------
        keyword

        Returns
        -------

        """
        raise NotImplementedError

    @property
    def ex_ey(self):
        return [self.ex, self.ey]

    @property
    def hx_hy(self):
        return [self.hx, self.hy]

    @property
    def ex_ey_hz(self):
        return [self.ex, self.ey, self.hz]

    @property
    def default_input_channels(self):
        return self.hx_hy

    @property
    def default_output_channels(self):
        return self.ex_ey_hz

    @property
    def default_reference_channels(self):
        return self.hx_hy

    @property
    def keyword(self):
        return self._keyword

    @keyword.setter
    def keyword(self, keyword):
        self._keyword = keyword
        self._update_by_keyword(keyword)

    def get_channel_map(self, keyword):
        if keyword == "default":
            channel_map = CHANNEL_MAPS["default"]
        elif keyword == "LEMI12":
            channel_map = LEMI_CHANNEL_MAP_12
        elif keyword == "LEMI34":
            channel_map = LEMI_CHANNEL_MAP_34
        elif keyword == "NIMS":
            channel_map = CHANNEL_MAPS["default"]
        elif keyword == "PHOENIX123":
            channel_map = PHOENIX_CHANNEL_MAP_123
        else:
            print(f"whoops mt_system {keyword} unknown")
            raise NotImplementedError
        return channel_map

    def _update_by_keyword(self, keyword):
        channel_map = self.get_channel_map(keyword)
        self.ex = channel_map["ex"]
        self.ey = channel_map["ey"]
        self.hx = channel_map["hx"]
        self.hy = channel_map["hy"]
        self.hz = channel_map["hz"]

    def unpack(self):
        return self.ex, self.ey, self.hx, self.hy, self.hz

    @property
    def channels(self):
        channels = list(self.get_channel_map(self.keyword).values())
        return channels
