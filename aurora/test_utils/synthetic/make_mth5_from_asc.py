# -*- coding: utf-8 -*-
"""

Notes: This module has been migrated to MTH5.

"""

from mth5.data.make_mth5_from_asc import create_test1_h5
from mth5.data.make_mth5_from_asc import create_test2_h5
from mth5.data.make_mth5_from_asc import create_test1_h5_with_nan
from mth5.data.make_mth5_from_asc import create_test12rr_h5
from mth5.data.make_mth5_from_asc import create_test3_h5
from mth5.data.make_mth5_from_asc import create_test4_h5


def main(file_version="0.1.0"):
    """Allow the module to be called from the command line"""
    file_version = "0.2.0"
    create_test1_h5(file_version=file_version)
    create_test1_h5_with_nan(file_version=file_version)
    create_test2_h5(file_version=file_version)
    create_test12rr_h5(file_version=file_version, channel_nomenclature="lemi12")
    create_test3_h5(file_version=file_version)
    create_test4_h5(file_version=file_version)


if __name__ == "__main__":
    main()
