"""
    Helper functions to handle workarounds.
"""

import numpy as np
from mt_metadata.transfer_functions import TF


def tfs_nearly_equal(tf1: TF, tf2: TF) -> bool:
    """
    Addresses issue # 381.
    Werid bug on MacOS involving numeric noise.

    Also forces the creation time of tf2 to match tf1.

    """
    cond1 = np.isclose(
        tf2.residual_covariance, tf1.residual_covariance, atol=1e-26, rtol=1e-30
    ).all()
    cond2 = np.isclose(
        tf2.transfer_function_error, tf1.transfer_function_error, atol=1e-14, rtol=1e-15
    ).all()
    cond3 = np.isclose(
        tf2.transfer_function, tf1.transfer_function, atol=1e-14, rtol=1e-15
    ).all()

    if cond1 and cond2 and cond3:
        tf2_copy = tf2.copy()
        tf2_copy.residual_covariance = tf1.residual_covariance
        tf2_copy.transfer_function_error = tf1.transfer_function_error
        tf2_copy.transfer_function = tf1.transfer_function
        # Triage the creation time
        tf2_copy.station_metadata.provenance.creation_time = (
            tf1.station_metadata.provenance.creation_time
        )
        return tf1 == tf2_copy

    else:
        return False
