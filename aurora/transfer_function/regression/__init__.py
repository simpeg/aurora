from aurora.transfer_function.regression.base import RegressionEstimator
from aurora.transfer_function.regression.TRME import TRME
from aurora.transfer_function.regression.TRME_RR import TRME_RR
from loguru import logger

ESTIMATOR_LIBRARY = {"OLS": RegressionEstimator, "RME": TRME, "RME_RR": TRME_RR}


def get_regression_estimator(estimation_engine):
    """

    Parameters
    ----------
    estimation_engine: str
        One of the keys in the ESTIMATOR_LIBRARY, designates the method that will be
        used to estimate the transfer function

    Returns
    -------
    estimator_class: aurora.transfer_function.regression.base.RegressionEstimator
        The class that will do the TF estimation
    """
    try:
        estimator_class = ESTIMATOR_LIBRARY[estimation_engine]
    except KeyError:
        logger.error(f"processing_scheme {estimation_engine} not supported")
        logger.error(
            f"processing_scheme must be one of {list(ESTIMATOR_LIBRARY.keys())}"
        )
        raise Exception
    return estimator_class
