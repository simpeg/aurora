"""
This module contains some helper functions that are called during the
execution of aurora's tests of processing on synthetic data.
"""

import mt_metadata.transfer_functions
import pathlib
from aurora.pipelines.process_mth5 import process_mth5
from aurora.test_utils.synthetic.make_processing_configs import (
    make_processing_config_and_kernel_dataset,
)

from mth5.data.make_mth5_from_asc import create_test1_h5
from mth5.data.make_mth5_from_asc import create_test2_h5
from mth5.data.make_mth5_from_asc import create_test12rr_h5

from typing import Optional, Union

def get_example_kernel_dataset(num_stations: int = 1):
    """
    Creates a kernel dataset object from the synthetic data
     - Helper function for synthetic tests.

    Returns
    -------
    kernel_dataset: aurora.transfer_function.kernel_dataset.KernelDataset
        The kernel dataset from a synthetic, single station mth5
    """

    from mth5.processing import RunSummary, KernelDataset

    if num_stations == 1:
        mth5_path = create_test1_h5(force_make_mth5=False)
    elif num_stations == 2:
        mth5_path = create_test12rr_h5(force_make_mth5=False)

    run_summary = RunSummary()
    run_summary.from_mth5s(
        [
            mth5_path,
        ]
    )

    kernel_dataset = KernelDataset()
    station_id = run_summary.df.station.iloc[0]
    if num_stations == 1:
        kernel_dataset.from_run_summary(
            run_summary=run_summary, local_station_id=station_id
        )
    elif num_stations == 2:
        remote_station_id = run_summary.df.station.iloc[1]
        kernel_dataset.from_run_summary(
            run_summary=run_summary,
            local_station_id=station_id,
            remote_station_id=remote_station_id,
        )
    return kernel_dataset


def tf_obj_from_synthetic_data(
    mth5_path: pathlib.Path,
) -> mt_metadata.transfer_functions.TF:
    """
    Executes aurora processing on mth5_path, and returns mt_metadata TF object.
    - Helper function for test_issue_139

    """
    from aurora.config.config_creator import ConfigCreator
    from mth5.processing import RunSummary, KernelDataset

    run_summary = RunSummary()
    run_summary.from_mth5s(list((mth5_path,)))

    kernel_dataset = KernelDataset()
    kernel_dataset.from_run_summary(run_summary, "test1", "test2")

    # Define the processing Configuration
    cc = ConfigCreator()
    config = cc.create_from_kernel_dataset(kernel_dataset)

    tf_cls = process_mth5(
        config,
        kernel_dataset,
        units="MT",
        z_file_path="test1_RRtest2.zrr",
    )
    return tf_cls


def process_synthetic_1(
    config_keyword: Optional[str] = "test1",
    z_file_path: Optional[Union[str, pathlib.Path]] = "",
    test_scale_factor: Optional[bool] = False,
    simultaneous_regression: Optional[bool] = False,
    file_version: Optional[str] = "0.1.0",  # TODO: set to Literal["0.1.0", "0.2.0"]
    return_collection: Optional[bool] = False,
    channel_nomenclature: Optional[str] = "default",
    reload_config: Optional[bool] = False,
):
    """

    Parameters
    ----------
    config_keyword: str
        "test1", "test1_tfk", this is an argument passed to the create_test_run_config
        as test_case_id.
    z_file_path: str or path
        Where the z-file will be output
    test_scale_factor: bool
        If true, will assign scale factors to the channels
    simultaneous_regression: bool
        If True will do regression all outut channels in one step, rather than the
        usual, channel-by-channel method
    file_version: str
        one of ["0.1.0", "0.2.0"]

    Returns
    -------
    tf_result: TransferFunctionCollection or mt_metadata.transfer_functions.TF
        Should change so that it is mt_metadata.TF (see Issue #143)
    """
    mth5_path = create_test1_h5(
        file_version=file_version, channel_nomenclature=channel_nomenclature
    )
    mth5_paths = [
        mth5_path,
    ]
    station_id = "test1"
    tfk_dataset, processing_config = make_processing_config_and_kernel_dataset(
        config_keyword=station_id,
        station_id=station_id,
        remote_id=None,  # TODO: allow empty str instead of None
        mth5s=mth5_paths,
        channel_nomenclature=channel_nomenclature,
    )

    # Test that channel_scale_factors column is optional
    if test_scale_factor:
        scale_factors = {
            "ex": 10.0,
            "ey": 3.0,
            "hx": 6.0,
            "hy": 5.0,
            "hz": 100.0,
        }
        tfk_dataset.df["channel_scale_factors"].at[0] = scale_factors
    else:
        tfk_dataset.df.drop(columns=["channel_scale_factors"], inplace=True)

    # Relates to issue #172
    # reload_config = True
    # if reload_config:
    #     from mt_metadata.transfer_functions.processing.aurora import Processing
    #     p = Processing()
    #     config_path = pathlib.Path("config")
    #     json_fn = config_path.joinpath(processing_config.json_fn())
    #     p.from_json(json_fn)

    if simultaneous_regression:
        for decimation in processing_config.decimations:
            decimation.estimator.estimate_per_channel = False

    tf_result = process_mth5(
        processing_config,
        tfk_dataset=tfk_dataset,
        z_file_path=z_file_path,
        return_collection=return_collection,
    )

    if return_collection:
        z_figure_name = z_file_path.name.replace("zss", "png")
        for xy_or_yx in ["xy", "yx"]:
            ttl_str = f"{xy_or_yx} component, test_scale_factor = {test_scale_factor}"
            out_png_name = f"{xy_or_yx}_{z_figure_name}"
            tf_result.rho_phi_plot(
                xy_or_yx=xy_or_yx,
                ttl_str=ttl_str,
                show=False,
                figure_basename=out_png_name,
                figures_path=AURORA_RESULTS_PATH,
            )
    return tf_result

def process_synthetic_2(
    force_make_mth5: Optional[bool] = True,
    z_file_path: Optional[Union[str, pathlib.Path, None]] = None,
    save_fc: Optional[bool] = False,
    file_version: Optional[str] = "0.2.0",
    channel_nomenclature: Optional[str] = "default",
):
    """"""
    station_id = "test2"
    mth5_path = create_test2_h5(
        force_make_mth5=force_make_mth5, file_version=file_version
    )
    mth5_paths = [
        mth5_path,
    ]

    tfk_dataset, processing_config = make_processing_config_and_kernel_dataset(
        config_keyword=station_id,
        station_id=station_id,
        remote_id=None,
        mth5s=mth5_paths,
        channel_nomenclature=channel_nomenclature,
    )

    for decimation_level in processing_config.decimations:
        if save_fc:
            decimation_level.save_fcs = True
            decimation_level.save_fcs_type = "h5"  # h5 instead of "csv"

    tfc = process_mth5(
        processing_config,
        tfk_dataset=tfk_dataset,
        z_file_path=z_file_path,
    )
    return tfc

def process_synthetic_1r2(
    config_keyword="test1r2",
    channel_nomenclature="default",
    return_collection=False,
):
    mth5_path = create_test12rr_h5(channel_nomenclature=channel_nomenclature)
    mth5_paths = [
        mth5_path,
    ]

    tfk_dataset, processing_config = make_processing_config_and_kernel_dataset(
        config_keyword,
        station_id="test1",
        remote_id="test2",
        mth5s=mth5_paths,
        channel_nomenclature=channel_nomenclature,
    )

    tfc = process_mth5(
        processing_config,
        tfk_dataset=tfk_dataset,
        return_collection=return_collection,
    )
    return tfc