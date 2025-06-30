"""

Integrated test of the functionality of feature weights.

1. This test uses degraded sythetic data to test the feature weighting.
Noise is added to some fraction (50-75%) of the data.

Then regular (single station) processing is called on the data and
feature weighting processing is called on the data.

"""

from aurora.config.metadata import Processing
from aurora.config.metadata.processing import _processing_obj_from_json_file
from aurora.general_helper_functions import TEST_PATH
from aurora.general_helper_functions import PROCESSING_TEMPLATES_PATH
from aurora.general_helper_functions import MT_METADATA_FEATURES_TEST_HELPERS_PATH
from aurora.pipelines.process_mth5 import process_mth5
from aurora.test_utils.synthetic.paths import SyntheticTestPaths
from mth5.data.make_mth5_from_asc import create_test1_h5
from mth5.data.make_mth5_from_asc import create_test12rr_h5
from mth5.mth5 import MTH5
from mt_metadata.features.weights.channel_weight_spec import ChannelWeightSpec

import json
import numpy as np
import pathlib
import unittest

import mt_metadata.transfer_functions

from loguru import logger
from mth5.timeseries import ChannelTS, RunTS
from typing import Optional


# TODO: this could be moved to a more general test utils file
def create_synthetic_mth5_with_noise(
    source_file: Optional[pathlib.Path] = None,
    target_file: Optional[pathlib.Path] = None,
    noise_channels=("ex", "hy"),
    frac=0.5,
    noise_level=1000.0,
    seed=None,
):
    """
    Copy a synthetic MTH5, injecting noise into specified channels for a fraction of the data.
    """
    if source_file is None:
        source_file = create_test1_h5(
            file_version="0.1.0",
            channel_nomenclature="default",
            force_make_mth5=True,
            target_folder=TEST_PATH.joinpath("synthetic"),
        )
    if target_file is None:
        target_file = TEST_PATH.joinpath("synthetic", "test1_noisy.h5")
        if target_file.exists():
            target_file.unlink()
    if seed is None:
        seed = 42  # Default seed for reproducibility

    rng = np.random.default_rng(seed)
    m_source = MTH5(source_file)
    m_source.open_mth5(mode="r")
    m_target = MTH5(target_file, file_version=m_source.file_version)
    m_target.open_mth5(mode="w")

    for station_id in m_source.station_list:
        station = m_source.get_station(station_id)
        if station_id not in m_target.station_list:
            m_target.add_station(station_id, station_metadata=station.metadata)
        for run_id in station.run_summary["id"].unique():
            run = station.get_run(run_id)
            ch_list = []
            for ch in run.channel_summary.component.to_list():
                ch_obj = run.get_channel(ch)
                ch_ts = ch_obj.to_channel_ts()
                data = ch_ts.data_array.data.copy()
                n = len(data)
                if ch in noise_channels:
                    noisy_idx = slice(0, int(frac * n))
                    noise = rng.normal(0, noise_level, size=data[noisy_idx].shape)
                    noise = noise.astype(
                        data.dtype
                    )  # Ensure noise is the same dtype as data
                    data[noisy_idx] += noise
                ch_ts.data_array.data = data
                ch_list.append(ch_ts)
            runts = RunTS(array_list=ch_list, run_metadata=run.metadata)
            runts.run_metadata.id = run_id
            target_station = m_target.get_station(station_id)
            target_station.add_run(run_id).from_runts(runts)
    m_source.close_mth5()
    m_target.close_mth5()
    return target_file


def _load_example_channel_weight_specs(
    keep_only=[
        "striding_window_coherence",
    ]
) -> list:
    """

    Loads example channel weight specifications from a JSON file.

    Modifies it for this test so that the feature_weight_specs are only striding_window_coherence.

    Parameters
    ----------
    keep_only: list
        List of feature names to keep in the feature_weight_specs.
        Default is ["striding_window_coherence"].
    Returns
    -------
    output: list
        List of ChannelWeightSpec objects with modified feature_weight_specs.

    """
    feature_weight_json = MT_METADATA_FEATURES_TEST_HELPERS_PATH.joinpath(
        "channel_weight_specs_example.json"
    )
    assert (
        feature_weight_json.exists()
    ), f"Could not find feature weighting block json at {feature_weight_json}"

    with open(feature_weight_json, "r") as f:
        data = json.load(f)

    output = []
    channel_weight_specs = data.get("channel_weight_specs", data)
    for cws_dict in channel_weight_specs:
        cws = ChannelWeightSpec()
        cws.from_dict(cws_dict)

        # Modify the feature_weight_specs to only include striding_window_coherence
        if keep_only:
            cws.feature_weight_specs = [
                fws for fws in cws.feature_weight_specs if fws.feature.name in keep_only
            ]
            # get rid of Remote reference channels (work in progress)
            cws.feature_weight_specs = [
                fws for fws in cws.feature_weight_specs if fws.feature.ch2 != "rx"
            ]
            cws.feature_weight_specs = [
                fws for fws in cws.feature_weight_specs if fws.feature.ch2 != "ry"
            ]

        # Ensure that the feature_weight_specs is not empty
        if not cws.feature_weight_specs:
            msg = "No valid feature_weight_specs found in channel weight spec."
            logger.error(msg)
        else:
            output.append(cws)

    return output


def load_processing_objects() -> dict:
    """
    Loads the 'default' and 'with_weights' processing objects.

    'default' is loaded from the processing configuration template.
    'with_weights' is loaded from the same template but with channel weight specs
    set to only include 'striding_window_coherence'.

    Returns
    -------
    dict
        Dictionary with keys 'default' and 'with_weights' mapping to Processing objects.
    """
    processing_params_json = PROCESSING_TEMPLATES_PATH.joinpath(
        "processing_configuration_template.json"
    )

    processing_objects = {}
    processing_objects["default"] = _processing_obj_from_json_file(
        processing_params_json
    )

    cws_list = _load_example_channel_weight_specs(
        keep_only=[
            "striding_window_coherence",
        ]
    )
    processing_objects["with_weights"] = _processing_obj_from_json_file(
        processing_params_json
    )
    processing_objects["with_weights"].decimations[0].channel_weight_specs = cws_list

    return processing_objects


def process_mth5_with_config(
    mth5_path: pathlib.Path, processing_obj: Processing, z_file="test1.zss"
) -> mt_metadata.transfer_functions.TF:
    """
    Executes aurora processing on mth5_path, and returns mt_metadata TF object.

    """
    from mth5.processing import RunSummary, KernelDataset

    run_summary = RunSummary()
    run_summary.from_mth5s(list((mth5_path,)))

    kernel_dataset = KernelDataset()
    kernel_dataset.from_run_summary(run_summary, "test1")
    config = processing_obj
    config.stations.remote = []  # TODO: allow this to be False
    for dec in config.decimations:
        dec.estimator.engine = "RME"
        dec.reference_channels = []

    tf_cls = process_mth5(
        config,
        kernel_dataset,
        units="MT",
        z_file_path=z_file,
        show_plot=False,
    )
    return tf_cls


def print_apparent_resistivity(tf, label="TF"):
    """
    Print apparent resistivity and phase for each period/frequency in the TF object.
    Returns the mean apparent resistivity (averaged over all frequencies and both Zxy/Zyx).
    """
    if not hasattr(tf, "impedance"):
        print(f"{label}: TF object missing impedance attribute.")
        return np.nan
    z = tf.impedance
    print(
        f"{label} impedance shape: {getattr(z, 'shape', None)}, dims: {getattr(z, 'dims', None)}"
    )

    # Get period and convert to frequency
    if hasattr(tf, "period"):
        period = np.array(tf.period)
        freq = 1.0 / period
    elif hasattr(tf, "frequency"):
        freq = np.array(tf.frequency)
    else:
        print(f"{label}: TF object missing period/frequency attribute.")
        return np.nan

    n_periods = z.shape[0]
    n_out = z.shape[1]
    n_in = z.shape[2]
    print(
        f"{label} n_periods={n_periods}, n_out={n_out}, n_in={n_in}, len(freq)={len(freq)}"
    )

    rho_vals = []
    for i in range(min(n_periods, len(freq))):
        f = freq[i]
        for comp, out_idx, in_idx in [("Zxy", 0, 1), ("Zyx", 1, 0)]:
            if out_idx < n_out and in_idx < n_in:
                zval = z[i, out_idx, in_idx]
                rho = (np.abs(zval) ** 2) / (2 * np.pi * f)
                phase = np.angle(zval, deg=True)
                print(
                    f"{label} f={f:.4g} Hz {comp}: rho={rho:.3g} ohm-m, phase={phase:.2f} deg"
                )
                rho_vals.append(rho)
            else:
                print(
                    f"{label} index out of bounds: out_idx={out_idx}, in_idx={in_idx}"
                )
    mean_rho = np.nanmean(rho_vals) if rho_vals else np.nan
    print(
        f"{label} MEAN apparent resistivity (all freqs, Zxy/Zyx): {mean_rho:.3g} ohm-m"
    )
    return mean_rho


# Uncomment the blocks below to run the test as a script
# def main():
#     SYNTHETIC_FOLDER = TEST_PATH.joinpath("synthetic")
#     # Create a synthetic mth5 file for testing
#     mth5_path = create_synthetic_mth5_with_noise()
#     # mth5_path = SYNTHETIC_FOLDER.joinpath("test1_noisy.h5")

#     processing_objects = load_processing_objects()

#     # TODO: compare this against stored template
#     # json_str = processing_objects["with_weights"].to_json()
#     # with open(SYNTHETIC_FOLDER.joinpath("used_processing.json"), "w") as f:
#     #     f.write(json_str)

#     process_mth5_with_config(
#         mth5_path, processing_objects["default"], z_file="test1_default.zss"
#     )
#     process_mth5_with_config(
#         mth5_path, processing_objects["with_weights"], z_file="test1_weights.zss"
#     )
#     from aurora.transfer_function.plot.comparison_plots import compare_two_z_files

#     compare_two_z_files(
#         z_path1=SYNTHETIC_FOLDER.joinpath("test1_default.zss"),
#         z_path2=SYNTHETIC_FOLDER.joinpath("test1_weights.zss"),
#         label1="default",
#         label2="weights",
#         scale_factor1=1,
#         out_file="output_png.png",
#         markersize=3,
#         rho_ylims=[1e-2, 5e2],
#         xlims=[1.0, 500],
#     )


def test_feature_weighting():
    SYNTHETIC_FOLDER = TEST_PATH.joinpath("synthetic")
    # Create a synthetic mth5 file for testing
    mth5_path = create_synthetic_mth5_with_noise()
    # mth5_path = SYNTHETIC_FOLDER.joinpath("test1_noisy.h5")

    processing_objects = load_processing_objects()
    z_path1 = SYNTHETIC_FOLDER.joinpath("test1_default.zss")
    z_path2 = SYNTHETIC_FOLDER.joinpath("test1_weights.zss")
    process_mth5_with_config(mth5_path, processing_objects["default"], z_file=z_path1)
    process_mth5_with_config(
        mth5_path, processing_objects["with_weights"], z_file=z_path2
    )

    from mt_metadata.transfer_functions import TF

    tf1 = TF(fn=z_path1)
    tf2 = TF(fn=z_path2)
    tf1.read()
    tf2.read()
    assert (
        tf1.impedance.data != tf2.impedance.data
    ).any(), "TF1 and TF2 should have different impedance values after processing with weights."

    print("TF1 Apparent Resistivity and Phase:")
    mean_rho1 = print_apparent_resistivity(tf1, label="TF1")
    print("TF2 Apparent Resistivity and Phase:")
    mean_rho2 = print_apparent_resistivity(tf2, label="TF2")
    print(
        f"\nSUMMARY: Mean apparent resistivity TF1: {mean_rho1:.3g} ohm-m, TF2: {mean_rho2:.3g} ohm-m"
    )


# Uncomment the blocks below to run the test as a script
# def main():
#     SYNTHETIC_FOLDER = TEST_PATH.joinpath("synthetic")
#     # Create a synthetic mth5 file for testing
#     mth5_path = create_synthetic_mth5_with_noise()
#     # mth5_path = SYNTHETIC_FOLDER.joinpath("test1_noisy.h5")

#     processing_objects = load_processing_objects()

#     # TODO: compare this against stored template
#     # json_str = processing_objects["with_weights"].to_json()
#     # with open(SYNTHETIC_FOLDER.joinpath("used_processing.json"), "w") as f:
#     #     f.write(json_str)

#     process_mth5_with_config(
#         mth5_path, processing_objects["default"], z_file="test1_default.zss"
#     )
#     process_mth5_with_config(
#         mth5_path, processing_objects["with_weights"], z_file="test1_weights.zss"
#     )
#     from aurora.transfer_function.plot.comparison_plots import compare_two_z_files

#     compare_two_z_files(
#         z_path1=SYNTHETIC_FOLDER.joinpath("test1_default.zss"),
#         z_path2=SYNTHETIC_FOLDER.joinpath("test1_weights.zss"),
#         label1="default",
#         label2="weights",
#         scale_factor1=1,
#         out_file="output_png.png",
#         markersize=3,
#         rho_ylims=[1e-2, 5e2],
#         xlims=[1.0, 500],
#     )

# if __name__ == "__main__":
#     main()
#     # test_feature_weighting()
