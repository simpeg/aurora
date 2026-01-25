"""
Integrated test of the functionality of feature weights.

1. This test uses degraded synthetic data to test the feature weighting.
Noise is added to some fraction (50-75%) of the data.

Then regular (single station) processing is called on the data and
feature weighting processing is called on the data.

---
Feature weights are specified using the mt_metadata.features.weights module.
This test demonstrates how feature-based channel weighting (e.g., striding_window_coherence)
can be injected into Aurora's processing pipeline. In the future, these features will be
used to enable more robust, data-driven weighting strategies for transfer function estimation,
including integration of new features from mt_metadata and more flexible weighting schemes.

See also: mt_metadata.features.weights.channel_weight_spec and test_feature_weighting.py for
examples of how to define, load, and use feature weights in Aurora workflows.
"""

import json
import pathlib
from typing import Optional

import numpy as np
from loguru import logger
from mt_metadata.features.weights.channel_weight_spec import ChannelWeightSpec
from mt_metadata.transfer_functions import TF
from mth5.mth5 import MTH5
from mth5.processing import KernelDataset, RunSummary
from mth5.timeseries import RunTS

from aurora.config.metadata import Processing
from aurora.config.metadata.processing import _processing_obj_from_json_file
from aurora.general_helper_functions import (
    MT_METADATA_FEATURES_TEST_HELPERS_PATH,
    PROCESSING_TEMPLATES_PATH,
    TEST_PATH,
)
from aurora.pipelines.process_mth5 import process_mth5


def create_synthetic_mth5_with_noise(
    source_file: pathlib.Path,
    target_file: Optional[pathlib.Path] = None,
    noise_channels=("ex", "hy"),
    frac=0.5,
    noise_level=1000.0,
    seed=None,
):
    """
    Copy a synthetic MTH5, injecting noise into specified channels for a fraction of the data.
    """
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
        # Unwrap the nested structure
        cws_data = cws_dict.get("channel_weight_spec", cws_dict)

        # Process feature_weight_specs to unwrap nested dicts
        if "feature_weight_specs" in cws_data:
            fws_list = []
            for fws_item in cws_data["feature_weight_specs"]:
                fws_data = fws_item.get("feature_weight_spec", fws_item)
                fws_list.append(fws_data)
            cws_data["feature_weight_specs"] = fws_list

        # Construct directly from dict to ensure proper deserialization
        cws = ChannelWeightSpec(**cws_data)

        # Modify the feature_weight_specs to only include striding_window_coherence
        if keep_only:
            cws.feature_weight_specs = [
                fws for fws in cws.feature_weight_specs if fws.feature.name in keep_only
            ]
            # get rid of Remote reference channels (work in progress)
            cws.feature_weight_specs = [
                fws for fws in cws.feature_weight_specs if fws.feature.channel_2 != "rx"
            ]
            cws.feature_weight_specs = [
                fws for fws in cws.feature_weight_specs if fws.feature.channel_2 != "ry"
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
) -> TF:
    """
    Executes aurora processing on mth5_path, and returns mt_metadata TF object.
    """
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


def test_feature_weighting(synthetic_test_paths, worker_safe_test1_h5):
    """Test that feature weighting affects TF processing results."""
    SYNTHETIC_FOLDER = synthetic_test_paths.aurora_results_path.parent

    # Create a synthetic mth5 file for testing
    mth5_path = create_synthetic_mth5_with_noise(source_file=worker_safe_test1_h5)

    processing_objects = load_processing_objects()
    z_path1 = SYNTHETIC_FOLDER.joinpath("test1_default.zss")
    z_path2 = SYNTHETIC_FOLDER.joinpath("test1_weights.zss")
    process_mth5_with_config(mth5_path, processing_objects["default"], z_file=z_path1)
    process_mth5_with_config(
        mth5_path, processing_objects["with_weights"], z_file=z_path2
    )

    tf1 = TF(fn=z_path1)
    tf2 = TF(fn=z_path2)
    tf1.read(**{"rotate_to_measurement_coordinates": False})
    tf2.read(**{"rotate_to_measurement_coordinates": False})

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
