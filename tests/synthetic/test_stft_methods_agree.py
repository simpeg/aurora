"""
Pseudocode
1. Load a run_xrts
2. compute stft from
"""
import numpy as np

from pathlib import Path

from aurora.pipelines.process_mth5 import get_data_from_decimation_level_from_mth5
from aurora.pipelines.process_mth5 import initialize_pipeline
from aurora.pipelines.process_mth5 import prototype_decimate
from aurora.pipelines.processing_helpers import run_ts_to_stft
from aurora.pipelines.processing_helpers import run_ts_to_stft_scipy

def test_stft_methods_agree():
    run_config = Path("config", "test1_run_config_standard.json")
    run_id = "001"
    run_config, mth5_obj = initialize_pipeline(run_config)

    local_run_obj = mth5_obj.get_run(run_config["local_station_id"], run_id)
    local_run_ts = local_run_obj.to_runts()


    for dec_level_id in run_config.decimation_level_ids:
        processing_config = run_config.decimation_level_configs[dec_level_id]
        processing_config.local_station_id = run_config.local_station_id
#processing_config.reference_station_id = run_config.reference_station_id


        if dec_level_id == 0:
            local, remote = get_data_from_decimation_level_from_mth5(
                processing_config, mth5_obj, run_id)
        else:
            local = prototype_decimate(processing_config, local)
            # if processing_config.reference_station_id:
            #     remote = prototype_decimate(processing_config, remote)

        # </GET DATA>
        local_run_obj = local["run"]
        local_run_xrts = local["mvts"]
        processing_config.extra_pre_fft_detrend_type = ""
        local_stft_obj = run_ts_to_stft(processing_config, local_run_xrts)
        local_stft_obj2 = run_ts_to_stft_scipy(processing_config, local_run_xrts)
        stft_difference = local_stft_obj - local_stft_obj2
        stft_difference = stft_difference.to_array()

        #drop dc term
        stft_difference = stft_difference.where(stft_difference.frequency > 0,drop=True)

        assert np.isclose(stft_difference, 0).all()

        # ifc = 3
        #
        # qq = np.median(np.abs(local_stft_obj2["ex"][:, ifc]) / np.abs(
        #     local_stft_obj["ex"][:, ifc]))
        # print(qq)
#        np.abs(1.41 * local_stft_obj2["ex"][:, ifc]) / np.abs(
#            local_stft_obj["ex"][:, ifc])
        print("compare")
    #     # local_stft_obj = run_ts_to_stft_scipy(config, local_run_xrts)
    #     local_stft_obj = calibrate_stft_obj(local_stft_obj, local_run_obj,
    #                                         units=units)
    #     # local_stft_obj = run_ts_to_calibrated_stft(local_run_ts, local_run_obj,
    #     #                                          config, units=units)
    #     remote_run_obj = remote["run"]
    #     remote_run_xrts = remote["mvts"]
    #     if config.reference_station_id:
    #         remote_stft_obj = run_ts_to_stft(config, remote_run_xrts)
    #         remote_stft_obj = calibrate_stft_obj(remote_stft_obj,
    #                                              remote_run_obj,
    #                                              units=units)
    #         remote_stft_obj = remote_stft_obj.to_array("channel")
    #         # remote_stft_obj = run_ts_to_calibrated_stft(remote_run_ts,
    #         #                                             remote_run_obj,
    #         #                                             config, units=units)
    #     else:
    #         remote_stft_obj = None
    #
    #     tf_obj = process_mth5_decimation_level(processing_config, local,
    #                                            remote, units=units)
    # pass




def main():
    test_stft_methods_agree()

if __name__ == '__main__':
    main()
