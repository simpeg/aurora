from pathlib import Path
from aurora.general_helper_functions import TEST_PATH
from aurora.general_helper_functions import SANDBOX
from aurora.pipelines.process_mth5 import process_mth5_run
from aurora.sandbox.processing_config import RunConfig

processing_run_cfg = Path(TEST_PATH, "parkfield", "config", "pkd_run_config.json")
mth5_path = Path(TEST_PATH, "parkfield", "data", "pkd_test_00.h5")


run_id = "001"

# Ensure there is an mth5 to process
if not mth5_path.exists():
    from make_parkfield_mth5 import test_make_parkfield_mth5

    test_make_parkfield_mth5()

# Overwrite local paths in the processing config
# temporary solution until we close aurora issue #71
config = RunConfig()
config.from_json(processing_run_cfg)
band_setup_file = SANDBOX.joinpath("bs_256.cfg")
for i_dec in config.decimation_level_ids:
    config.decimation_level_configs[0].emtf_band_setup_file = band_setup_file
processing_run_cfg = config

tf_collection = process_mth5_run(
    processing_run_cfg, run_id, mth5_path=mth5_path, units="SI"
)
