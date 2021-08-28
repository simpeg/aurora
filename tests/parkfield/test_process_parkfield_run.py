from pathlib import Path
from aurora.pipelines.process_mth5 import process_mth5_run
from aurora.general_helper_functions import TEST_PATH

processing_cfg = Path(TEST_PATH, "parkfield", "config", "pkd_run_config.json")
mth5_path = Path(TEST_PATH, "parkfield", "data", "pkd_test_00.h5")

run_id = "001"

if not mth5_path.exists():
    from make_parkfield_mth5 import test_make_parkfield_mth5

    test_make_parkfield_mth5()

tf_collection = process_mth5_run(
    processing_cfg, run_id, mth5_path=mth5_path, units="SI"
)
