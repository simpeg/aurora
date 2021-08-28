from pathlib import Path
from aurora.pipelines.process_mth5 import process_mth5_run
from aurora.general_helper_functions import TEST_PATH

processing_cfg = Path(TEST_PATH, "parkfield", "config", "pkd_run_config.json")
run_id = "001"
tf_collection = process_mth5_run(processing_cfg, run_id, units="SI")
