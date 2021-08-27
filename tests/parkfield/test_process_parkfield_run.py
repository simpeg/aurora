import os
from pathlib import Path
from aurora.pipelines.process_mth5 import process_mth5_run

here = os.getcwd()
processing_cfg = Path(here, "config", "pkd_run_config.json")
run_id = "001"
tf_collection = process_mth5_run(processing_cfg, run_id, units="SI")
