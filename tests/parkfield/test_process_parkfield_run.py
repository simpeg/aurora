from pathlib import Path
from aurora.pipelines.process_mth5 import process_mth5_run




processing_cfg = Path("config", "pkd_run_config.json")
tf_collection = process_mth5_run(processing_cfg, "001", units="SI")
