from pathlib import Path

from aurora.config.processing_config import RunConfig
from aurora.pipelines.process_mth5 import process_mth5_run

from make_parkfield_mth5 import test_make_parkfield_mth5
from make_processing_configs import create_run_test_config


def test():
    processing_run_cfg = create_run_test_config()

    config = RunConfig()
    config.from_json(processing_run_cfg)
    mth5_path = Path(config.mth5_path)

    # Ensure there is an mth5 to process
    if not mth5_path.exists():
        test_make_parkfield_mth5()

    run_id = "001"
    show_plot = True
    tf_collection = process_mth5_run(
        processing_run_cfg, run_id, mth5_path=mth5_path, units="SI", show_plot=show_plot
    )
    return tf_collection


def main():
    test()


if __name__ == "__main__":
    main()
