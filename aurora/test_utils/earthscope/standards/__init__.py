# package file
from pathlib import Path

SCHEMA_PATH = Path(__file__).parent

SCHEMA_FN_PATHS = list(SCHEMA_PATH.glob("*.json"))

csvs  = list(SCHEMA_PATH.glob("schema*.csv"))
SCHEMA_CSVS = {int(csv.stem.split("_")[-1]):csv for csv in csvs}
