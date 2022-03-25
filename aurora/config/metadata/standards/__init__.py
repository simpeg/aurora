# package file
from pathlib import Path

SCHEMA_PATH = Path(__file__).parent

SCHEMA_FN_PATHS = list(SCHEMA_PATH.glob("*.json"))
