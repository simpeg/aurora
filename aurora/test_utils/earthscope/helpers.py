import pathlib

HOME = pathlib.Path().home()
CACHE_PATH = HOME.joinpath(".cache").joinpath("earthscope")
CACHE_PATH.mkdir(parents=True, exist_ok=True)

DATA_AVAILABILITY_PATH = CACHE_PATH.joinpath("data_availability")
DATA_AVAILABILITY_PATH.mkdir(parents=True, exist_ok=True)
PUBLIC_DATA_AVAILABILITY_PATH = DATA_AVAILABILITY_PATH.joinpath("public")
PUBLIC_DATA_AVAILABILITY_PATH.mkdir(parents=True, exist_ok=True)
RESTRICTED_DATA_AVAILABILITY_PATH = DATA_AVAILABILITY_PATH.joinpath("restricted")
RESTRICTED_DATA_AVAILABILITY_PATH.mkdir(parents=True, exist_ok=True)
DATA_AVAILABILITY_CSV = DATA_AVAILABILITY_PATH.joinpath("MT_acquisitions.csv")

DATA_PATH = CACHE_PATH.joinpath("data")
DATA_PATH.mkdir(parents=True, exist_ok=True)

SPUD_XML_PATH = CACHE_PATH.joinpath("spud_xml")
SPUD_XML_CSV = SPUD_XML_PATH.joinpath("spud_summary.csv")
SPUD_EMTF_PATH = SPUD_XML_PATH.joinpath("emtf")
SPUD_DATA_PATH = SPUD_XML_PATH.joinpath("data")
SPUD_EMTF_PATH.mkdir(parents=True, exist_ok=True)
SPUD_DATA_PATH.mkdir(parents=True, exist_ok=True)
