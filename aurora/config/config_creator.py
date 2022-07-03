"""
Helper class to make config files.

Note: the config is still evolving and this class and its methods are expected to
change.


"""
from pathlib import Path

from aurora.config import Processing, Station, Run, BANDS_DEFAULT_FILE


class ConfigCreator:
    
    def __init__(self, **kwargs):
        default_config_path = Path("config")
        self.config_path = kwargs.get("config_path", default_config_path)



    
    def create_run_processing_object(
            self, station_id=None, run_id=None, mth5_path=None, sample_rate=1, 
            input_channels=["hx", "hy"], output_channels=["hz", "ex", "ey"],
            estimator=None,
            emtf_band_file=BANDS_DEFAULT_FILE, **kwargs):
        """
        Create a default processing object
        
        :return: DESCRIPTION
        :rtype: TYPE

        """
        processing_id = f"{station_id}-{run_id}"
        processing_obj = Processing(id=processing_id, **kwargs)
        
        if not isinstance(run_id, list):
            run_id = [run_id]
            
        runs = []
        for run in run_id:
            run_obj = Run(
                id=run_id,
                input_channels=input_channels,
                output_channels=output_channels,
                sample_rate=sample_rate)
            runs.append(run_obj)
            
        station_obj = Station(id=station_id, mth5_path=mth5_path)
        station_obj.runs = runs

        processing_obj.stations.local = station_obj
        if emtf_band_file is not None:
            processing_obj.read_emtf_bands(emtf_band_file)
        
            for key in sorted(processing_obj.decimations_dict.keys()):
                if key in [0, "0"]:
                    d = 1
                    sr = sample_rate
                else:
                    d = 4
                    sr = sample_rate / (d ** int(key))
                decimation_obj = processing_obj.decimations_dict[key]
                decimation_obj.decimation.factor = d
                decimation_obj.decimation.sample_rate = sr
                decimation_obj.input_channels = input_channels
                decimation_obj.output_channels = output_channels
                #set estimator if provided as kwarg
                if estimator:
                    try:
                        decimation_obj.estimator.engine = estimator["engine"]
                    except KeyError:
                        pass
        return processing_obj
    
    def to_json(self, processing_obj, path=None, nested=True, required=False):
        """
        Write a processing object to path
        
        :param path: DESCRIPTION
        :type path: TYPE
        :param processing_obj: DESCRIPTION
        :type processing_obj: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        json_fn = processing_obj.json_fn()#config_id + "_run_config.json"
        if path is None:
            json_fn = processing_obj.json_fn()#config_id + "_run_config.json"
            self.config_path.mkdir(exist_ok=True)
            path = self.config_path.joinpath(json_fn)
        with open(path, "w") as fid:
            fid.write(processing_obj.to_json(nested=nested, required=required))