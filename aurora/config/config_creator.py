"""
Helper class to make config files.

Note: the config is still evolving and this class and its methods are expected to
change.


"""
from loguru import logger

from aurora.config.metadata.processing import Processing
from aurora.config import BANDS_DEFAULT_FILE
from mt_metadata.transfer_functions.processing.aurora.window import Window
from aurora.sandbox.io_helpers.emtf_band_setup import EMTFBandSetupFile

SUPPORTED_BAND_SPECIFICATION_STYLES = ["EMTF", "band_edges"]


class ConfigCreator:
    def __init__(self, **kwargs):
        self._emtf_band_file = kwargs.get("emtf_band_file", None)
        self._band_edges = kwargs.get("band_edges", None)
        self._band_specification_style = None

    def processing_id(self, kernel_dataset):
        """
        In the past, we used f"{local}-{remote}" or  f"{local}-{run_id}"
        Neither of these is sufficiently unique.  In fact, they only describe the
        dataset, and not the processing config.  It is difficult to see how to make a
        comprehensive, unique id without it being very long or involving hash
        functions.

        For now, will try to use {local}-{local_runs}-{remote}-{remote_runs},
        which at least describes the dataset, then a string can be generated by the
        config and appended if needed.


        Parameters
        ----------
        kernel_dataset

        Returns
        -------

        """
        id = f"{kernel_dataset.local_station_id}-{kernel_dataset.remote_station_id}"
        return id

    @property
    def band_specification_style(self):
        return self._band_specification_style

    @band_specification_style.setter
    def band_specification_style(self, value):
        if value not in SUPPORTED_BAND_SPECIFICATION_STYLES:
            msg = f"Won't set band specification style to unrecognized value {value}"
            logger.warning(msg)
            raise NotImplementedError(msg)
            # return
        else:
            self._band_specification_style = value

    def determine_band_specification_style(self):
        """
        TODO: Should emtf_band_file path be stored in config to support reproducibility?

        """

        if (self._emtf_band_file is None) & (self._band_edges is None):
            msg = "Bands not defined; setting to EMTF BANDS_DEFAULT_FILE"
            logger.info(msg)
            self._emtf_band_file = BANDS_DEFAULT_FILE
            self._band_specification_style = "EMTF"
        elif (self._emtf_band_file is not None) & (self._band_edges is not None):
            msg = "Bands defined twice, and possibly inconsistently"
            logger.error(msg)
            raise ValueError(msg)
        elif self._band_edges is not None:
            self._band_specification_style = "band_edges"
        elif self._emtf_band_file is not None:
            self._band_specification_style = "EMTF"

    def create_from_kernel_dataset(
        self,
        kernel_dataset,
        input_channels=["hx", "hy"],
        output_channels=["hz", "ex", "ey"],
        estimator=None,
        **kwargs,
    ):
        """
        Hmmm, why not make this a method of kernel_dataset??

        Early on we want to know how may decimation levels there will be.
        This is defined either by:
         1. decimation_factors argument (normally accompanied by a bands_dict)
         2. number of decimations implied by EMTF band setup file.
        Theoretically, you could also use the number of decimations implied by
        bands_dict but this is sloppy, because it would be bad practice to assume
        the decimation factor.

        Notes:
        1.  2022-09-10
        The reading-in from EMTF band setup file used to be very terse, carried
        some baked in assumptions about decimation factors, and did not acknowlege
        specific frequency bands in Hz.  I am adding some complexity to the method
        that populates bands from EMTF band setup file but am now explict about the
        assumtion of decimation factors, and do provide the frequency bands in Hz.


        Parameters
        ----------
        kernel_dataset
        emtf_band_file: while the default here is None, it will get assigned the
        value BANDS_DEFAULT_FILE in the set_frequecy_bands method if band edges is
        also None.
        input_channels
        output_channels
        estimator
        band_edges
        kwargs

        Returns
        -------

        """

        processing_id = self.processing_id(kernel_dataset)
        processing_obj = Processing(id=processing_id)  # , **kwargs)

        # pack station and run info into processing object
        processing_obj.stations.from_dataset_dataframe(kernel_dataset.df)

        # Unpack kwargs
        self._emtf_band_file = kwargs.get("emtf_band_file", None)
        self._band_edges = kwargs.get("band_edges", None)
        decimation_factors = kwargs.get("decimation_factors", None)
        num_samples_window = kwargs.get("num_samples_window", None)

        # determine window parameters:
        # check if they have been passed as kwargs, otherwise extract default values

        # Determine if band_setup or edges dict is to be used for bands
        self.determine_band_specification_style()

        # Set Frequency Bands
        if self.band_specification_style == "EMTF":
            # see note 1
            emtf_band_setup_file = EMTFBandSetupFile(
                filepath=self._emtf_band_file, sample_rate=kernel_dataset.sample_rate
            )
            num_decimations = emtf_band_setup_file.num_decimation_levels
            if decimation_factors is None:
                # set default values to EMTF default values [1, 4, 4, 4, ..., 4]
                decimation_factors = num_decimations * [4]
                decimation_factors[0] = 1
            if num_samples_window is None:
                default_window = Window()
                num_samples_window = num_decimations * [default_window.num_samples]
            elif isinstance(num_samples_window, int):
                num_samples_window = num_decimations * [num_samples_window]
            # now you can define the frequency bands
            band_edges = emtf_band_setup_file.compute_band_edges(
                decimation_factors, num_samples_window
            )
            self._band_edges = band_edges

        processing_obj.assign_bands(
            self._band_edges,
            kernel_dataset.sample_rate,
            decimation_factors,
            num_samples_window,
        )
        processing_obj.band_specification_style = self.band_specification_style
        if self.band_specification_style == "EMTF":
            processing_obj.band_setup_file = str(self._emtf_band_file)
        for key, decimation_obj in processing_obj.decimations_dict.items():
            decimation_obj.input_channels = input_channels
            decimation_obj.output_channels = output_channels
            if num_samples_window is not None:
                decimation_obj.window.num_samples = num_samples_window[key]
            # set estimator if provided as kwarg
            if estimator:
                try:
                    decimation_obj.estimator.engine = estimator["engine"]
                except KeyError:
                    pass
        return processing_obj
