import pandas as pd


class EMTFBandSetupFile:
    def __init__(self, **kwargs):
        """

        Parameters
        ----------
        sample_rate: float
            this is the sampling rate of the un-decimated data,
            aka decimation level 1 in EMTF (and probably zero in the
            aurora nomenclature)
        kwargs
        """
        self.filepath = kwargs.get("filepath", None)
        self.df = None
        self.n_bands = None
        self.sample_rate = None

    def load(self, filepath=None):
        if filepath is None:
            filepath = self.filepath
        print(filepath)
        f = open(str(filepath), "r")
        n_bands = f.readline()
        self.n_bands = int(n_bands)
        f.close()
        df = pd.read_csv(
            filepath,
            skiprows=1,
            sep="\s+",
            names=["decimation_level", "lower_bound_index", "upper_bound_index"],
        )
        if len(df) != self.n_bands:
            print(f"unexpected number of bounds read in from {filepath}")
            raise Exception
        self.df = df

    def get_decimation_level(self, decimation_level, order="ascending_frequency"):
        if self.df is None:
            self.load()
        decimation_level_df = self.df[self.df["decimation_level"] == decimation_level]
        if order == "ascending_frequency":
            decimation_level_df = decimation_level_df.sort_values(
                by="lower_bound_index"
            )

        return decimation_level_df

    def to_band_averaging_scheme(self):
        """
        probably better to give band averaging scheme a "from emtf"
        method
        Returns
        -------

        """
        pass
