"""
follows Gary's TFHeader.m
iris_mt_scratch/egbert_codes-20210121T193218Z-001/egbert_codes/matlabPrototype_10-13-20/TF/classes
"""

class TransferFunctionHeader(object):
    """
    class for storing metadata for a TF estimate
    """

    def __init__(self, **kwargs):
        """
        Parameters
        local_site : mt_metadata.station object ?@jared is this correct class?
            Station metadata object for the station to be estimated (
            location, channel_azimuths, etc.)
        remote_site: same object type as local site
            if no remote reference then this can be None
        output_channels: list
            Probably a list of channel keys -- usually ["ex","ey","hz"]
        input_channels : list
            Probably a list of channel keys -- usually ["hx","hy"]
            These are the channels being provided as input to the regression.
        reference_channels : list
            These are the channels being used from the RR station. This is a
            channel list -- usually [?, ?]
        processing_scheme: str
            One of "single station" or "remote reference".  Future versions
            will include , "multivariate array", "multiple remote",
            etc.

        """
        self.processing_scheme = kwargs.get("processing_scheme",
                                            "single_station")
        self.local_site = kwargs.get("local_site", None)
        self.remote_site = kwargs.get("remote_site", None)
        self.input_channels = kwargs.get("input_channels", ["hx", "hy"])
        self.output_channels = kwargs.get("output_channels", ["ex", "ey"])
        self.reference_channels = kwargs.get("reference_channels", [])
        self.user_meta_data = None #placeholder for anything

    @property
    def num_input_channels(self):
        return len(self.input_channels)

    @property
    def num_output_channels(self):
        return len(self.output_channels)

    #TO BE DEPRECATED
    def array_header_to_tf_header(self, array_header, sites):
        """
        This will likely be made from MTH5.  The overarching point of this
        method is to
        1. associated the right station header with the local site
        2. populate the input and output channels,
        3. populate the remote channels

        Notes that the population of input and output channels is done
        based on data_availability and basically can be done at the
        config level

        This entire method can probably be replaced with a proper config, and
        a processing_config_validation


        Looks like SITES are basically keys to array_header which is basically
        a dict of station_headers.

        It looks like this assigns the processing_scheme
        %   Usage: obj = ArrayHeader2TFHeader(obj,ArrayHD, OPTIONS)
            %        given an input TArrayHeader,
            %   and SITES, a structure defining:
            %          RR    -- true for RR processing, otherwise SS
            %          LocalSite   -- site ID or number for local site
            %          RemoteSite  -- site ID or number for Reference site
            %          VTF     --  true if Vertical Field TF should also be estimated
            %
            %          Could add more if we want to generalize to other
            %          estimation schemes
            %    this always uses horizontal magnetics for ChIn and ChRef,
            %          electrics and Hz for ChOut -- could generalize

        Parameters
        ----------
        array_header
        sites

        Returns
        -------

        """
        #find local site number if a character string is provide
        if ischar(SITES.LocalSite):
            LocalInd = find(strcmp(SITES.LocalSite, array_header.SiteIDs));
        else:
            LocalInd = SITES.LocalSite;
        end

        # find local magnetic and electric field channel numbers
        self.local_site = array_header.Sites(LocalInd);
        self.ChIn =[];
        self.ChOut =[];
        HZind =[];
        for i_ch in range(self.local_site.num_channels):
            if isa(self.local_site.Channels(i_ch), 'MagneticChannel'):
                #CAN WE HAVE MORE THAN ONE VERTICAL MAGNETIC LOCALLY
                #WHAT BEHAVIOUR WOULD WE EXPECT FROM TF ESTIMATE IF SO?
                #MAYBE BETTER TO TREAT A DUPLICATE CHANNEL AS A SECOND (
                # possibly degnerate) SITE???
                if self.local_site.Channels(i_ch).vertical:
                    HZind += [i_ch,]
                else:
                    self.input_channels += [i_ch,]
            elif isa(self.local_site.Channels(i_ch), 'ElectricChannel'):
                self.output_channels += [i_ch,]

        if self.num_input_channels != 2:
            print('did not find exactly 2 horizontal magnetic channels for '
                 'local site')
            #?raise Exception?

        if SITES.VTF:
            if isempty(HZind):
                print('no vertical magnetic channel found for local site')
            elif length(HZind) > 1:
                print('more than one vertical magnetic channel found for local '
                  'site')
                #?raise Exception?
            else:
                self.ChOut = HZind + self.output_channels

        if SITES.RR:
            self.processing_scheme = 'RR'
            #find reference site number if a character string is provide
            if ischar(SITES.RemoteSite):
                ReferenceInd = find(strcmp(SITES.RemoteSite, obj.Header.SiteIDs));
            else:
                ReferenceInd = SITES.RemoteSite;
            end
            # extract reference channels --  here we assume these are always
            # magnetic (the normal approach), but this code could easily be
            # modified to allow more general reference channels
            self.RemoteSite = array_header.Sites(ReferenceInd);
            self.ChRef =[];
            for channel in self.RemoteSite.channels:
                if (channel.is_mangetic) & (not channel.is_vertical):
                    self.ChRef += [i_ch,]
            if length(self.ChRef) != 2:
                print('did not find exactly 2 horizontal magnetic channels '
                      'for reference site')
                #?raise Exception?
        else:
            self.processing_scheme = 'SS';
