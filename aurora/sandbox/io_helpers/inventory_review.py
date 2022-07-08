def scan_inventory_for_nonconformity(inventory, verbose=False):
    """
    One off method for dealing with issues of historical data.
    Checks for the following:
    1. Channel Codes: Q2, Q3 --> Q1, Q2
    2. Field-type code: "T" instead of "F"
    3. Tesla to nT
    Parameters
    ----------
    inventory : obspy.core.inventory.inventory.Inventory
        Object containing metadata about station and channels

    Returns
    -------
    inventory : obspy.core.inventory.inventory.Inventory
        Object containing metadata about station and channels
        Might be modified during this function
    """
    networks = inventory.networks
    for network in networks:
        for station in network:
            channel_codes = [x.code[1:3] for x in station.channels]
            if verbose:
                print(channel_codes)

            # Electric channel remap {Q2, Q3}-->{Q1, Q2}>
            if ("Q2" in channel_codes) & ("Q3" in channel_codes):
                if verbose:
                    print(
                        "Detected a likely non-FDSN conformant convnetion "
                        "unless there is a vertical electric dipole"
                    )
                    print("Fixing Electric channel codes")
                # run the loop twice so don't accidentally map Q3 to Q2 and Q2 to Q3
                for channel in station.channels:
                    if channel.code[1:3] == "Q2":
                        channel._code = f"{channel.code[0]}Q1"
                for channel in station.channels:
                    if channel.code[1:3] == "Q3":
                        channel._code = f"{channel.code[0]}Q2"
                # print("Applied unstable fix to electric channel names")
                print("{Q2, Q3} --> {Q1, Q2}")

            # Magnetic channle remap {T1,T2,T3}-->{F1, F2, F3}
            cond1 = "T1" in channel_codes
            cond2 = "T2" in channel_codes
            cond3 = "T3" in channel_codes
            if cond1 or cond2 or cond3:
                if verbose:
                    print("Detected a likely non-FDSN conformant convention ")
                    # unless there is Tidal data (channel code T)
                    print("Fixing Magnetic channel codes")
                for channel in station.channels:
                    if channel.code[1] == "T":
                        channel._code = f"{channel.code[0]}F{channel.code[2]}"
                # print("Applied unstable fix to magnetic channel names")
                print("{T1,T2,T3} --> {F1, F2, F3}")

            # Tesla to nanoTesla
            for channel in station:
                response = channel.response
                for stage in response.response_stages:
                    if verbose:
                        msg = f"{channel.code} {stage.stage_sequence_number}"
                        msg = f"{msg} {stage.input_units}"
                        print(msg)
                    if stage.input_units == "T":
                        stage.input_units = "nT"
                        stage.stage_gain *= 1e-9
    return inventory
