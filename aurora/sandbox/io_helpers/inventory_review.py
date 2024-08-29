"""
    In this module are some helper functions for working with FDSN inventory objects.
"""

from loguru import logger
from typing import Optional
import obspy


def describe_inventory_stages(
    inventory: obspy.Inventory,
    assign_names: Optional[bool] = False,
    verbose: Optional[bool] = False,
) -> None:

    """
    Scans inventory looking for stages.  Has option to assign names to stages,
    these names are used as keys in MTH5. Modifies inventory in place.

    Parameters
    ----------
    inventory: obspy.Inventory
        obspy representation of station XML
    assign_names: bool
        If True, inventory stage names will be assigned
    verbose: bool
        If True, some info logs are created.

    """
    new_names_were_assigned = False
    networks = inventory.networks
    for network in networks:
        for station in network:
            for channel in station:
                response = channel.response
                stages = response.response_stages
                if verbose:
                    info = (
                        f"{network.code}-{station.code}-{channel.code}"
                        f" {len(stages)}-stage response"
                    )
                    logger.info(info)

                for i, stage in enumerate(stages):
                    if verbose:
                        logger.info(f"stagename {stage.name}")
                    if stage.name is None:
                        if assign_names:
                            new_names_were_assigned = True
                            new_name = f"{station.code}_{channel.code}_{i}"
                            stage.name = new_name
                            if verbose:
                                logger.info(
                                    f"ASSIGNING stage {stage}, name {stage.name}"
                                )
                    if hasattr(stage, "symmetry"):
                        pass
                        # import matplotlib.pyplot as plt
                        # print(f"symmetry: {stage.symmetry}")
                        # plt.figure()
                        # plt.clf()
                        # plt.plot(stage.coefficients)
                        # plt.ylabel("Filter Amplitude")
                        # plt.xlabel("Filter 'Tap'")
                        # plt.title(f"{stage.name}; symmetry: {stage.symmetry}")
                        # plt.savefig(f"{stage.name}.png"))
                        # plt.show()
    if new_names_were_assigned:
        inventory.networks = networks
        logger.info("Inventory Networks Reassigned")
    return


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
                logger.info(channel_codes)

            # Electric channel remap {Q2, Q3}-->{Q1, Q2}>
            if ("Q2" in channel_codes) & ("Q3" in channel_codes):
                if verbose:
                    logger.info(
                        "Detected a likely non-FDSN conformant convnetion "
                        "unless there is a vertical electric dipole"
                    )
                    logger.info("Fixing Electric channel codes")
                # run the loop twice so don't accidentally map Q3 to Q2 and Q2 to Q3
                for channel in station.channels:
                    if channel.code[1:3] == "Q2":
                        channel._code = f"{channel.code[0]}Q1"
                for channel in station.channels:
                    if channel.code[1:3] == "Q3":
                        channel._code = f"{channel.code[0]}Q2"
                # print("Applied unstable fix to electric channel names")
                logger.info("{Q2, Q3} --> {Q1, Q2}")

            # Magnetic channle remap {T1,T2,T3}-->{F1, F2, F3}
            cond1 = "T1" in channel_codes
            cond2 = "T2" in channel_codes
            cond3 = "T3" in channel_codes
            if cond1 or cond2 or cond3:
                if verbose:
                    logger.info("Detected a likely non-FDSN conformant convention ")
                    # unless there is Tidal data (channel code T)
                    logger.info("Fixing Magnetic channel codes")
                for channel in station.channels:
                    if channel.code[1] == "T":
                        channel._code = f"{channel.code[0]}F{channel.code[2]}"
                # print("Applied unstable fix to magnetic channel names")
                logger.info("{T1,T2,T3} --> {F1, F2, F3}")

            # Below represents correct syntax for a valid, but not conventional way to correct units
            # leaving the code (commented) for future reference
            # # Tesla to nanoTesla
            # for channel in station:
            #     response = channel.response
            #     for stage in response.response_stages:
            #         if verbose:
            #             msg = f"{channel.code} {stage.stage_sequence_number}"
            #             msg = f"{msg} {stage.input_units}"
            #             logger.info(msg)
            #         if stage.input_units == "T":
            #             stage.input_units = "nT"
            #             stage.stage_gain *= 1e-9
    return inventory
