def make_subtitle(
    rho_rms_aurora,
    rho_rms_emtf,
    phi_rms_aurora,
    phi_rms_emtf,
    matlab_or_fortran,
    ttl_str="",
):
    """

    Parameters
    ----------
    rho_rms_aurora: float
        rho_rms for aurora data differenced against a model. comes from compute_rms
    rho_rms_emtf:
        rho_rms for emtf data differenced against a model. comes from compute_rms
    phi_rms_aurora:
        phi_rms for aurora data differenced against a model. comes from compute_rms
    phi_rms_emtf:
        phi_rms for emtf data differenced against a model. comes from compute_rms
    matlab_or_fortran: str
        "matlab" or "fortran".  A specifer for the version of emtf.
    ttl_str: str
        string onto which we add the subtitle

    Returns
    -------
    ttl_str: str
        Figure title with subtitle

    """
    ttl_str += (
        f"\n rho rms_aurora {rho_rms_aurora:.1f} rms_{matlab_or_fortran}"
        f" {rho_rms_emtf:.1f}"
    )
    ttl_str += (
        f"\n phi rms_aurora {phi_rms_aurora:.1f} rms_{matlab_or_fortran}"
        f" {phi_rms_emtf:.1f}"
    )
    return ttl_str


def make_figure_basename(
    local_station_id: str, remote_station_id: str, xy_or_yx: str, matlab_or_fortran: str
):
    """

    Parameters
    ----------
    local_station_id: str
        station label
    remote_station_id: str
        remote reference station label
    xy_or_yx: str
        mode: "xy" or "yx"
    matlab_or_fortran: str
        "matlab" or "fortran".  A specifer for the version of emtf.

    Returns
    -------
    figure_basename: str
        filename for figure

    """
    station_string = f"{local_station_id}"
    if remote_station_id:
        station_string = f"{station_string}_rr{remote_station_id}"
    figure_basename = f"synthetic_{station_string}_{xy_or_yx}_{matlab_or_fortran}.png"
    return figure_basename


def plot_rho_phi(
    xy_or_yx,
    tf_collection,
    rho_rms_aurora,
    rho_rms_emtf,
    phi_rms_aurora,
    phi_rms_emtf,
    matlab_or_fortran,
    aux_data=None,
    use_subtitle=True,
    show_plot=False,
    output_path=None,
):
    """
    Could be made into a method of TF Collection
    Parameters
    ----------
    xy_or_yx
    tf_collection
    rho_rms_aurora
    rho_rms_emtf
    phi_rms_aurora
    phi_rms_emtf
    matlab_or_fortran
    aux_data
    use_subtitle
    show_plot

    Returns
    -------

    """
    ttl_str = ""
    if use_subtitle:
        ttl_str = make_subtitle(
            rho_rms_aurora,
            rho_rms_emtf,
            phi_rms_aurora,
            phi_rms_emtf,
            matlab_or_fortran,
        )

    figure_basename = make_figure_basename(
        tf_collection.local_station_id,
        tf_collection.remote_station_id,
        xy_or_yx,
        matlab_or_fortran,
    )
    tf_collection.rho_phi_plot(
        xy_or_yx=xy_or_yx,
        aux_data=aux_data,
        ttl_str=ttl_str,
        show=show_plot,
        figure_basename=figure_basename,
        figures_path=output_path,
    )
    return
