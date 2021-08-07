def plot_tf_obj(tf_obj, out_filename=None):
    print("GET PLOTTER FROM MTpy")
    from aurora.transfer_function.plot.rho_plot import RhoPlot
    import matplotlib.pyplot as plt
    plotter = RhoPlot(tf_obj)
    fig, axs = plt.subplots(nrows=2)
    ttl_str = tf_obj.tf_header.local_station_id
    plotter.rho_sub_plot(axs[0], ttl_str=ttl_str)
    plotter.phase_sub_plot(axs[1], ttl_str=ttl_str)
    if out_filename:
        plt.savefig(f"{out_filename}.png")
    plt.show()