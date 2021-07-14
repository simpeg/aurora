def plot_tf_obj(tf_obj):
    print("GET PLOTTER FROM MTpy")
    from aurora.transfer_function.rho_plot import RhoPlot
    import matplotlib.pyplot as plt
    plotter = RhoPlot(tf_obj)
    fig, axs = plt.subplots(nrows=2)
    plotter.rho_sub_plot(axs[0])
    plotter.phase_sub_plot(axs[1])
    plt.show()