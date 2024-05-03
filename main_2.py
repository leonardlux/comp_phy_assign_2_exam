import numpy as np
import simulation as sim
import plot_tools as pltools
# this time a more funcitonal orientated approach


def main():
    # options
    plot = True

    # parameter
    # basic
    L = 1           # length of box, which is in dimensionless coordinates 1
    # for different tasks
    # fist one is for all the picture that actually show a distribution
    n_points = 101
    # this one is for the error values
    n_points_list = [11, 101, 501, 1001, 2001, ]

    # basic simulation
    basic_sim = sim.run_simulation(n_points, L)

    # solution for different n or dx

    sim_n = []
    for i, n in enumerate(n_points_list):
        sim_n.append(sim.run_simulation(n, L))

    # sin intial values
    def psi_0(x): return np.sqrt(2) * np.sin(np.pi * x)
    # lol my linter makes lambda functions to one line definitions xD
    inital_vector_sin = psi_0(basic_sim.x)
    time_evolution_sin = sim.get_expansion_eig_fct(
        inital_vector_sin, basic_sim)

    # delta_peak inital values
    inital_vector_delta = np.zeros(n_points)
    inital_vector_delta[int(n_points/2)] = 1/np.sqrt(basic_sim.dx)
    time_evolution_delta = sim.get_expansion_eig_fct(
        inital_vector_delta, basic_sim)

    # plots

    if plot is True:
        def fn(name):
            plt_dir = "images_2/"
            filetype = ".pdf"
            return plt_dir + name + filetype

        # basic plots
        pltools.plot_eigenvalues(
            basic_sim, filename=fn("eigenvalues"))
        pltools.plot_eigenvectors(
            basic_sim, filename=fn("eigenvectors"))
        # error plot
        pltools.plt_error(sim_n, filename=fn("error"))
        # orthogonality check
        pltools.plot_check_orthogonality(
            basic_sim, filename=fn("orthogonality"))

        # plot time depended functions
        pltools.plot_diff_times(
            basic_sim.x,
            time_evolution_sin,
            filename=fn("time_dep_sin"),
        )
        pltools.plot_norm_over_time(
            basic_sim.x, 
            time_evolution_sin,
            filename=fn("norm_over_time_sin"),
            )
        pltools.plot_diff_times(
            basic_sim.x,
            time_evolution_delta,
            ts=[0.0005, 0.001, 0.005,],
            filename=fn("time_dep_delta"))
        pltools.plot_norm_over_time(
            basic_sim.x, 
            time_evolution_delta,
            filename=fn("norm_over_time_delta"),
            )


if __name__ == "__main__":
    main()
