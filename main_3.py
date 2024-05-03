import numpy as np
import simulation as sim
import plot_tools as pltools
from collections import namedtuple
# this time a more funcitonal orientated approach

calc_all = True

def fn(name):
    plt_dir = "images_3/"
    filetype = ".pdf"
    return plt_dir + name + filetype


def plot_some(sim, name_add):
    # basic plots
    pltools.plot_eigenvalues(
        sim,
        filename=fn(f"eigenvalues_{name_add}"),
        analy=False,
        n_max=20,
    )
    pltools.plot_eigenvectors(
        sim, filename=fn(f"eigenvectors_{name_add}"),
        analy=False)
    pltools.plot_probability(
        sim, filename=fn(f"prob_{name_add}"),
        analy=False
    )
    # error plot
    # orthogonality check
    pltools.plot_check_orthogonality(
        sim, filename=fn(f"orthogonality_{name_add}"))


def factory_v(v_values, x_boundaries):
    v_0_l, v_0_m, v_0_r = v_values
    x_l, x_r = x_boundaries

    def v_fct(x):
        conditions = [
            x < x_l,
            (x >= x_l) & (x <= x_r),
            x > x_r,
        ]
        v = np.select(conditions, v_values, np.nan)
        return v
    return v_fct


def main():

    SimConfig = namedtuple("Config",
                           [
                               "L",
                               "n_points",
                               "potential_fct",
                           ]
                           )

    # parameter
    # basic
    L = 1           # length of box
    basic_boundaries = (1/3 * L, 2/3 * L)

    n_points = 101 # for testing
    #n_points = 501

    # generate potential fct

    # Setup Configs

    # Trivial Simulation Config
    trivial_v_values = (0, 0, 0)
    trival_v_fct = factory_v(trivial_v_values, basic_boundaries)
    trival_config = SimConfig(
        L=L,
        n_points=n_points,
        potential_fct=trival_v_fct,
    )
    # simulation
    trival_sim = sim.run_simulation_config(trival_config)
    plot_some(trival_sim, "trivial")

    # Basic Simulation Config
    basic_v_values = (0, 1e3, 0)
    basic_v_fct = factory_v(basic_v_values, basic_boundaries)
    basic_config = SimConfig(
        L=L,
        n_points=n_points,
        potential_fct=basic_v_fct,
    )

    # basic simulation
    basic_sim = sim.run_simulation_config(basic_config)
    plot_some(basic_sim, "basic")

    # Now mix up the first two eigenfunction to a new wavevector

    inital_vec = 1/np.sqrt(2) * (
        basic_sim.eig_vectors[1] +
        basic_sim.eig_vectors[2]
    )
    t_interest = np.pi / (
        basic_sim.eig_values[2] -
        basic_sim.eig_values[1]
    )
    time_evolution = sim.get_expansion_eig_fct(inital_vec, basic_sim)

    pltools.plot_time_development(
        basic_sim.x,
        time_evolution,
        t_interest,
        filename=fn("super_pos_time_dev"),
    )

    pltools.plot_diff_times(
        basic_sim.x,
        time_evolution,
        [0,t_interest],
        filename=fn("super_pos_discrete_time"),
    )

    pltools.plot_f_fct(
        eig_value_max=1e3,
        v_0=basic_v_values[1],
        filename=fn("f_fct_test")
    )

    if False or calc_all:
        # root finder is broken

        v_0s = [10,20,24,25,30,40]#,1e1,1e2,1e3,1e4,1e5]
        roots = []
        for v_0 in v_0s:
            roots.append(sim.root_finder(v_0,delta_e=1,plot=True))
        for i in range(len(roots)):
            print(f"{v_0s[i]}, {roots[i]}")
            
    
    if False or calc_all:
        # therefore we use a scipy root finder
        # conditional because this is computing intensive

        v_0s = [10,20,25,1e2,1e3,1e4,1e5]
        guess_amount_of_roots = [2,2,2, 3, 7, 23, 69]
        roots = []
        print("Scipty version follows")
        print("v_0, roots")
        for v_0, n_guess in zip(v_0s,guess_amount_of_roots):
            root_v_values = (0, v_0, 0)
            root_v_fct = factory_v(root_v_values, basic_boundaries)
            root_config = SimConfig(
                L=L,
                n_points=n_points,
                potential_fct=root_v_fct,
            )

            # basic simulation
            root_sim = sim.run_simulation_config(root_config)
            print("First two eigenvalues for a bound state")
            print(root_sim.eig_values[:2])
            roots_i = sim.root_finder_scipy(v_0, root_sim.eig_values[:n_guess])
            roots.append(roots_i.x.shape)
            print(f"v_0: {v_0}, results from scipy roots \n{roots_i}")
        
        print("\nResults Scipy root finder")
        for i in range(len(roots)):
            print(f"{v_0s[i]}, {roots[i]}")

        print("Scipy also finds the root at zero, therefore we need to subtract one allowed root")
    
    # plot roots against v_0

    if False or calc_all:
        v_0s            = [10,20,25,1e2,1e3,1e4,1e5]
        amount_of_roots = [0, 0, 1,  2,  6, 22, 68]
        pltools.plot_v_0_f_roots(v_0s, amount_of_roots,fn("v_0_roots_f"))


        # Euler forwards for task 3.7
        # simple case
        ts = [10,200,400,600,1000,1500, 2000]
        ev_euler = sim.euler_forward(basic_sim,dt=1e-6,n_t=1e5)
        pltools.plot_euler_forward(basic_sim.x,ev_euler,ts,fn("euler_forward"),dt=1e-3,)

        dts = [1e-3,1e-4,1e-5,1e-6,1e-7][:]
        names = ["3","4","5","6","7"][:]
        labels = [f"$\\Delta t = 10^{name}$" for name in names]
        ev_eulers = []
        for dt,name in zip(dts,names):
            ev_euler = sim.euler_forward(basic_sim,dt=dt,n_t=1e5)
            pltools.plot_euler_forward(basic_sim.x,ev_euler,ts,fn(f"euler_forward_{name}"),dt=dt)
            ev_eulers.append(ev_euler)

        # norm over time would be quit nice but not as usefule and harder to implement correctly
        pltools.plot_euler_forward_norm(ev_eulers,labels,ts=ts,filename=fn("euler_forward_norm"),dts=dts)

        pltools.plot_euler_forward_norm_relation(ev_eulers,labels,ts=ts,filename=fn("euler_forward_norm_relation"),dts=dts)



    ts = [10,200,400,600,1000,1500, 2000]
    dt = 1 
    ev_cn = sim.crank_nicholson(basic_sim,dt=dt,n_t=2e3+1)
    pltools.plot_euler_forward(basic_sim.x,ev_cn,ts,fn("crank_nicholson"),dt=dt,)

 


    # how to test for CFL 

if __name__ == "__main__":
    main()
