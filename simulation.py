import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple
import scipy

from scipy.optimize import root


def discrete_x(n_points, L=1, ):
    x, dx = np.linspace(0, L, n_points, retstep=True)
    return x, dx


def generate_matrix(dx, n_points, *args):
    # generate the matrix, which solves the hamilitonian
    main_diag = 2/dx**2 * np.ones(n_points)
    side_diag = -1/dx**2 * np.ones(n_points-1)
    sub_diag = side_diag.copy()
    sup_diag = side_diag.copy()
    matrix = np.diag(main_diag) \
        + np.diag(sup_diag, k=1) \
        + np.diag(sub_diag, k=-1)
    return matrix


def generate_matrix_pot(dx, n_points, v_fct):
    # v_fct, should depend on x'
    # generate the matrix, which solves the hamilitonian
    main_diag = 2/dx**2 * np.ones(n_points) + v_fct(np.arange(n_points) * dx)
    side_diag = -1/dx**2 * np.ones(n_points-1)
    sub_diag = side_diag.copy()
    sup_diag = side_diag.copy()

    matrix = np.diag(main_diag) \
        + np.diag(sup_diag, k=1) \
        + np.diag(sub_diag, k=-1)
    return matrix


def find_eigenvalues_and_states(matrix):
    main = np.diag(matrix, k=0)
    side = np.diag(matrix, k=1)
    eig_values, eig_vectors = scipy.linalg.eigh_tridiagonal(main, side)
    # this function returns a weird layout: eig_vectors[i][n]
    # we want eig_vectors[n][i], therefore we transpose
    eig_vectors = eig_vectors.transpose()
    # Add trivial eigen-value and -vector
    # to make it comparible to analytical solutions
    eig_values = np.insert(eig_values[:-1], 0, 0)
    eig_vectors = np.insert(
        eig_vectors[:-1], 0, np.zeros(eig_vectors.shape[1]), axis=0)
    return eig_values, eig_vectors


def normalize_eig_vectors(x, eig_vectors):
    norm = []
    n = eig_vectors.shape[0]
    for i in range(n):
        norm.append(np.sqrt(np.trapz(np.abs(eig_vectors[i])**2, x)))

    n_eig_vectors = np.copy(eig_vectors)
    # normalization of eigenvectors
    for i in range(1, n):
        n_eig_vectors[i] = eig_vectors[i] / \
            norm[i]  # ignoring the trival vector
        # trival vector can not be normalized
    return n_eig_vectors


def analytical_eig_values(n_points):
    analy_eig_values = (np.arange(n_points) * np.pi)**2
    return analy_eig_values


def analytical_eig_vector(x, n_points):
    analy_eig_vectors = [np.sqrt(2) * np.sin(n * np.pi * x)
                         for n in range(n_points)]
    return analy_eig_vectors


def calc_error_values(analy_eig_vector, eig_vector):
    error_values = np.sum(
        (analy_eig_vector - eig_vector) ** 2,
        axis=1
    )/eig_vector.shape[1]
    return error_values


def quant_scalar_prod_own(dx, vector_1, vector_2):
    # in case we need to define our own ... not that hard
    result = np.sum(dx * vector_1 * vector_2)
    return result


def quant_scalar_prod(dx, vec_1, vec_2):
    result = np.trapz(vec_1 * vec_2, dx=dx)
    return result


def get_alpha_values(dx, eig_vectors, inital_vector, ):
    alphas = []
    # we dont need to use the complex conjugate, because all is real
    for n in range(len(eig_vectors)):
        alphas.append(np.trapz(
            eig_vectors[n] * inital_vector,
            dx=dx
        ))
    return alphas


def generate_time_evolution_fct(eig_vectors, eig_values, alphas):
    def time_evolution(t):
        values_n = np.empty(len(eig_vectors), np.ndarray)
        for n in range(len(eig_values)):
            values_n[n] = alphas[n] * \
                np.exp(1j * eig_values[n] * t) * eig_vectors[n]
        return np.sum(values_n, axis=0)
    return time_evolution


def exspected_value(x):
    return np.absolute(x)**2


def f_fct(eig_values, v_0):
    k = np.sqrt(eig_values)
    kappa = np.sqrt(v_0 - eig_values)
    term1 = np.exp(kappa / 3) * (kappa * np.sin(k / 3) + k * np.cos(k / 3))**2
    term2 = np.exp(-kappa / 3) * (kappa * np.sin(k / 3) - k * np.cos(k / 3))**2
    return term1 - term2


def root_finder(v_0, delta_e, plot=True, n_min_res=1e3, num_tolerance=1e-3):
    roots = []
    # we use this function for 0 < e < v_0
    # one root can only be located between to extrema
    # here we choose a step size of delta_e
    # all roots that are only distanced in the magnitude of delta_e are
    # hard to resolve
    # 1. Split the whole intervall
    e_values = np.linspace(0, v_0, int(v_0/delta_e))
    # 2. differentiate the function numerically
    values = f_fct(e_values, v_0)
    diffs = values[1:] - values[:-1]
    # 3. find switch of signs of the diffs
    sign_switch = diffs[:-1] * diffs[1:]
    extrema = np.where(sign_switch < 0)[0]
    # check if code is doing the right thing
    if plot is True:
        def plot_test(poi, name):
            plt.figure(figsize=(10, 6))
            plt.plot(e_values, f_fct(e_values, v_0))
            plt.yscale("log")
            for e_poi in e_values[poi]:
                plt.axvline(e_poi, c="r", alpha=0.5)
            plt.xlabel("$\\lambda$")
            plt.ylabel("$f(\\lambda)$")
            plt.savefig(f"images_3/test_{name}.pdf")
        plot_test(extrema, "extrema")

    # find minima (diff sign switch -1 -> 1)
    minima = []
    for ext in extrema:
        if diffs[ext] < 0:
            minima.append(ext + 1)
            # we use ext +1 to get the index for the smallest value
    n_minima = len(minima)

    if plot:
        plot_test(minima, "minima" + str(int(v_0)))

    # we know that the ture mima lays between the index we found,
    # and ond or the neighbouring values
    # for minima_i in minima:
    min_values = []
    n_roots = 0
    for min_i in minima:
        e_min = np.linspace(e_values[min_i-2],
                            e_values[(min_i+2)%len(e_values)], int(n_min_res))
        f_values = f_fct(e_min, v_0)
        min_value = np.min(f_values)
        min_values.append(min_value)
        if np.abs(min_value) < num_tolerance:
            n_roots += 1
        elif min_value < -num_tolerance:
            # we just assume that the highpoints are above zero
            n_roots += 2
        else:
            None

    # now find out how many root
    print("Results Root finder")
    print("v_0: ", int(v_0))
    print("Minima: ", n_minima)
    print("Roots: ", n_roots)
    return n_roots

def root_finder_scipy(v_0,e_guesses):

    # set potential to a constant in f 
    def f(x):
        return f_fct(x,v_0)
    # calculate guesses using the matrix
    
    roots = root(f,e_guesses)

    return roots

# Euler forward

def euler_forward(sim,dt=1e-3, n_t = 1e4):
    # euler forward for the first eigenvector
    # (first non triviel eig_vector)
    # we are only interested in showing the evolution of the real part
    n_t = int(n_t)
    ev_euler = np.zeros((n_t,len( sim.eig_vectors[1])),dtype=np.complex128)
    ev_euler[0] = sim.eig_vectors[1]
    for i in range(1, n_t):
         ev_euler[i] = (1-1j*dt*sim.eig_values[1]) * ev_euler[i-1] 
    return ev_euler


# collections


def calculate_numerically(x, dx, n_points,
                          potential_fct=None):
    if potential_fct is None:
        matrix = generate_matrix(dx, n_points)
    else:
        matrix = generate_matrix_pot(dx, n_points, potential_fct)
    eig_values, eig_vectors = find_eigenvalues_and_states(matrix)
    eig_vectors = normalize_eig_vectors(x, eig_vectors)
    return eig_values, eig_vectors


def calculate_analytically(x, n_points):
    analy_eig_values = analytical_eig_values(n_points)
    analy_eig_vectors = analytical_eig_vector(x, n_points)
    return analy_eig_values, analy_eig_vectors


def run_simulation(n_points, L=1):
    SimResults = namedtuple("Simulation",
                            [
                                "x",
                                "dx",
                                "eig_values",
                                "eig_vectors",
                                "analy_eig_values",
                                "analy_eig_vectors",
                                "error_values",
                            ]
                            )
    x, dx = discrete_x(n_points)
    eig_values, eig_vectors = calculate_numerically(x, dx, n_points)
    analy_eig_values, analy_eig_vectors = calculate_analytically(x, n_points)
    error_values = calc_error_values(analy_eig_vectors, eig_vectors)
    return SimResults(
        x,
        dx,
        eig_values,
        eig_vectors,
        analy_eig_values,
        analy_eig_vectors,
        error_values,
    )


def run_simulation_config(config):
    SimResults = namedtuple("Simulation",
                            [
                                "x",
                                "dx",
                                "eig_values",
                                "eig_vectors",
                                "analy_eig_values",
                                "analy_eig_vectors",
                                "error_values",
                            ]
                            )
    x, dx = discrete_x(config.n_points)
    eig_values, eig_vectors = calculate_numerically(
        x,
        dx,
        config.n_points,
        config.potential_fct,
    )
    analy_eig_values, analy_eig_vectors = calculate_analytically(
        x, config.n_points)
    error_values = calc_error_values(analy_eig_vectors, eig_vectors)
    return SimResults(
        x,
        dx,
        eig_values,
        eig_vectors,
        analy_eig_values,
        analy_eig_vectors,
        error_values,
    )


def get_expansion_eig_fct(inital_vector, SimResults):
    alphas = get_alpha_values(
        SimResults.dx,
        SimResults.eig_vectors,
        inital_vector,
    )
    time_evolution = generate_time_evolution_fct(
        SimResults.eig_vectors,
        SimResults.eig_values,
        alphas
    )
    return time_evolution

# plots
