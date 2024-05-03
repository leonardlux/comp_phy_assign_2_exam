import numpy as np
import matplotlib.pyplot as plt
import simulation as sim_f

plt.style.use("bmh")


def plot_eigenvalues(
        sim,
        filename="images_2/eig_values.pdf",
        analy=True,
        n_max=-1,
):
    # anlytical eigenvalues
    n = np.arange(len(sim.eig_values))[:n_max]
    plt.figure(figsize=(5, 4))
    plt.scatter(n, sim.eig_values[:n_max],
                marker=".", label="numerical eigenvalues")
    if analy:
        plt.plot(n, sim.analy_eig_values[:n_max], "r--",
                 label="analytical eigenvalues")
    plt.xlabel("eigenstate index n")
    plt.ylabel("$\\lambda_n = E_n t_0/\\hbar$")
    # plt.title("numerical and exact energy eigenvalues")
    if analy:
        plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    pass


def plot_probability(
        sim,
        n_vectors=range(4),
        filename="images_2/probability.pdf",
        analy=True,
):
    plt.figure(figsize=(6,5))
    for i in n_vectors:
        # last eigenvector is the one for the trivial solution e = 0
        plt.scatter(sim.x,
                    np.abs(sim.eig_vectors[i]**2),
                    label=f"eigenstate {i}",
                    marker="o",
                    alpha=0.7,
                    )
        if analy:
            plt.plot(sim.x,
                     np.abs(sim.analy_eig_vectors[i]**2),
                     label=f"analy. sol. {i}",
                     linestyle="--",
                     alpha=0.7,
                     )
    plt.xlabel("$x'$")
    plt.ylabel("$<\\Psi_i | \\Psi_j>$")
    # plt.title("some eigenfunctions")
    plt.legend()
    plt.savefig(filename)


def plot_eigenvectors(
        sim,
        n_vectors=range(4),
        filename="images_2/eig_vectors.pdf",
        analy=True,
):
    plt.figure(figsize=(6,5))
    for i in n_vectors:
        # last eigenvector is the one for the trivial solution e = 0
        plt.scatter(sim.x,
                    sim.eig_vectors[i],
                    label=f"eigenstate {i}",
                    marker=".",
                    alpha=0.7,
                    )
        if analy:
            plt.plot(sim.x,
                     sim.analy_eig_vectors[i],
                     label=f"analy. sol. {i}",
                     linestyle="-",
                     alpha=0.7,
                     )
    plt.xlabel("$x'0$")
    plt.ylabel("$\\Psi(x')$")
    # plt.title("some eigenfunctions")
    plt.legend()
    plt.savefig(filename)


def plt_error(sim_n, n_n=10, filename="images_2/eig_vectors_error.pdf"):
    plt.figure(figsize=(6,5))
    for n in range(1, n_n):
        plt.loglog(
            [sim_n[i].dx for i in range(len(sim_n))],
            [sim_n[i].error_values[n] for i in range(len(sim_n))],
            marker="o",
            label=f"eigenfct. n = {n}",
            alpha=0.7,
        )
    plt.xlabel("$\\Delta x$")
    plt.ylabel("$err_n(\\Delta x)$")
    # plt.title("error plot")
    plt.legend()
    plt.savefig(filename)


def plot_check_orthogonality(
        SimResults,
        filename="images_2/orthogonality_check.pdf"
):
    eig_vectors = SimResults.eig_vectors
    dx = SimResults.dx
    values = np.empty((eig_vectors.shape[0], eig_vectors.shape[0]))
    for i, vec_1 in enumerate(eig_vectors):
        for j, vec_2 in enumerate(eig_vectors):
            if i == j:
                values[i][j] = None
                continue
            values[i][j] = sim_f.quant_scalar_prod(dx, vec_1, vec_2)
    plt.figure(figsize=(10, 6))
    plt.xlabel("$\\Psi_i$")
    plt.ylabel("$\\Psi_j$")
    plt.title("$<\\Psi_i | \\Psi_j>$")
    plt.imshow(values)
    plt.colorbar()
    plt.savefig(filename)


def plot_diff_times(
        x,
        time_evolution_fct,
        ts=[0, 1, 5, 100],
        filename="diff_times_test.pdf"
):
    plt.figure(figsize=(6,5))
    for t in ts:
        exspected_v = sim_f.exspected_value(time_evolution_fct(t))
        plt.plot(x, exspected_v,
                 marker=".", label=f"$t/t_0={t:.2f}$", alpha=0.8)
        print(f"normalization at t/t_0={t}: {np.trapz(exspected_v, x):.5f}")
    plt.xlabel("$x'$")
    plt.ylabel("$|\\Psi^2|$")
    plt.legend()
    plt.savefig(filename)


def plot_time_development(
        x,
        time_evolution_fct,
        t_final,
        filename="time_dev_test.pdf"
):
    plt.figure(figsize=(6,5))
    t = np.linspace(0, t_final*5)
    values = []
    for t_i in t:
        values.append(
            sim_f.exspected_value(time_evolution_fct(t_i))
        )

    #values = np.transpose(np.array(values))

    plt.imshow(values, aspect='auto', origin='lower',
               extent=[x[0], x[-1], t[0], t[-1]])
    plt.colorbar(label="$|\\Psi^2|$")
    plt.xlabel("$x'$")
    plt.ylabel("$t'$")
    plt.savefig(filename)


def plot_norm_over_time(
        x,
        time_evolution_fct,
        filename="images_2/norm_over_time.pdf"
):
    ts = np.logspace(-4, 3,)
    plt.figure(figsize=(6,5))
    plt.plot(
        ts,
        [np.trapz(sim_f.exspected_value(time_evolution_fct(t)), x)
         for t in ts],
        marker="o",
    )
    plt.ylabel("$\\sum |\\Psi^2|$")
    plt.xlabel("$t/t_0$")
    plt.xscale("log")
    plt.savefig(filename)


def plot_f_fct(eig_value_max, v_0, filename):
    eig_values = np.linspace(0, eig_value_max, 1000)
    # we make the
    plt.figure(figsize=(6,5))
    plt.plot(eig_values, sim_f.f_fct(eig_values, v_0))
    plt.xlabel("$\\lambda$")
    plt.ylabel("$f(\\lambda)$")
    plt.savefig(filename)


def plot_v_0_f_roots(v_0s,roots,filename):
    plt.figure(figsize=(6,5))
    plt.plot(v_0s,roots,marker="o")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("$\\nu_0$", fontsize=18)
    plt.ylabel("$n_{roots}$ ", fontsize=18)
    plt.savefig(filename)


def plot_euler_forward(x,ev_euler,ts,filename,dt=1e-3,):
    plt.figure(figsize=(6,5))
    for t in ts:
        plt.plot(x,ev_euler[t].real,marker="o",label=f"t'={dt*t:.2f}")
    plt.xlabel("$x'$", fontsize=18)
    plt.ylabel("$Re(\phi_1)_{euler}$ ", fontsize=18)
    plt.legend()
    plt.savefig(filename)

def plot_euler_forward_norm(ev_eulers,labels,dts,ts,filename,):
    norms = np.zeros((len(ev_eulers),len(ts)))
    for i, ev_euler in enumerate(ev_eulers):
        for j,t in enumerate(ts):
            norms[i,j] = np.linalg.norm(ev_euler[t])

    plt.figure(figsize=(6,5))
    for i,norm in enumerate(norms):
        plt.plot(np.array(ts)*dts[i],(norm),label=labels[i],marker="o")
    

    plt.xlabel("$t'$", fontsize=18)
    plt.yscale("log")
    plt.ylabel("$||\phi_{1,euler}||$ ", fontsize=18)
    plt.legend()
    plt.savefig(filename)


def plot_euler_forward_norm_relation(ev_eulers,labels,dts,ts,filename,):
    norms = np.zeros((len(ev_eulers),len(ts)))
    for i, ev_euler in enumerate(ev_eulers):
        for j,t in enumerate(ts):
            norms[i,j] = np.linalg.norm(ev_euler[t])

    plt.figure(figsize=(6,5))
    
    # predicitons 
    a_s = []
    for i,norm in enumerate(norms):
        t = np.array(ts)*dts[i]
        a = (np.log(norm[-1]) - np.log(norm[0]))/ (t[-1]-t[0])
        a_s.append(a)
        
    plt.plot(dts,a_s,marker="o")

    plt.xlabel("$\Delta t'$", fontsize=18)
    plt.ylabel("$a$ ", fontsize=18)
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.savefig(filename)

