import numpy as np
import matplotlib.pyplot as plt
from solver import burger_solver
import os

def main():
    C = 1  # Courant Number
    C2 = C ** 2  # Courant Number^2
    L = 20  # Length of the domain [m]
    T = 5  # Simulation duration [s]
    c = 330  # Wave propagation speed
    Nx = int(1e3)  # Number of space elements
    dx = L / Nx  # Space discretisation step
    # dt = C*dx/c                    # Time discretisation step
    dt = .01
    Nt = int(round(T / dt))  # Number of temporal elements
    x = np.linspace(0, L, Nx + 1)  # Space elements vector
    t = np.linspace(0, Nt * dt, Nt + 1)  # Time elements vector
    u_lim = 20  # Upper limiti for displacement
    cont = 0  # Iterations counter
    reset_cont = 0  # Iteration counter reset
    nu = .9
    # Define the upper and lower action bounds
    xa = 13.2  # Action position (x)
    sigma = 0.2  # Sigma of the Gaussian (variance)
    A_H = np.zeros(Nx + 1)
    # Defining the forcing vector

    frequence = 0.005e2  # Frequency of the forcing vector
    A = 1e2  # Amplitude
    xf = 6.6  # Position of the forcing vector (x)

    # Define the displacement vectors at different time steps
    u = np.zeros(Nx + 1)  # t
    u1 = np.zeros(Nx + 1)  # t-1
    u_store = []
    t_store = []
    for cont in range(5 * Nt):
        u = burger_solver(nu, dt, dx, u1, Nx, A, frequence, t,
                          cont, x, xf, sigma, np.zeros(len(u)))

        u1[:] = u[:]

        if cont > int(Nt / 2):
            u_store.append(u)
            t_store.append(dt * cont)

    np.savez('./burgers_u0.npz', u=u_store, t=t_store)
    print('DONE')

    pass

def make_plots(vec):
    os.makedirs('Fig', exist_ok=True)

    for cont in range(len(vec[:, 0])):
        plt.figure()
        plt.plot(vec[cont, :])
        plt.savefig(f'Fig/{cont}.png')
        plt.close()

    return True

if __name__ == '__main__':
    # main()
    u0 = np.load('sim_burgers_0.npz')['field']
    Ok = make_plots(u0)