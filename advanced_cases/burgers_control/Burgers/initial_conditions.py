import numpy as np
import os
import sys
# sys.path.append('/Alexandria/Environments/Burgers/')

def initial_conditions(ic: str, Nt: int, dt: float, Nx: int,
                       FOLDER: str = './Alexandria/Environments/Burgers/', name: str = 'sim_burgers.npz',
                       key: str = 'field', ind_develop: int = 240):
    """
    A script for loading/computing the initial conditions of the current environment.

    :param ic:
    :param Nt:
    :param dt:
    :param Nx:
    :param FOLDER:
    :param name:

    :return: u0, u1, t, ind
    """
    ic = ic.lower()
    FOLDER = os.path.dirname(os.path.abspath(__file__)) + os.sep  # independent of the cwd

    # if not os.path.isfile(FOLDER + name) and ic != 'zero':
    #     raise FileNotFoundError('Initial conditions file not found!')

    if ic == 'zero':
        u0 = np.zeros(Nx)
        u1 = np.zeros(Nx)
        ind0 = 0
    elif ic == 'fully_developed_deterministic':
        ind0 = ind_develop
        ind1 = ind_develop - 1
        u0 = np.load(FOLDER + 'sim_burgers_0.npz')[key][ind0, :]
        u1 = np.load(FOLDER + 'sim_burgers_0.npz')[key][ind1, :]
    elif ic == 'fully_developed_random':
        ind0 = np.random.choice(range(240, 500), 1)
        ind1 = ind0 - 1
        u0 = np.load(FOLDER + 'sim_burgers_0.npz')[key][ind0, :][0]
        u1 = np.load(FOLDER + 'sim_burgers_0.npz')[key][ind1, :][0]
    else:
        raise('Initial condition not recognised. Current options are: zero, '
              'fully developed deterministic or fully developed stochastic. Please, try again.')

    t0 = int(ind0 * dt)
    t = np.linspace(t0, t0 + Nt * dt, Nt + 1)

    return u0, u1, t, ind0



