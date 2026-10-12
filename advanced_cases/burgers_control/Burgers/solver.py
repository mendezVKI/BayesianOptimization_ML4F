import numpy as np
import numpy.linalg
from scipy.linalg import solve_banded


def burger_solver(nu: float, dt: float, dx: float, u1: np.array, Nx: int, A: float, frequency: float, t: np.array,
                  cont: int, x: np.array, xf: float, sigma: float, action_vec: np.array, forecasting: bool = False):
    """
    An implicit solver of the Burgers equation, implementing homogenous Neumann boundary conditions. For the derivation
    see:
    http://hmf.enseeiht.fr/travaux/CD0001/travaux/optmfn/hi/01pa/hyb41/reportbid.htm
    --------------------------------------------------------------------------------------------------------------------
    Params:
    ----------
    :param nu: np.float,
            Viscosity
    :param dt: np.float,
            Temporal step size
    :param dx: np.float,
            Spatial step size
    :param u1: np.array,
            Displacement field at the previous time step
    :param Nx: np.int,
            Number of spatial points
    :param A: np.float,
            Amplitude of the perturbation
    :param frequency: np.float,
            Frequency of the perturbtion
    :param t: np.array,
            Time array
    :param cont: int,
            Actual step. cont/Nt
    :param x: np.array,
            Spatial axes
    :param xf:  np.float,
            Position of the perturbance
    :param sigma: np.float,
            Standard deviation of both the control action and the perturbance
    :param action_vec: np.array,
            An array containing the action taken
    :param forecasting: bool,
            If the solver is called for computing the impulsive reward,
            forecasting == True. In this case, the action is kept
            only for the first bunch of episodes, and then deleted,
            in order to assess the effect of the action a_t only
            over the displacement field.
    :return: u, flag np.array, bool
            u is the new array containing the displacement field,
            flag checks that the solution of the temporal step went well and
            without errors.

    """

    if forecasting:
        '''
        If doing forecasting, then the action is switched off after the first time step. 
        We only wanted to see the effect of the current action!
        '''
        if dt * cont > t[0]:
            action_vec = np.zeros(len(u1))

    s = (nu * dt) / (dx ** 2)
    a = -(dt / (4 * dx)) * u1[:-2] - s / 2
    a = np.concatenate((a, - (dt / (4 * dx)) * u1[-2] - 0.5 * s), axis=None)
    b = (1 + s) * np.ones(Nx - 1)
    b = np.concatenate((- (dt / (4 * dx)) * u1[0] + 0.5 * s + 1, b), axis=None)
    b = np.concatenate((b, (dt / (4 * dx)) * u1[-1] + 0.5 * s + 1), axis=None)
    c = (dt / (4 * dx)) * u1[2:] - s / 2
    c = np.concatenate(((dt / (4 * dx)) * u1[1] - s / 2, c), axis=None)
    d = 0.5 * s * u1[:-2] + (1 - s) * u1[1:-1] + (s / 2) * u1[2:]
    d = np.concatenate((d, 0.5 * s * u1[-2] + u1[-1] * ((1 - s) + 0.5 * s)), axis=None)
    d = np.concatenate((u1[0] * ((1 - s) + 0.5 * s) + 0.5 * s * u1[1], d), axis=None)
    # RHS terms
    forcing = A * (np.sin(2 * np.pi * frequency * dt * cont))
    fun = forcing * np.exp(-((x - xf) ** 2) / (2 * sigma ** 2))
    rhs = d + dt * (fun + np.asarray(action_vec))
    # Solving for timestep
    Ab = np.zeros((3, len(b)))
    Ab[0, 1:] = c
    Ab[1, :] = b
    Ab[2, :-1] = a
    try:
        u = solve_banded((1, 1), Ab, rhs)
        flag = False
    except (ValueError, numpy.linalg.LinAlgError):
        flag = True
        u = u1

    return u, flag , fun # Miguel adds the fun for plotting purposes
