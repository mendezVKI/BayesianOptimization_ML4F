import math
import os
import shutil
from collections import deque
from Burgers.solver import burger_solver
import matplotlib.pyplot as plt
import numpy as np
#import seaborn as sns
try:                                  # gymnasium is the maintained fork of gym
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:                   # fall back to the old gym if gymnasium is not installed
    import gym
    from gym import spaces
#from scipy.linalg import solve_banded
#import pickle
from Burgers.initial_conditions import initial_conditions
import imageio
# plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
# plt.rc('text', usetex=True)
# sns.set_theme(style="white")
    
pi = math.pi
    
# TODO: Make the initial condition as input (eg. if wanted, start from zero instead of loading the developed wave)
    
# Needed to debugg the code


# Debugging helper
# # Note: for debuggin purposes use: class self: pass  



class Burgers_training(gym.Env):
    """
    Burgers Environment
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self, name: str, action_min=-300, action_max=300, n_actions=1,
                 n_states=3,reward_shape: str = '2-norm',
                 lower_bound_security: bool = False,
                 GAMMA: float = 0.0,
                 FOLDER: str = './',
                 ic: str = 'fully_developed_random',
                 full_log: bool = True, T: float = 15,
                 dt: float = 0.01,
                 nu: float = 0.9, xa: float = 13.2, A: float = 1e2,
                 frequency: float = 0.5, minmax: str ='min', verbose: bool = False):
        """
        Environment for the control of the Burgers equation.
        -------------------------------------------------------------------------
        Params:
        -------
        :param name: str,
                name of the simulation, it will be used to create the full log
                of the training phase.
        :param action_min: np.int,
                Lower bound of the action space
        :param action_max: np.int,
                Upper bound of the action space
        :param n_actions: np.int,
                Number of actions requested to the agent
        :param n_states: np.int,
                Number of observation points
        :param reward_shape: str,
                Defines the reward to be computed during training. Options are:
                - '2-norm': classic reward computing the euclidean norm of the displacement field
                - 'impulsive': new reward recently implemented, in which the reward for the current
                action is obtained 'looking into the future', waiting for the current control
                to propagate.
        :param lower_bound_security:
        :param GAMMA: np.float,
                This coefficient penalizes the action, inside the reward. I.e.
                r_t = J_a + \\gamma * action ** 2
                Doing so, large actions should be avoided, in favour of more contained ones.
        :param FOLDER: str,
                Folder in which all the logs will be saved. './' by default.
        :param ic: str,
                Different initial conditions are available. Current options are:
                - 'zero': the wave starts from 0 (not suggested for learning)
                - 'fully_developed_deterministic': in this case the environment is initialized with
                    the fully developed wave. 'Deterministic' means that the time snapshot loaded is
                    always the same (index 240 of the .npz file here attached), on the other hand
                - 'fully_developed_random': randomly picks an temporal snapshot where the wave
                    is fully developed. Preferred option, starting from different points help
                    the agent to generalize?
        :param full_log: bool,
                If True, all the interactions with the environment are saved on disk,
                in a dictionary that contains:
                - 'actions'
                - 'observations'
                - 'rewards'
                - 'ep_rewards'
                To open it, use the procedure for opening a pickle, i.e. with open()......
        :param T: np.float,
                Temporal length of the episode
        :param dt: np.float,
                Discretization of the temporal axes
        :param nu: np.float,
                Viscosity, this coefficient multiplies the second derivative at RHS.
        :param xa: np.float,
                Point in space where to place the control action
        :param A: np.float,
                Amplitude of the perturbation
        :param frequency: np.float,
                Frequency of the perturbation
        """
        
        # --- Computational domain parameters
        self.L = 20      # Length of the domain [m]
        self.T = T               # Simulation duration [s]
        self.Nx = int(1e3)         # Number of space elements
        self.dx = self.L / self.Nx      # Space discretisation step
        self.dt = dt
        self.Nt = int(round(self.T / self.dt)) # Number of temporal elements
        self.x = np.linspace(0, self.L, self.Nx + 1)   # Space elements vector
        self.u_lim = 15               # Upper limiti for displacement
        self.xa = xa                   # Action position (x)
        self.sigma = 0.2             # Sigma of the Gaussian (variance)
        self.A_H = np.zeros(self.Nx + 1)   # Defining the forcing vector
        self.lower_bound_security = lower_bound_security
        self.frequence = frequency      # Frequency of the forcing vector
        self.A = A                # Amplitude
        self.xf = 6.6             # Position of the forcing vector (x)
        self.nu = nu
        # --- Cost function settings: J = J_s + gamma*J_actuation
        self.reward_shape = reward_shape
        self.GAMMA = GAMMA
        self.minmax = minmax
        self.verbose = verbose
        # --- Logging stuff
        self.FOLDER = FOLDER
        self.full_log = full_log
        # if self.full_log:
        keys = ['ep_rewards', 'actions', 'observations', 'J_s', 'J_a']
        self.episodes_logbook = {chapter: [] for chapter in keys}
        self.current_rewards = []
        self.current_actions = []
        self.current_observations = []
        self.j_s = []
        self.j_a = []
        
        # Miguel--------------------------------------------------
        # Add the perturbation/action for plotting purposes
        self.pert=np.zeros(self.Nx+1) 
        self.action_vec=np.zeros(self.Nx+1) 
            
        # --- Loading the initial condition
        self.ic = ic
        self.u, self.u1, self.t, self.cont = initial_conditions(ic=self.ic, dt=self.dt, Nt=self.Nt, Nx=self.Nx, FOLDER='./')
        self.cont0 = np.copy(self.cont)
        # --- Environment settings
        self.reset_cont = 0  # Iteration counter reset
        self.action_replay = deque(maxlen=4)  # Action dynamic vector
        self.n_states = n_states
        
        # Define the upper and lower action bounds
        self.action_max = action_max
        self.name_sim = name
        self.action_min = action_min
        # Action space definition
        self.action_space = spaces.Box(low=action_min, high=action_max, shape=(1,), dtype=np.float32)
        
        self.tot_reward = 0
        
        # Observation space definition
        self.observation_space = spaces.Box(low=-self.u_lim, high=self.u_lim, shape=(n_states,), dtype=np.float32)
        
    def step(self, action, log_full_state=False):
        
        self.action_replay.append(action)  # adding scaled action to the dynamic vector
        action_vec = [0] * len(self.u)
        action_vec[:] = action* self.action_max * np.exp(-((self.x - self.xa) ** 2) / (2 * self.sigma ** 2))
        
        # Miguel: I add the action for plotting purposes
        self.action_vec=action_vec        
        
        # --- Calling the implicit solver for the next_state
        self.u, flag, self.pert = burger_solver(self.nu, self.dt, self.dx, self.u1, self.Nx, self.A, self.frequence, self.t,
                                     self.cont, self.x, self.xf,self.sigma, action_vec)
        
        if flag:
            '''
            Check that the solver didn't return any error, else raise error and stops the episode, giving a low reward. 
            '''
            # print('Crashed for NaNs!')            
            if self.minmax == 'min':
                reward = 1e5
            else:
                reward = -1e5
            
            done = True
            # observation = np.array([self.u1[770], self.u1[800], self.u1[830]])
            observation = np.array([self.u1[400], self.u1[450], self.u1[500]])
            info = 0
            return observation, reward, done, info
        
        # --- Computing the reward
        if self.reward_shape == 'impulsive' and not flag:
            t_future = np.linspace(self.cont * self.dt, (self.cont + 300) * self.dt, 300)
            
            for forecast_step in t_future:
                u_forecast, _ = burger_solver(self.nu, self.dt, self.dx, self.u1, self.Nx, self.A, self.frequence, t_future,
                                           forecast_step, self.x, self.xf, self.sigma, action_vec, forecasting=True)
                
            snapshot = - abs(u_forecast[770]) - abs(u_forecast[790]) - abs(u_forecast[810])
            future_norm = - np.linalg.norm(u_forecast[770:820])
            reward = snapshot + future_norm - self.GAMMA * action ** 2
            
            # --- For logging purposes
            if self.full_log:
                rew_l2 = - (np.linalg.norm(self.u[770:820]) + (self.GAMMA * action ** 2))
                
        elif self.reward_shape == '2-norm' and not flag:
            J_s = np.linalg.norm(self.u[770:820])
            J_a = (self.GAMMA * (action) ** 2)
            self.j_a.append(J_a)
            self.j_s.append(J_s)
            reward = - (J_s + J_a)
            # if self.verbose:
                # print('J_s = {:.5f} ::::::: J_a = {:.5f} \n'.format(J_s, J_a))
            
            if self.minmax == 'min':
                rew_l2 = -reward
            else:
                rew_l2 = reward
        # print(f'{self.cont} of {self.cont0 + self.Nt}')
            
        if self.full_log:
            self.current_rewards.append(rew_l2)
            self.current_actions.append(action)
        # Switch variables before next step
        self.u1[:] = self.u

        # Exit condition
        if self.cont == self.cont0 + self.Nt: # or abs(np.max(self.u))  > self.u_lim
            # if self.full_log:
            self.episodes_logbook.update({
                'actions': self.current_actions,
                'observations': self.current_observations,
                'J_s': self.j_s,
                'J_a': self.j_a,
                'rewards': self.current_rewards
            })
            self.episodes_logbook['ep_rewards'].append(np.sum(self.current_rewards))
            done = True
            
            if self.verbose:
                print(f"Episode reward: {self.tot_reward}")
        else:
            done = False
            
        # Observation collection
        # observation = np.array([self.u[770], self.u[800], self.u[830]])
        observation = np.array([self.u[400], self.u[450], self.u[500]])
        if self.full_log:
            self.current_observations.append(observation)
        self.cont += 1
        info = {}  # Optional
        
        self.tot_reward += rew_l2
        
        if log_full_state==True:
            return observation, rew_l2, done, self.u
        else:
            return observation, rew_l2, done, self.episodes_logbook
    
    def reset(self):
        """
        This function reset the simulation to its starting point
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        
        self.tot_reward = 0 

        self.u, self.u1, self.t, self.cont = initial_conditions(ic=self.ic, dt=self.dt,
                                                                Nt=self.Nt, Nx=self.Nx,
                                                                FOLDER='./')
        self.cont0 = np.copy(self.cont)
        # Initialize the conunter
        self.action_replay.clear()
        # --- Saving the current logs and delete current one
        # if self.full_log:
        keys = ['ep_rewards', 'actions', 'observations', 'J_s', 'J_a']
        self.episodes_logbook = {chapter: [] for chapter in keys}
        self.current_rewards = []
        self.current_actions = []
        self.current_observations = []
        self.j_s = []
        self.j_a = []

        # Observation collection
        # observation = np.array([self.u[770], self.u[800], self.u[830]])
        observation = np.array([self.u[400], self.u[450], self.u[500]])
        return observation
    
    def render(self, mode='human', make_gif: bool = False,
               visual_inspection: bool = False, folder_out: str = './'):

        fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, figsize=(15, 15))
        ax1.plot(self.current_actions, label='actions')
        ax1.set_xlabel('t [t]')
        ax1.set_ylabel('$a_t [-]$')
        # ax1.plot(np.asarray(self.log['noise']), label='noise', ls='--', lw=0.4)
        # ax1.plot(raw_actions, label='raw actions', ls='--', lw=0.4)
        ax1.set_ylim(-1.2, 1.2)
        # ax1.legend()
        ax2.set_title(r'Functional analysis. $\gamma = {}$'.format(self.GAMMA))
        ax2.plot(self.current_rewards, label='$r_t$')
        ax2.plot(self.j_s, label='$J_s = || u ||_2$')
        ax2.plot(self.j_a, label='$J_a = \\gamma a_t^2$')
        ax2.legend()
        ax2.set_xlabel('timestep')
        ax2.set_ylabel('$r_i$')
        ax3.plot(np.asarray(self.current_observations)[:, 0], label='o1')
        ax3.plot(np.asarray(self.current_observations)[:, 1], label='o2')
        ax3.plot(np.asarray(self.current_observations)[:, 2], label='o3')
        ax3.legend()
        ax4.plot(np.asarray(self.current_observations)[-100:, 0], self.current_actions[-100:], label='o1')
        ax5.plot(np.asarray(self.current_observations)[-100:, 1], self.current_actions[-100:], label='o2')
        ax6.plot(np.asarray(self.current_observations)[-100:, 2], self.current_actions[-100:], label='o3')
        ax4.set_xlabel('$s_1$')
        ax5.set_xlabel('$s_2$')
        ax6.set_xlabel('$s_3$')
        ax4.set_ylabel('$a_t$')
        ax4.legend()
        ax4.set_title(r'$\pi (s_i) \rightarrow a_t$')
        fig.tight_layout()
        # if visual_inspection:
        plt.show()
        
        if make_gif:
            plt.savefig(folder_out + 'visual_inspection_{}.pdf'.format(self.name_sim))
            plt.close()
            tmp_list = []
            tmp_actions = np.copy(self.current_actions)
            self.reset()
            done_tmp = False
            tmp_folder = folder_out + 'pic_tmp_burger_env/'
            os.makedirs(tmp_folder, exist_ok=True)
            ii = 0
            while not done_tmp:
                _, _, done_tmp, tmp_u = self.step(tmp_actions[ii], log_full_state=True)
                if ii % 20 == 0:
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 15))
                    ax1.plot(self.x, tmp_u)
                    ax1.set_xlabel('x [x]')
                    ax1.set_ylabel('u [u]')
                    ax1.set_ylim(- (self.u_lim * 1.8), (self.u_lim * 1.3))
                    ax2.plot(self.current_actions[:ii])
                    ax2.set_xlabel('t [t]')
                    ax2.set_ylabel('$a_t$[-]')
                    ax2.set_ylim(-1.3, 1.3)
                    fig.tight_layout()
                    plt.savefig(tmp_folder + 'step_{}.png'.format(ii))
                    plt.close()
                    tmp_list.append(imageio.imread(tmp_folder + 'step_{}.png'.format(ii)))
                ii += 1
                
            imageio.mimsave(folder_out + '{}.gif'.format(self.name_sim), tmp_list)
            shutil.rmtree(tmp_folder)
            
        pass
    
    def close(self):
        pass

