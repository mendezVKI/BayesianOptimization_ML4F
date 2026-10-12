import os
import typing
import warnings
from collections import deque
from typing import Union, Optional
import matplotlib.pyplot as plt
import pickle

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})

import gym
import numpy as np
from stable_baselines.common.vec_env import VecEnv

if typing.TYPE_CHECKING:
    pass
from stable_baselines.common.callbacks import BaseCallback, sync_envs_normalization
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.callbacks import EventCallback


# -------------------------------------------------------------------------
#                       Custom Callback
# -------------------------------------------------------------------------
class custom_EvalCallback(EventCallback):
    """
    Custom Callback for evaluating an agent. Features:
    + exit simulation if the agent is stuck in some local optima
    + dumping rew_history, momentum and n_timesteps for each sim for post processing
    + plot of rewards hist VS timestep, in order to have a feedback of the sim @ glance

    Two main parameters:
    :param treshold: modifying this the user can change the actual variance to be met before exiting the simulation
    :param K: defines when starting to store the rewards
    """

    def __init__(self, eval_env: Union[gym.Env, VecEnv],
                 ntrial: int = 1, # if doing hpo recall to pass each time an update value
                 callback_on_new_best: Optional[BaseCallback] = None,
                 n_eval_episodes: int = 5,
                 eval_freq: int = 10000,
                 log_path: str = None,
                 best_model_save_path: str = None,
                 deterministic: bool = True,
                 render: bool = False,
                 verbose: int = 1,
                 treshold: float = .15,
                 K: int = 15
                 ):
        super(custom_EvalCallback, self).__init__(callback_on_new_best, verbose=verbose)
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.render = render
        self.log_reward = deque(maxlen=10)
        self.treshold = treshold
        self.cont = 0
        self.activate = False
        self.rew_hist = []
        self.trial = ntrial
        self.momentum_list = []
        self.K = K

        # Creating the directory in which save the results: Folder + subfolder per each trial (if HPO)
        os.makedirs("./callback_tmp", exist_ok=True)
        self.FOLDER = f"./callback_tmp/sim_{ntrial}"
        os.makedirs(self.FOLDER, exist_ok=True)


        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])

        assert eval_env.num_envs == 1, "You must pass only one environment for evaluation"

        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        # Logs will be written in `evaluations.npz`5
        if log_path is not None:
            log_path = os.path.join(log_path, 'evaluations')
        self.log_path = log_path
        self.evaluations_results = []
        self.evaluations_timesteps = []
        self.evaluations_length = []

    def _init_callback(self):
        # Does not work in some corner cases, where the wrapper is not the same
        if not type(self.training_env) is type(self.eval_env):
            warnings.warn("Training and eval env are not of the same type"
                          "{} != {}".format(self.training_env, self.eval_env))

        # Create folders if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)


    def _on_step(self) -> bool:

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            sync_envs_normalization(self.training_env, self.eval_env)


            if self.num_timesteps > self.K*self.eval_freq: # to be sure that var != 0
                self.activate = True

            episode_rewards, episode_lengths = evaluate_policy(self.model, self.eval_env,
                                                                                     n_eval_episodes=self.n_eval_episodes,
                                                                                     render=self.render,
                                                                                     deterministic=self.deterministic,
                                                                                     return_episode_rewards=True)

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)
                np.savez(self.log_path, timesteps=self.evaluations_timesteps,
                         results=self.evaluations_results, ep_lengths=self.evaluations_length)

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            # Keep track of the last evaluation, useful for classes that derive from this callback
            self.last_mean_reward = mean_reward

            self.rew_hist.append(mean_reward) # --- History of rewards for plotting purposes
            self.log_reward.append(mean_reward) # - Last N rewards for exit purposes

            var = np.var(np.asarray(self.log_reward))

            if self.activate and var < self.treshold:
                '''
                If there is not a variation in the collected reward of the treshold% minimum then the simulation is killed
                '''
                print("## \n")
                print(f"This is not the simulation you're looking for. EXIT @ STEP:{self.num_timesteps}")
                print("## \n")

                np.savez(self.FOLDER + f"/sim_res_{self.trial}", rew=self.rew_hist, n=self.num_timesteps)
                self.model.save(os.path.join(self.FOLDER, 'model'), cloudpickle=True)


                return False



            if self.verbose > 0:
                print("Eval num_timesteps={}, "
                      "episode_reward={:.2f} +/- {:.2f}".format(self.num_timesteps, mean_reward, std_reward))
                print("Episode length: {:.2f} +/- {:.2f}".format(mean_ep_length, std_ep_length))

            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, 'best_model'))

                self.best_mean_reward = mean_reward
                # Trigger callback if needed
                if self.callback is not None:
                    return self._on_event()

        return True

    def on_rollout_end(self) -> None:
        '''

        :return:
        '''

        pass

    def _on_training_end(self) -> None:
        '''
        Making some post processing
        :return:
        '''

        # playing with finance indicators
        # --- Momentum: comparison between the reward now and X timesteps ago. Shorter X means instant comparison,
        # larger X vv. Compare with momentum finance indicator

        self.momentum_list = abs(np.asarray(self.rew_hist[5:]) - np.asarray(self.rew_hist[:-5]))
        np.savez(self.FOLDER + f"/momentum_sim_{self.trial}", self.momentum_list)
        n = np.linspace(1, self.num_timesteps, len(self.rew_hist))
        plt.figure()
        plt.plot(n, self.rew_hist)
        plt.xlabel("n of timesteps")
        plt.ylabel("reward")
        plt.grid()
        plt.savefig(self.FOLDER + "/test_overview.png", dpi = 800)
        plt.close()


        pass
