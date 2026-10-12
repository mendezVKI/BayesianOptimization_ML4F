"""
A script to launch a TensorForce simulation

"""

from Burgers_implicit_env import Burgers_training
from tensorforce.agents import Agent
from tensorforce.environments import Environment


def main():

    env = Environment.create(
        environment=Burgers_training, max_episode_timesteps=100
    )

    agent = Agent.create(
        agent='ppo', environment=env, batch_size=10, learning_rate=1e-3
    )

    # agent.initialize()
    sum_rewards = 0
    print("start simulation")

    for _ in range(100):
        states = env.reset()
        terminal = False
        while not terminal:
            actions = agent.act(states=states)
            states, terminal, reward = env.execute(actions=actions)
            agent.observe(terminal=terminal, reward=reward)
            sum_rewards += reward

    print('Mean episode reward:', sum_rewards / 100)

    pass


if __name__ == '__main__':
    main()
