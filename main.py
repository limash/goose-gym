import time
import random

import gym
from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration
# from kaggle_environments import make

from gym_goose.envs.goose_env import get_obs, ACTION_NAMES, OPPOSITE_ACTION_NAMES

ACTIONS = [0, 1, 2, 3]


def random_policy(unused_input):
    """
    It chooses a random action excluding an opposite action to the previous step
    """
    global ACTIONS
    action = random.choice(ACTIONS)
    restricted = OPPOSITE_ACTION_NAMES[action]
    ACTIONS = [x for x in ACTION_NAMES if ACTION_NAMES[x] != restricted]
    return ACTION_NAMES[action]


def get_geese_agent(policy):
    def geese_agent(obs_dict, config_dict):
        state = Observation(obs_dict)
        config = Configuration(config_dict)
        obs = get_obs(config, state)
        action = policy(obs)
        return action
    return geese_agent


def show_gym(number_of_iterations):
    env = gym.make('gym_goose:goose-v0', debug=True)
    for i in range(number_of_iterations):
        t0 = time.time()
        env.reset()
        actions = [0, 1, 2, 3]
        for step in range(200):
            obs, reward, done, info = env.step(random.choice(actions))  # take a random action
            actions = info['allowed_actions']
            if done:
                break
        t1 = time.time()
        print(f"A number of steps is {step+1}")
        print(f"Time elapsed is {t1-t0}")


if __name__ == '__main__':
    number_of_games = 10
    show_gym(number_of_games)

    # environment = make("hungry_geese", debug=True)
    # goose = get_geese_agent(random_policy)
    # logs = environment.run([goose, "greedy"])
