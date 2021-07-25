import time
import random
import pickle

import numpy as np
import tensorflow as tf
import gym
from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration
# from kaggle_environments import make

from gym_goose.envs.goose_env_0 import get_obs, ACTION_NAMES, OPPOSITE_ACTION_NAMES
from tf_reinforcement_testcases import models

ACTIONS = [0, 1, 2, 3]
previous_obs = None


def random_policy(unused_input):
    """
    It chooses a random action excluding an opposite action to the previous step
    """
    global ACTIONS
    action = random.choice(ACTIONS)
    restricted = OPPOSITE_ACTION_NAMES[action]
    ACTIONS = [x for x in ACTION_NAMES if ACTION_NAMES[x] != restricted]
    return ACTION_NAMES[action]


def get_cat_policy(env_name):
    try:
        with open('data/data.pickle', 'rb') as file:
            init_data = pickle.load(file)
    except FileNotFoundError as err:
        raise err

    env = gym.make(env_name)
    input_shape = env.observation_space.shape[0]
    n_outputs = env.action_space.n
    min_q_value = 0
    max_q_value = 51
    n_atoms = 51
    cat_n_outputs = n_outputs * n_atoms
    support = tf.linspace(min_q_value, max_q_value, n_atoms)
    support = tf.cast(support, tf.float32)

    model = models.get_mlp(input_shape, cat_n_outputs)
    model.set_weights(init_data['weights'])

    def policy(obs):
        global previous_obs

        if previous_obs is None:
            previous_obs = obs
        double_obs = np.concatenate((previous_obs, obs))
        previous_obs = obs

        logits = model(double_obs[None, ...])
        logits = tf.reshape(logits, [-1, n_outputs, n_atoms])
        probabilities = tf.nn.softmax(logits)
        Q_values = tf.reduce_sum(support * probabilities, axis=-1)  # Q values expected return
        int_act = np.argmax(Q_values[0])
        return ACTION_NAMES[int_act]
    return policy


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
    # trained_policy = get_cat_policy('gym_goose:goose-v0')
    # goose = get_geese_agent(trained_policy)
    # logs = environment.run([goose, "greedy"])
