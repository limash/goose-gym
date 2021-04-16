import time
import random
import pickle

import numpy as np
import tensorflow as tf
import gym
from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration
from kaggle_environments import make

from gym_goose.envs.goose_env_full_control_challenge import ACTION_NAMES, OPPOSITE_ACTION_NAMES
# from gym_goose.envs.goose_env_full_control import get_obs, get_obs_queue
from gym_goose.envs.goose_env_full_control_challenge import get_feature_maps, to_binary
from goose_agent import models

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


def get_dqn_policy(env_name, is_duel=False):
    try:
        with open('data/data.pickle', 'rb') as file:
            init_data = pickle.load(file)
    except FileNotFoundError as err:
        raise err

    env = gym.make(env_name)
    space = env.observation_space
    feature_maps_shape = space[0][0].shape  # height, width, channels
    scalar_features_shape = space[1].shape
    input_shape = (feature_maps_shape, scalar_features_shape)
    n_outputs = env.action_space.n

    model = models.get_dqn(input_shape, n_outputs, is_duel=is_duel)
    model.set_weights(init_data['weights'])

    def policy(obs_in):
        obs = tf.nest.map_structure(lambda x: tf.expand_dims(x, axis=0), obs_in)
        Q_values = model(obs)
        int_act = np.argmax(Q_values[0])
        return ACTION_NAMES[int_act]
    return policy


def get_cat_policy(env_name):
    try:
        with open('data/data.pickle', 'rb') as file:
            init_data = pickle.load(file)
    except FileNotFoundError as err:
        raise err

    env = gym.make(env_name)
    space = env.observation_space
    feature_maps_shape = space[0][0].shape  # height, width, channels
    scalar_features_shape = space[1].shape
    input_shape = (feature_maps_shape, scalar_features_shape)
    n_outputs = env.action_space.n
    min_q_value = -5
    max_q_value = 20 
    n_atoms = 71
    cat_n_outputs = n_outputs * n_atoms
    support = tf.linspace(min_q_value, max_q_value, n_atoms)
    support = tf.cast(support, tf.float32)

    model = models.get_dqn(input_shape, cat_n_outputs)
    model.set_weights(init_data['weights'])

    def policy(obs):
        obs = tf.nest.map_structure(lambda x: tf.expand_dims(x, axis=0), obs)
        logits = model(obs)
        logits = tf.reshape(logits, [-1, n_outputs, n_atoms])
        probabilities = tf.nn.softmax(logits)
        Q_values = tf.reduce_sum(support * probabilities, axis=-1)  # Q values expected return
        int_act = np.argmax(Q_values[0])
        return ACTION_NAMES[int_act]
    return policy


# def get_geese_agent(policy):
#     def geese_agent(obs_dict, config_dict):
#         global previous_obs
#
#         state = Observation(obs_dict)
#         config = Configuration(config_dict)
#
#         obs = get_obs(config, state)  # get an observation
#         scalars = np.asarray((state.step / config.episode_steps,))
#         previous_obs = get_obs_queue(obs, previous_obs)  # put observation into a queue
#
#         action = policy((previous_obs, scalars))
#         return action
#     return geese_agent


class GeeseAgent:
    def __init__(self, policy):
        # self._previous_obs = None
        self._old_heads = np.zeros((4, 7 * 11), dtype=np.uint8)
        self._policy = policy
        self._n_agents = 4
        self._binary_positions = 8

    def get_action(self, obs_dict, config_dict):
        state = Observation(obs_dict)
        config = Configuration(config_dict)
        state.geese[0], state.geese[state.index] = state.geese[state.index], state.geese[0]

        # obs = get_obs(config, state)  # get an observation
        obs, self._old_heads = get_feature_maps(config,
                                                state,
                                                self._old_heads)
        
        time_step = np.asarray((state.step,))
        geese_len = np.array([len(state.geese[i]) for i in range(self._n_agents)])
        scalars_decimal = np.concatenate([geese_len, time_step])
        scalars = to_binary(scalars_decimal, self._binary_positions).ravel()
        # scalars = np.asarray((state.step,), dtype=np.uint8)
        # self._previous_obs = get_obs_queue(obs, self._previous_obs)  # put observation into a queue

        action = self._policy((obs, scalars))
        return action


def show_gym(number_of_iterations):
    env = gym.make('gym_goose:goose-full_control-v0', debug=True)
    for i in range(number_of_iterations):
        t0 = time.time()
        obs = env.reset()
        n_players = len(obs[0])
        available_actions = [0, 1, 2, 3]
        actions = [random.choice(available_actions) for _ in range(n_players)]
        for step in range(200):
            obs, reward, done, info = env.step(actions)
            actions = [random.choice(info[i]['allowed_actions']) for i in range(n_players)]
            if all(done):
                break
        t1 = time.time()
        print(f"A number of steps is {step+1}")
        print(f"Time elapsed is {t1-t0}")


if __name__ == '__main__':
    number_of_games = 10
    # show_gym(number_of_games)

    environment = make('hungry_geese', configuration={'min_food': 2})
    # trained_policy = get_dqn_policy('gym_goose:goose-full_control-v0')
    trained_policy = get_cat_policy('gym_goose:goose-full_control-v0')
    geese = [GeeseAgent(trained_policy) for _ in range(4)]
    logs = environment.run([goose.get_action for goose in geese])
    print("Done")
