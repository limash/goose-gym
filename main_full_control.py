import time
import random
import pickle

import numpy as np
import tensorflow as tf
import gym
from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action, row_col
from kaggle_environments import make

# from gym_goose.envs.goose_env_3 import ACTION_NAMES, OPPOSITE_ACTION_NAMES
# from gym_goose.envs.goose_env_3 import get_feature_maps, to_binary
from gym_goose.envs.goose_env_5 import ACTION_NAMES, OPPOSITE_ACTION_NAMES
from gym_goose.envs.goose_env_5 import get_feature_maps, to_binary
from tf_reinforcement_agents import models

ACTIONS = [0, 1, 2, 3]
ACTION_INTS = {'NORTH': 0,
               'SOUTH': 1,
               'WEST': 2,
               'EAST': 3}
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
    scalar_features_shape = space[0][1].shape
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
    scalar_features_shape = space[0][1].shape
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


def get_pg_policy(env_name, file='data/data.pickle'):
    try:
        with open(file, 'rb') as file:
            init_data = pickle.load(file)
    except FileNotFoundError as err:
        raise err

    env = gym.make(env_name)
    space = env.observation_space
    feature_maps_shape = space[0][0].shape
    scalar_features_shape = space[0][1].shape
    input_shape = (feature_maps_shape, scalar_features_shape)
    n_outputs = env.action_space.n

    # model = models.get_actor_critic(input_shape, n_outputs)
    model = models.get_actor_critic3()
    # call a model once to build it before setting weights
    dummy_input = (tf.ones(feature_maps_shape, dtype=tf.uint8),
                   tf.ones(scalar_features_shape, dtype=tf.uint8))
    dummy_input = tf.nest.map_structure(lambda x: tf.expand_dims(x, axis=0), dummy_input)
    model(dummy_input)
    model.set_weights(init_data['weights'])

    def policy(obs_in):
        obs = tf.nest.map_structure(lambda x: tf.expand_dims(x, axis=0), obs_in)
        policy_logits, _ = model(obs)
        int_act = tf.random.categorical(policy_logits, num_samples=1, dtype=tf.int32)
        probs = tf.nn.softmax(policy_logits)
        return ACTION_NAMES[int_act.numpy()[0][0]]

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
        # self._old_heads = np.zeros((4, 7 * 11), dtype=np.uint8)
        self._old_heads = np.zeros(4, dtype=np.uint8)
        self._policy = policy
        self._n_agents = 4
        # self._binary_positions = 8

    def get_action(self, obs_dict, config_dict):
        state = Observation(obs_dict)
        config = Configuration(config_dict)
        state.geese[0], state.geese[state.index] = state.geese[state.index], state.geese[0]

        geese_len = np.array([len(state.geese[i]) for i in range(self._n_agents)])
        obs, self._old_heads = get_feature_maps(config,
                                                state,
                                                geese_len,
                                                self._old_heads)

        # time = to_binary(time_step, self._binary_positions).ravel()

        time_step = np.asarray((state.step,), dtype=np.uint8)
        food = np.zeros(2, dtype=np.uint8)
        food[:] = state['food']
        scalars = np.concatenate([time_step, food])

        # scalars_decimal = np.concatenate([geese_len, time_step])
        # scalars = to_binary(scalars_decimal, self._binary_positions).ravel()
        # scalars = np.asarray((state.step,), dtype=np.uint8)
        # self._previous_obs = get_obs_queue(obs, self._previous_obs)  # put observation into a queue

        action = self._policy((obs, scalars))
        return action


class GeeseAgent2:
    def __init__(self, policy):
        self._actions = None
        self._heads = None
        self._policy = policy
        self._n_agents = 4
        self._binary_positions = 8

    def get_action(self, obs_dict, config_dict):
        state = Observation(obs_dict)
        config = Configuration(config_dict)
        state.geese[0], state.geese[state.index] = state.geese[state.index], state.geese[0]

        if self._actions is None:
            self._actions = [0 for _ in range(self._n_agents)]
            self._heads = [row_col(state.geese[0][0], config.columns),
                           row_col(state.geese[1][0], config.columns),
                           row_col(state.geese[2][0], config.columns),
                           row_col(state.geese[3][0], config.columns)]
        else:
            # heads = [row_col(state.geese[0][0], config.columns),
            #          row_col(state.geese[1][0], config.columns),
            #          row_col(state.geese[2][0], config.columns),
            #          row_col(state.geese[3][0], config.columns)]
            heads = [None, None, None, None]
            for i in range(4):
                try:
                    heads[i] = row_col(state.geese[i][0], config.columns)
                    if (heads[i][0] < self._heads[i][0] and not(heads[i][0] == 0 and self._heads[i][0] == 6)
                            or (heads[i][0] == 6 and self._heads[i][0] == 0)):
                        self._actions[i] = 1  # north (+1 to geese style)
                    elif (heads[i][0] > self._heads[i][0] and not (heads[i][0] == 6 and self._heads[i][0] == 0)
                          or (heads[i][0] == 0 and self._heads[i][0] == 6)):
                        self._actions[i] = 2  # south
                    elif (heads[i][1] < self._heads[i][1] and not (heads[i][1] == 0 and self._heads[i][1] == 10)
                          or (heads[i][1] == 10 and self._heads[i][1] == 0)):
                        self._actions[i] = 3  # west
                    elif (heads[i][1] > self._heads[i][1] and not (heads[i][1] == 10 and self._heads[i][1] == 0)
                          or (heads[i][1] == 0 and self._heads[i][1] == 10)):
                        self._actions[i] = 4  # east
                except IndexError:
                    self._actions[i] = 0
            self._heads = heads

        obs = get_feature_maps(config, state, self._actions)

        time_step = np.asarray((state.step,))
        times = to_binary(time_step, self._binary_positions).ravel()
        food = np.zeros((config.rows * config.columns), dtype=np.uint8)
        food[state['food']] = 1
        scalars = np.concatenate([food, times])

        action = self._policy((obs, scalars))
        return action


def show_gym(number_of_iterations, policy=None):
    env = gym.make('gym_goose:goose-v5', debug=True)
    for i in range(number_of_iterations):
        all_rewards = np.zeros(4)
        t0 = time.time()
        obs = env.reset()
        n_players = len(obs)
        available_actions = [0, 1, 2, 3]
        if policy is None:
            actions = [random.choice(available_actions) for _ in range(n_players)]
        else:
            actions = [policy(obs[i]) for i in range(n_players)]
        for step in range(199):
            if policy is not None:
                actions = [ACTION_INTS[action] for action in actions]
            obs, reward, done, info = env.step(actions)
            all_rewards += np.array(reward)
            if policy is None:
                actions = [random.choice(info[i]['allowed_actions']) for i in range(n_players)]
            else:
                actions = [policy(obs[i]) for i in range(n_players)]
            if all(done):
                break
        t1 = time.time()
        print(f"A number of steps is {step + 1}")
        print(f"Time elapsed is {t1 - t0}")


if __name__ == '__main__':
    number_of_games = 1

    # trained_policy = get_dqn_policy('gym_goose:goose-full_control-v3')
    # trained_policy = get_cat_policy('gym_goose:goose-full_control-v0')
    trained_policy = get_pg_policy('gym_goose:goose-v5', file='data/data4000.pickle')

    # show_gym(number_of_games)  # , trained_policy)

    geese = [GeeseAgent(trained_policy) for _ in range(4)]
    environment = make('hungry_geese', configuration={'min_food': 2})
    logs = environment.run([goose.get_action for goose in geese])
    print("Done")
