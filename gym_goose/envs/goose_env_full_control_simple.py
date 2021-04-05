from abc import ABC
from collections import deque

import numpy as np
import gym
from gym import spaces

from kaggle_environments import make

ACTION_NAMES = {0: 'NORTH',
                1: 'SOUTH',
                2: 'WEST',
                3: 'EAST'}

OPPOSITE_ACTION_NAMES = {0: 'SOUTH',
                         1: 'NORTH',
                         2: 'EAST',
                         3: 'WEST'}


class GooseEnvFullControl(gym.Env, ABC):

    def __init__(self, debug=False):
        self._env = make('hungry_geese',
                         configuration={
                             'min_food': 2
                         },
                         debug=debug)
        self._config = self._env.configuration

        self._debug = debug
        self._n_agents = 4
        self._geese_length = np.ones(self._n_agents)  # for rewards
        self._players_obs = None
        self._old_heads = [np.zeros((self._n_agents, self._config.rows * self._config.columns), dtype=np.uint8)
                           for _ in range(self._n_agents)]

        self.action_space = spaces.Discrete(4)
        # 4 maps for -3, -2, -1, 0 steps
        # observations = tuple([spaces.Box(low=-0.5,
        #                                  high=1,
        #                                  shape=(self._config.rows, self._config.columns, 4),
        #                                  dtype=np.float64)
        #                       for _ in range(self._n_agents)])
        observations = tuple([spaces.Box(low=0,
                                         high=1,
                                         # for each agent positions of head, tail, body, previous head + food position
                                         shape=(self._config.rows, self._config.columns, self._n_agents * 4 + 1),
                                         dtype=np.uint8)
                              for _ in range(self._n_agents)])
        self._binary_positions = 8
        self.observation_space = spaces.Tuple((
            spaces.Tuple(observations),
            spaces.Box(low=0,
                       high=1,
                       # 4 geese and time
                       shape=(self._binary_positions * 5,),
                       dtype=np.uint8)
        ))

    def reset(self):
        state = self._env.reset(self._n_agents)
        if self._debug:
            printout(state)
        # map observation
        self._players_obs = [None for _ in range(self._n_agents)]
        self._players_obs = self.get_players_obs(state, self._players_obs)
        # scalar observations
        time_step = np.asarray((state[0].observation.step,))
        geese_len = np.array([len(state[0].observation.geese[i]) for i in range(self._n_agents)])
        scalars_decimal = np.concatenate([geese_len, time_step])
        scalars = to_binary(scalars_decimal, self._binary_positions).ravel()
        return self._players_obs, scalars

    def step(self, actions):
        action_names = [ACTION_NAMES[action] for action in actions]
        state = self._env.step(action_names)
        if self._debug:
            printout(state)
            if any([] == x for x in state[0].observation['geese']):
                print("Somebody is dead")
            print("----Next step----:")

        # get map observations for all geese
        self._players_obs = self.get_players_obs(state, self._players_obs)

        # get scalar observations
        # scalars = np.asarray((state[0].observation.step / self._config.episodeSteps,))
        time_step = np.asarray((state[0].observation.step,))
        geese_len = np.array([len(state[0].observation.geese[i]) for i in range(self._n_agents)])
        scalars_decimal = np.concatenate([geese_len, time_step])
        scalars = to_binary(scalars_decimal, self._binary_positions).ravel()

        # calculate rewards for all geese
        done = [True if state[i].status != 'ACTIVE' else False for i in range(self._n_agents)]
        if all(done):
            reward = [len(state[0].observation.geese[i]) + any(state[0].observation.geese[i])*10 for i in range(self._n_agents)]
            # reward = [len(state[0].observation.geese[i]) +
            #           any(state[0].observation.geese[i]) * state[0].observation.step for i in range(self._n_agents)]
            reward += (geese_len - self._geese_length)
        else:
            reward = geese_len - self._geese_length

        self._geese_length = geese_len
        # death_penalty = np.where(geese_len == 0, -10, 0)
        # num_dead = np.count_nonzero(geese_len == 0)
        # alive_bonus = np.where(geese_len > 0, num_dead * 0.333, 0)
        # if all(done):
        #     # to avoid passing geese_len == [0, 0, 0, 0] to get_len_bonus
        #     len_bonus = 0
        # else:
        #     len_bonus = get_len_bonus(geese_len)

        # reward = death_penalty + alive_bonus + len_bonus

        # if all(done):
        #     winner_bonus = np.where(geese_len > 0, 10, 0)
        #     reward += winner_bonus

        # add to info allowed actions information (allowed != opposite to the previous actions)
        info = []
        for i in range(self._n_agents):
            info.append(state[i].info)
            info[i]['allowed_actions'] = []
        restricted = [OPPOSITE_ACTION_NAMES[actions[i]] for i in range(self._n_agents)]
        [info[y]['allowed_actions'].append(x) for y in range(self._n_agents) for x in ACTION_NAMES
         if ACTION_NAMES[x] != restricted[y]]

        return (self._players_obs, scalars), list(reward), done, info

    def get_players_obs(self, state, players_obs):
        geese_deque = deque(state[0].observation['geese'])
        for i in range(self._n_agents):
            # obs = get_obs(self._env.configuration, state[0].observation)  # get an observation
            # players_obs[i] = get_obs_queue(obs, players_obs[i])  # put observation into a queue
            players_obs[i], self._old_heads[i] = get_feature_maps(self._env.configuration,
                                                                  state[0].observation,
                                                                  self._old_heads[i])
            geese_deque.rotate(-1)
            state[0].observation['geese'] = geese_deque
        return players_obs


def to_binary(d, m=8):
    """
    Args:
        d: is an array of decimal numbers to convert to binary
        m: is a number of positions in a binary number, 8 is enough for up to 256 decimal, 256 is 2^8
    Returns:
        np.ndarray of binary representation of d

    """
    reversed_order = ((d[:, None] & (1 << np.arange(m))) > 0).astype(np.uint8)
    return np.fliplr(reversed_order)


def get_len_bonus(geese_len):
    """
    Recursive, returns rewards for geese. The shortest goose gets 0, a longer one gets 0.33,
    and so on until 0.99.

    Args:
        geese_len: np array with length of all alive geese

    Returns:
        obs: np.ndarray with rewards for geese
    """
    min_length = np.min(geese_len[np.nonzero(geese_len)])
    non_min_length_args = np.where(geese_len > min_length)[0]
    length_bonus = np.where(geese_len > min_length, 0.333, 0)
    geese_len = geese_len[geese_len > min_length]
    if geese_len.size != 0:
        bonus = get_len_bonus(geese_len)
        length_bonus[non_min_length_args] += bonus
        return length_bonus
    else:
        return length_bonus


def printout(state):
    print(f"Step: {state[0].observation.step}")
    geese = state[0].observation.geese
    for i in range(len(state)):
        print(f"Goose index: {state[i].observation.index}, cells: {geese[i]}, "
              f"status: {state[i].status}, action: {state[i].action}, kaggle reward: {state[i].reward}")


def get_obs(config, state):
    """
    Returns an observation map with geese and food;

    Args:
        config: kaggle environment env.configuration object
        state: kaggle environment env.reset or env.step methods output object

    Returns:
        obs: np.ndarray with observations
    """

    player_number = 2
    food_number = 1
    enemy_number = -1

    # mark geese
    n_geese = len(state['geese'])
    line = np.zeros([config.rows * config.columns])
    line[state['geese'][0]] = player_number
    for i in range(1, n_geese):
        line[state['geese'][i]] = enemy_number
    # mark food
    line[state['food']] = food_number
    # normalize
    # norm_line = line / player_number
    norm_line = (line - line.mean()) / line.std()
    obs = np.reshape(norm_line, (config.rows, config.columns))
    return obs


def get_obs_queue(obs, old_obs_queue):
    if old_obs_queue is None:
        old_obs_queue = np.repeat(obs[:, :, np.newaxis], 4, axis=2)  # tf like: height, width, channels
    else:
        old_obs_queue[:, :, :3] = old_obs_queue[:, :, 1:]
        old_obs_queue[:, :, 3] = obs
    return old_obs_queue


def get_feature_maps(config, state, old_heads):
    n_geese = len(state['geese'])
    # head, tail, body, previous head plus food
    number_of_layers = n_geese * 4 + 1
    A = np.zeros((number_of_layers, config.rows * config.columns), dtype=np.uint8)
    for idx, goose in enumerate(state['geese']):
        A[0 + idx, goose[:1]] = 1  # head
        A[n_geese + idx, goose[-1:]] = 1  # tail
        A[2 * n_geese + idx, goose] = 1  # body
    A[3 * n_geese:4 * n_geese, :] = old_heads
    A[4 * n_geese, state['food']] = 1
    B = A.reshape((-1, config.rows, config.columns))

    # centering the player's goose
    center = (3, 5)  # row, column
    try:
        head_coords = np.argwhere(B[0, :, :] == 1)[0]
        row_shift = center[0] - head_coords[0]
        column_shift = center[1] - head_coords[1]
        B1 = np.roll(B, row_shift, axis=1)
        B2 = np.roll(B1, column_shift, axis=2)
    except IndexError:  # if the goose is dead
        B2 = B

    C = np.moveaxis(B2, 0, -1)
    return C, A[:n_geese, :]
