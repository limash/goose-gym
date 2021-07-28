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


class GooseEnv(gym.Env, ABC):

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
        self._old_heads = [np.zeros(4, dtype=np.uint8) for _ in range(self._n_agents)]
        # self._old_heads = [np.zeros((self._n_agents, self._config.rows * self._config.columns), dtype=np.uint8)
        #                    for _ in range(self._n_agents)]
        # self._actions = [0 for _ in range(self._n_agents)]

        self.action_space = spaces.Discrete(4)
        self._binary_positions = 8
        # 4 maps for -3, -2, -1, 0 steps
        # observations = tuple([spaces.Box(low=-0.5,
        #                                  high=1,
        #                                  shape=(self._config.rows, self._config.columns, 4),
        #                                  dtype=np.float64)
        #                       for _ in range(self._n_agents)])
        observation = spaces.Box(low=0, high=1,
                                 # goose number, goose length, head, previous head, tail, goose coordinates
                                 shape=(self._n_agents, 5 + self._config.rows * self._config.columns // 2),
                                 dtype=np.uint8)
        scalars = spaces.Box(low=0,
                             high=1,
                             # time and 2 foods
                             shape=(3,),
                             dtype=np.uint8)
        observations = tuple([spaces.Tuple((observation, scalars)) for _ in range(self._n_agents)])
        self.observation_space = spaces.Tuple(observations)
        self._rewards = [-1, -0.33, 0.33, 1]
        self._num_alive = 4
        # self.observation_space = observations

    def reset(self):
        self._geese_length = np.ones(self._n_agents)  # for rewards
        self._rewards = [-1, -0.33, 0.33, 1]
        self._num_alive = 4

        state = self._env.reset(self._n_agents)
        # self._actions = [0 for _ in range(self._n_agents)]
        self._old_heads = [np.zeros(4, dtype=np.uint8) for _ in range(self._n_agents)]
        if self._debug:
            printout(state)
        self._players_obs = [None for _ in range(self._n_agents)]
        self._players_obs = self.get_players_obs(state, self._players_obs)
        return self._players_obs

    def step(self, actions):
        action_names = [ACTION_NAMES[action] for action in actions]
        # self._actions = [action + 1 for action in actions]

        state = self._env.step(action_names)
        # for i in range(self._n_agents):
        #     if state[i].status == 'DONE':
        #         self._actions[i] = 0

        if self._debug:
            printout(state)
            if any([] == x for x in state[0].observation['geese']):
                print("Somebody is dead")
            print("----Next step----:")

        # get map observations for all geese
        self._players_obs = self.get_players_obs(state, self._players_obs)

        geese_len = np.array([len(state[0].observation.geese[i]) for i in range(self._n_agents)])
        done = [True if state[i].status != 'ACTIVE' else False for i in range(self._n_agents)]

        # calculate rewards for all geese
        alive = [1 if geese_len[i] > 0 else 0 for i in range(self._n_agents)]
        current_alive = sum(alive)
        if current_alive < self._num_alive or all(done):
            deaths = self._num_alive - current_alive
            self._num_alive = current_alive
            if all(done) and current_alive == 1:
                place = get_len_place(geese_len)
                previously_alive = np.where(self._geese_length != 0, 1, 0)
                sum_prev_alive = previously_alive.sum()
                sum_prev_dead = 4 - sum_prev_alive
                if sum_prev_alive > 2:
                    prev_place = get_len_place(self._geese_length)
                    max_prev = prev_place.max()
                    winner_arg = place.argmax()
                    if prev_place.argmax() != winner_arg:
                        prev_place[winner_arg] = max_prev + 1
                    prev_place = prev_place - sum_prev_dead
                    # place = prev_place
                    place = get_len_place(prev_place)
                # mask previously dead geese
                masked = np.where(self._geese_length == 0, 1, 0)
            elif all(done) and current_alive > 1:
                place = get_len_place(self._geese_length)
                masked = np.where(self._geese_length == 0, 1, 0)
            else:
                place = get_len_place(self._geese_length)
                # mask alive and previously dead geese
                masked = np.array(alive) + np.where(self._geese_length == 0, 1, 0)
            foo = np.ma.masked_array(place, mask=masked)
            bar = foo - foo.min()
            rewards = []
            for item in bar:
                if type(item) == np.int64:
                    rewards.append(self._rewards[item])
                else:
                    rewards.append(0)
            for _ in range(deaths):
                self._rewards.pop(0)
        else:
            rewards = [0, 0, 0, 0]

        self._geese_length = geese_len

        # add to info allowed actions information (allowed != opposite to the previous actions)
        info = []
        for i in range(self._n_agents):
            info.append(state[i].info)
            info[i]['allowed_actions'] = []
        restricted = [OPPOSITE_ACTION_NAMES[actions[i]] for i in range(self._n_agents)]
        [info[y]['allowed_actions'].append(x) for y in range(self._n_agents) for x in ACTION_NAMES
         if ACTION_NAMES[x] != restricted[y]]

        return self._players_obs, rewards, done, info

    def get_players_obs(self, state, players_obs):
        geese_deque = deque(state[0].observation['geese'])
        # actions_deque = deque(self._actions)
        geese_len_deque = deque([len(state[0].observation.geese[i]) for i in range(self._n_agents)])
        time_step = np.asarray((state[0].observation.step,))
        time = time_step
        # time = to_binary(time_step, self._binary_positions).ravel()
        food = np.zeros(2, dtype=np.uint8)
        food[:] = state[0].observation['food']
        scalars = np.concatenate([time, food])
        for i in range(self._n_agents):
            observation, self._old_heads[i] = get_feature_maps(
                self._env.configuration, state[0].observation, np.array(geese_len_deque), self._old_heads[i]
            )
            # store observations
            players_obs[i] = (observation, scalars)
            # rotate geese
            geese_deque.rotate(-1)
            geese_len_deque.rotate(-1)
            # actions_deque.rotate(-1)
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
    # min_length = np.min(geese_len[np.nonzero(geese_len)])
    min_length = np.min(geese_len)
    non_min_length_args = np.where(geese_len > min_length)[0]
    length_bonus = np.where(geese_len > min_length, 0.333, 0)
    geese_len = geese_len[geese_len > min_length]
    if geese_len.size != 0:
        bonus = get_len_bonus(geese_len)
        length_bonus[non_min_length_args] += bonus
        return length_bonus
    else:
        return length_bonus


def get_len_place(geese_len):

    min_length = np.min(geese_len)
    non_min_length_args = np.where(geese_len > min_length)[0]
    length_bonus = np.where(geese_len > min_length, 1, 0)
    geese_len = geese_len[geese_len > min_length]
    if geese_len.size != 0:
        bonus = get_len_place(geese_len)
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


def get_feature_maps(config, state, geese_len, old_heads):
    n_geese = len(state['geese'])
    goose_positions = config.rows * config.columns // 2
    # goose number, goose length, head, previous head, tail, goose coordinates
    A = np.zeros((n_geese, 1 + 1 + 1 + 1 + 1 + goose_positions), dtype=np.uint8)
    for i, goose in enumerate(state['geese']):
        goose_array = np.zeros(1, dtype=np.uint8) if not goose else np.array(goose) + 1
        A[i, 0] = i + 1
        A[i, 1] = geese_len[i]
        A[i, 2] = goose_array[0]
        A[i, 3] = old_heads[i]
        A[i, 4] = goose_array[-1]
        A[i, 5:5+geese_len[i]] = goose_array

    return A, A[:, 2]
