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

        self.action_space = spaces.Discrete(4)  # 4 discrete actions - to fix
        # 4 maps for -3, -2, -1, 0 steps
        # observations = tuple([spaces.Box(low=-0.5,
        #                                  high=1,
        #                                  shape=(self._config.rows, self._config.columns, 4),
        #                                  dtype=np.float64)
        #                       for _ in range(self._n_agents)])
        observations = tuple([spaces.Box(low=0,
                                         high=1,
                                         shape=(self._config.rows, self._config.columns, self._n_agents*4+1),
                                         dtype=np.uint8)
                              for _ in range(self._n_agents)])
        self.observation_space = spaces.Tuple((
            spaces.Tuple(observations),
            spaces.Box(low=0,
                       high=200,
                       shape=(1,),
                       dtype=np.uint8)
        ))

    def reset(self):
        state = self._env.reset(self._n_agents)
        if self._debug:
            printout(state)
        self._geese_length = np.ones(self._n_agents)  # for rewards
        self._players_obs = [None for _ in range(self._n_agents)]
        self._players_obs = self.get_players_obs(state, self._players_obs)
        # scalars = np.asarray((state[0].observation.step / self._config.episodeSteps,))
        scalars = np.asarray((state[0].observation.step,), dtype=np.uint8)
        return self._players_obs, scalars

    def step(self, actions):
        action_names = [ACTION_NAMES[action] for action in actions]
        state = self._env.step(action_names)
        if self._debug:
            printout(state)
            if any([] == x for x in state[0].observation['geese']):
                print("Somebody is dead")
            print("----Next step----:")
        self._players_obs = self.get_players_obs(state, self._players_obs)

        done = [True if state[i].status != 'ACTIVE' else False for i in range(self._n_agents)]
        if all(done):
            # reward = [len(state[0].observation.geese[i]) +
            # any(state[0].observation.geese[i])*state[0].observation.step for i in range(self._n_agents)]
            geese_len_new = np.array([len(state[0].observation.geese[i]) for i in range(self._n_agents)])
            reward = geese_len_new - self._geese_length
            self._geese_length = geese_len_new
            reward += [len(state[0].observation.geese[i]) for i in range(self._n_agents)]
        else:
            geese_len_new = np.array([len(state[0].observation.geese[i]) for i in range(self._n_agents)])
            reward = geese_len_new - self._geese_length
            self._geese_length = geese_len_new

        info = []
        for i in range(self._n_agents):
            info.append(state[i].info)
            info[i]['allowed_actions'] = []
        restricted = [OPPOSITE_ACTION_NAMES[actions[i]] for i in range(self._n_agents)]
        [info[y]['allowed_actions'].append(x) for y in range(self._n_agents) for x in ACTION_NAMES
            if ACTION_NAMES[x] != restricted[y]]
        # scalars = np.asarray((state[0].observation.step / self._config.episodeSteps,))
        scalars = np.asarray((state[0].observation.step,), dtype=np.uint8)
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
    line = np.zeros([config.rows*config.columns])
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
    A = np.zeros((number_of_layers, config.rows*config.columns), dtype=np.uint8)
    for idx, goose in enumerate(state['geese']):
        A[0 + idx, goose[:1]] = 1  # head
        A[n_geese + idx, goose[-1:]] = 1  # tail
        A[2*n_geese + idx, goose] = 1  # body
    A[3*n_geese:4*n_geese, :] = old_heads
    A[4*n_geese, state['food']] = 1
    B = A.reshape((-1, config.rows, config.columns))
    C = np.moveaxis(B, 0, -1)
    return C, A[:n_geese, :]
