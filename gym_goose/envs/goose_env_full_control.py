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
                             'min_food': 10
                         },
                         debug=debug)
        self._config = self._env.configuration

        self._debug = debug
        self._n_agents = 4
        self._players_obs = None

        self.action_space = spaces.Discrete(4)
        observations = tuple([spaces.Box(low=-0.5,
                                         high=1,
                                         shape=(self._config.rows, self._config.columns, 4),
                                         dtype=np.float64)
                              for _ in range(self._n_agents)])
        self.observation_space = spaces.Tuple((
            spaces.Tuple(observations),
            spaces.Box(low=0,
                       high=1,
                       shape=(1,),
                       dtype=np.float64)
        ))

    def reset(self):
        state = self._env.reset(self._n_agents)
        if self._debug:
            printout(state)
        self._players_obs = [None for _ in range(self._n_agents)]
        self._players_obs = self.get_players_obs(state, self._players_obs)
        scalars = np.asarray((state[0].observation.step / self._config.episodeSteps,))
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
            reward = [len(state[0].observation.geese[i]) + any(state[0].observation.geese[i])*400
                      for i in range(self._n_agents)]
        else:
            reward = [len(state[0].observation.geese[i]) for i in range(self._n_agents)]

        info = []
        for i in range(self._n_agents):
            info.append(state[i].info)
            info[i]['allowed_actions'] = []
        restricted = [OPPOSITE_ACTION_NAMES[actions[i]] for i in range(self._n_agents)]
        [info[y]['allowed_actions'].append(x) for y in range(self._n_agents) for x in ACTION_NAMES
            if ACTION_NAMES[x] != restricted[y]]
        scalars = np.asarray((state[0].observation.step / self._config.episodeSteps,))
        return (self._players_obs, scalars), reward, done, info

    def get_players_obs(self, state, players_obs):
        geese_deque = deque(state[0].observation['geese'])
        for i in range(self._n_agents):
            obs = get_obs(self._env.configuration, state[0].observation)  # get an observation
            players_obs[i] = get_obs_queue(obs, players_obs[i])  # put observation into a queue
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
    line = line / player_number
    obs = np.reshape(line, (config.rows, config.columns))
    return obs


def get_obs_queue(obs, old_obs_queue):
    if old_obs_queue is None:
        old_obs_queue = np.repeat(obs[:, :, np.newaxis], 4, axis=2)  # tf like: height, width, channels
    else:
        old_obs_queue[:, :, :3] = old_obs_queue[:, :, 1:]
        old_obs_queue[:, :, 3] = obs
    return old_obs_queue
