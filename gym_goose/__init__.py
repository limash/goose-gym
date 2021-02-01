from gym.envs.registration import register

register(
    id='goose-v0',
    entry_point='gym_goose.envs:GooseEnv',
)
