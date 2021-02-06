from gym.envs.registration import register

register(
    id='goose-v0',
    entry_point='gym_goose.envs:GooseEnv',
)

register(
    id='goose-full_control-v0',
    entry_point='gym_goose.envs:GooseEnvFullControl',
)
