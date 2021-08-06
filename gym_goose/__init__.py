from gym.envs.registration import register

register(
    id='goose-v0',
    entry_point='gym_goose.envs:GooseEnv0',
)

register(
    id='goose-v1',
    entry_point='gym_goose.envs:GooseEnv1',
)

register(
    id='goose-v2',
    entry_point='gym_goose.envs:GooseEnv2',
)

register(
    id='goose-v3',
    entry_point='gym_goose.envs:GooseEnv3',
)

register(
    id='goose-v4',
    entry_point='gym_goose.envs:GooseEnv4',
)

register(
    id='goose-v5',
    entry_point='gym_goose.envs:GooseEnv5',
)

register(
    id='goose-v6',
    entry_point='gym_goose.envs:GooseEnv6',
)
