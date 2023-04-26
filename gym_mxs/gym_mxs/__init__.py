from gym.envs.registration import register

# register(
#     id="gym_mxs/MXS-v0",
#     entry_point="gym_mxs.envs:MxsEnv",
# )
# from gymnasium.envs.registration import register

register(
    id="MXSBox2D-v0",
    entry_point="gym_mxs.envs:MxsEnvBox2D",
    max_episode_steps=1000,
)
