from gymnasium.envs.registration import register

register(
    id='LorenzEnv-v0',
    entry_point='env.4d_lorenz:LorenzEnv',
)

register(
    id='ChuaEnv-v0',
    entry_point='env.chua_env:ChuaEnv',
)

register(
    id='AiharaEnv-v0',
    entry_point='env.aihara_env:AiharaEnv',
)

register(
    id='HopfieldEnv-v0',
    entry_point='env.hopfield_env:HopfieldEnv',
)

register(
    id='ContinuousHopfieldEnv-v0',
    entry_point='env.continuous_hopfield_env:ContinuousHopfieldEnv',
)

register(
    id='ContinuousHopfield4DHHNNEnv-v0',
    entry_point='env.continuous_hopfield_4d_hhnn_env:ContinuousHopfield4DHHNNEnv',
)

register(
    id='ContinuousHopfield5DHOHNNEnv-v0',
    entry_point='env.continuous_hopfield_5d_hohnn_env:ContinuousHopfield5DHOHNNEnv',
)

register(
    id='FHNEnv-v0',
    entry_point='env.fhn_env:FHNEnv',
)

register(
    id='FHNEnv2D-v0',
    entry_point='env.fhn_env_2d:FHNEnv2D',
)

register(
    id='FHNEnv3D-v0',
    entry_point='env.fhn_env_3d:FHNEnv3D',
)

