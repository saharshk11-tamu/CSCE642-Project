from RadiationGridworld import RadiationGridworld

size = 100
agent_loc = [0, 0]
target_loc = [4, 4]
wall_locs = [[2, 3]]
radiation_locs = [[3, 2]]
radiation_consts = [[1, 1]]
render_mode = 'radiation_map'

env = RadiationGridworld(size, agent_loc, target_loc, wall_locs, radiation_locs, radiation_consts, render_mode)

env.render()