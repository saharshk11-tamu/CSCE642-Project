from RadiationGridworld import RadiationGridworld

size = 5
agent_loc = [0, 0]
target_loc = [4, 4]
wall_locs = [[2, 3]]
radiation_locs = [[3, 2]]
radiation_consts = [[1, 1]]
render_mode = 'ansi'

env = RadiationGridworld(size, agent_loc, target_loc, wall_locs, radiation_locs, radiation_consts, render_mode)

steps = 10
total_reward = 0
done = False
i = 0
while not done and i < steps:
    action = env.action_space.sample()
    print(action)
    state, reward, done, _ = env.step(action)
    total_reward += reward
    i += 1

env.render()
print(total_reward)
