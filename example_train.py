from radiation_env.env import RadiationEnv
import numpy as np

env = RadiationEnv()

obs, _ = env.reset()
done = False
total_reward = 0

while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    env.render()
    done = terminated or truncated

print("Episode reward:", total_reward)