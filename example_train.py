from Solvers.runner import Runner

runner = Runner('config.json')

print(runner.solver.policy_viz())
