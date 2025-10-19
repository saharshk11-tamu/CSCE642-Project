GRID_SIZE = 10
NUM_AGENTS = 1        # can increase later
GOAL_POS = (9, 9)
RADIATION_ZONES = [(3, 3), (6, 6), (4, 8)]
RADIATION_STRENGTHS = [5.0, 1.0, 5.0]
NOISE_STD = 0.1
MAX_STEPS = 5
GAMMA_FACTORS = {'Cs-137': 3.26,
                 'Co-60' : 13,
                 'Ir-192': 4.8,
                 'Ra-226': 8.25}