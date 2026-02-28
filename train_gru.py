# train_gru.py

import random
from modules.prediction import CrowdPredictor

data = []

density = 0.2
flow = 0.3

for _ in range(300):
    density += random.uniform(-0.05, 0.05)
    flow += random.uniform(-0.05, 0.05)

    density = max(0, min(1, density))
    flow = max(0, min(1, flow))

    data.append([density, flow])

predictor = CrowdPredictor()
predictor.train(data)