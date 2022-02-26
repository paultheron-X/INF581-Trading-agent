from collections import deque
import numpy as np
import random

tot = deque(maxlen= 100)

for i in range(300):
    tot.append(np.array([i, i+1]))

batch = random.sample(tot, 10)
