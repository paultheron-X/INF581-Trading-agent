from collections import deque
import numpy as np
import random
import torch
import torch.nn as nn

"""tot = deque(maxlen= 100)

for i in range(300):
    tot.append(np.array([i, i+1]))

batch = random.sample(tot, 10)
"""

m = nn.Conv2d(2, 28, 3, stride=1)


input = torch.randn(1, 2, 5, 5)

print(input)

output = m(input)

print(output)