#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from comb_gibbs_step import comb_gibbs_step 

THETA = 0.45
DIM = 60
NUM_ITERATIONS = 1000
SHOW_EVERY = 100

x = np.random.choice([1,-1], size=[DIM, DIM])

fig = plt.figure(figsize=(10, 10))
for it in range(1,NUM_ITERATIONS+1):
    x = comb_gibbs_step(x, THETA)
    if it % SHOW_EVERY == 0:        
        plt.subplot(4, 3, it//SHOW_EVERY)
        plt.imshow(x, cmap='gray')
plt.show()