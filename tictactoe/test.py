import numpy as np
import itertools

a = np.array([0,1,2,3,4,5])
for i in itertools.product((-1, 0, 1), repeat=9):
    print(i)