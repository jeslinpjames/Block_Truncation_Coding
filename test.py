import numpy as np
block = np.array([
            [161,160,163,155],
            [161,160,163,155],
            [160,159,154,154],
            [161,158,153,151]
        ], dtype=np.uint8)
mean = np.mean(block)
max_val = np.max(block)
min_val = np.min(block)
result = (min_val.astype(np.uint16) + max_val.astype(np.uint16) + mean.astype(np.uint16)) / 3
t = result.astype(np.uint8)
uh = np.mean(block[block > t])
ul = np.mean(block[block < t])
print("mean: ",mean)
print("max: ",max_val)
print("min: ",min_val)
print("T: ",t)
print("uh: ",uh)
print("ul: ",ul)