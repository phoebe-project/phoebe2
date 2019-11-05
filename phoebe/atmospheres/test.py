import numpy as np

x = np.zeros((14,7,35,2,1))
print(x.shape)

for i in range(len(x.shape)-1):
    v = np.moveaxis(x, i, 0)

    print(v.shape)
