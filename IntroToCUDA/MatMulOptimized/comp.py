import numpy as np
import time as t

a = np.random.randn(4096,4096)
b = np.random.randn(4096,4096)

t1 = t.time()
print(a@b)
t2 = t.time()-t1
print(t2*1000)