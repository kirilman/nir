import numpy as np
import time

mean = 0.99
N = 1000
a = np.ones((N,N))*mean
p = 1
start_time = time.time()
for i in range(a.shape[0]):
    for j in range(a.shape[1]):
        p = p * a[i,j]
        # print(time.time())
print('Time :',time.time()-start_time)
print(a)
print('P = ',p)
print(mean**25)
print(a.prod())
# print(time.ctime(start_time))