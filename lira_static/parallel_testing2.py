import multiprocessing
from multiprocessing.pool import Pool
import numpy as np

a = []
def costly_function(i):
    a.append(i[0]*i[0])
    print "asdf"
    return i[0]*i[0]

nprocs = 4#NUMBER OF PROCESSES TO SPAWN

"""
Automatically loops through all of arg
"""
n = 10
def gen(n, a, b):
    for i in range(n):
        yield [i, 3]

pool = Pool(processes=nprocs)
p = pool.imap_unordered(costly_function,gen(n, "Kappa", "Pride"))
print a
for i in p:
    print i
print a
"""
pool.close()
pool.join()
print p.get()
"""
