from joblib import Parallel, delayed
import multiprocessing

import numpy as np

    
# what are your inputs, and what operation do you want to 
# perform on each input. For example...
inputs = np.arange(10) *2
def f(i): return i + 2

num_cores = multiprocessing.cpu_count()
print "%i CPU Cores" % num_cores
    
#outputs = np.zeros((10))
outputs = multiprocessing.RawArray('f', (10))

print np.array(outputs)


print inputs
def generator():
    for i in inputs:
        yield i

def function_to_be_parallellized(i,e):
    print i
    outputs[i] = np.mean(np.array([i*i,i*i+2]))

def main():
    Parallel(n_jobs=num_cores)( 
                delayed(function_to_be_parallellized(i,e)) for i,e  in enumerate(generator()) 
            )
    print np.array(outputs)


    print np.reshape(outputs, (2,5))
main()
