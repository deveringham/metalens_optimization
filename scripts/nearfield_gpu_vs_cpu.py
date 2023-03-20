import subprocess as sp
import itertools
import time

t = []
for p in range(10,180):
    
    print('p: ' + str(p))
    
    # Time.
    time_start = time.time()
    
    # Run a single optimization in a subprocess.
    args = ['python3', 'nearfield_optimization_pt.py'] + [str(p), str(0), str(10.0), str(0.8), str(0)]
    sp.run(args)
    
    time_end = time.time()
    t.append(time_end - time_start)
   
print(t)
