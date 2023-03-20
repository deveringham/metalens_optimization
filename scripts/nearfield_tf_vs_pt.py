import subprocess as sp
import itertools
import time

use_pt = True

t = []
for p in range(50,180):
    
    print('p: ' + str(p))
    
    # Measure.
    time_start = time.time()
    
    if use_pt:
        # Run a single optimization in a subprocess (pt).
        args = ['python3', 'nearfield_optimization_pt.py'] + [str(p), str(0), str(10.0), str(0.8), str(0)]
        sp.run(args)
    
    else:
        # Run a single optimization in a subprocess (tf).
        args = ['python3', 'nearfield_optimization_tf.py'] + [str(p), str(0), str(10.0), str(0.8), str(0)]
        sp.run(args)
    
    time_end = time.time()
    t.append(time_end - time_start)
    
print(t)