import subprocess as sp
import itertools

param_grid = {'N': [100],
              'sigmoid_update': [5.0, 10.0, 20.0],
              'learning_rate': [4E-1, 8E-1, 1.2],
              'initial_height': [0,1,2,3,4,5]}

hp_grid = param_grid.values()
hp_names = param_grid.keys()

# Iterate over the grid.
for hyperparams in itertools.product(*hp_grid):
    
    print('\nTrying hyperparameters: ' + str(list(hp_names)))
    print(hyperparams)
    
    # Run a single optimization in a subprocess.
    args = ['python3', 'farfield_optimization_pt.py'] + [str(h) for h in hyperparams]
    sp.run(args)
