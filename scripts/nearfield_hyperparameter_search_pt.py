import subprocess as sp
import itertools

param_grid = {'N': [50],
              'sigmoid_update': [10.0],
              'learning_rate': [0.6, 0.7, 0.8, 0.9, 1.0],
              'initial_height': [3]}

hp_grid = param_grid.values()
hp_names = param_grid.keys()

# Iterate over the grid.
for hyperparams in itertools.product(*hp_grid):
    
    print('\nTrying hyperparameters: ' + str(list(hp_names)))
    print(hyperparams)
    
    # Run a single optimization in a subprocess.
    args = ['python3', 'nearfield_optimization_pt.py'] + [str(180)] + [str(h) for h in hyperparams]
    sp.run(args)
