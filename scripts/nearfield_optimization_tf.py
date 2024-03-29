# Use these parameters to choose which devices to use.
use_GPU = True

# Import device utils.
import sys
import os
sys.path.append('./src/')
sys.path.append('./rcwa_tf/src/')
import utils

# Configure GPUs.
if (use_GPU): utils.config_gpu_memory_usage()

# Measure GPU memory usage.
if (use_GPU):
    gpu_memory_init = utils.gpu_memory_info()

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import solver
import solver_metasurface

# Initialize parameters.
user_params = {}

# Tunable parameters.
# These are the values used if hyperparameter grid search is disabled.
user_params['pixelsX'] = int(sys.argv[1])
user_params['N'] = int(sys.argv[2])
user_params['sigmoid_update'] = float(sys.argv[3])
user_params['learning_rate'] = float(sys.argv[4])
user_params['initial_height'] = int(sys.argv[5])

user_params['parameter_string'] = 'N' + str(user_params['N']) \
    + '-sigmoid_update' + str(user_params['sigmoid_update']) \
    + '-learning_rate' + str(user_params['learning_rate']) \
    + '-initial_height' + str(user_params['initial_height'])

# Source parameters.
user_params['wavelengths'] = [120.0]
user_params['thetas'] = [0.0]
user_params['phis'] = [0.0]
user_params['pte'] = [1.0]
user_params['ptm'] = [0.0]

# Device parmeters.
user_params['pixelsY'] = user_params['pixelsX']
user_params['erd'] = 11.9
user_params['ers'] = user_params['erd']
user_params['L'] = [50.0, 50.0, 50.0, 50.0, 50.0, 950.0]
user_params['Lx'] = 5000.0 / user_params['pixelsX']
user_params['Ly'] = user_params['Lx']
user_params['f'] = 0.0 # Focal distance (nm)

# Solver parameters.
user_params['PQ'] = [3,3]
user_params['upsample'] = 11

# Problem parameters.
user_params['w_l1'] = 1.0
user_params['sigmoid_coeff'] = 0.1
user_params['focal_spot_radius'] = 10
user_params['enable_random_init'] = False
user_params['enable_debug'] = False
user_params['enable_print'] = True
user_params['enable_timing'] = True

# Logging parameters.
user_params['enable_logging'] = True
user_params['log_filename_prefix'] = './results/nearfield-' + str(user_params['pixelsX']) + 'x' + str(user_params['pixelsY']) + '-'
user_params['log_filename_extension'] = '.txt'

def loss_function(h, params):
    
    # Generate permittivity and permeability distributions.
    ER_t, UR_t = solver_metasurface.generate_layered_metasurface(h, params)

    # Simulate the system.
    outputs = solver.simulate_allsteps(ER_t, UR_t, params)
    
    # First loss term: maximize sum of electric field magnitude within some radius of the desired focal point.
    r = params['focal_spot_radius']
    field = outputs['ty'][:, :, :, np.prod(params['PQ']) // 2, 0]
    focal_plane = solver.propagate(params['input'] * field, params['propagator'], params['upsample'])
    index = (params['pixelsX'] * params['upsample']) // 2
    l1 = tf.reduce_sum(tf.abs(focal_plane[0, index-r:index+r, index-r:index+r]))

    # Final loss: (negative) field intensity at focal point + field intensity elsewhere.
    return -params['w_l1']*l1

# Set loss function.
user_params['loss_function'] = loss_function
    
# Optimize.
h, loss, params, focal_plane = solver_metasurface.optimize_device(user_params)

gpu_memory_final = utils.gpu_memory_info()
gpu_memory_used = [gpu_memory_final[1][0] - gpu_memory_init[1][0], gpu_memory_final[1][1] - gpu_memory_init[1][1]]

with open('tf_mem.txt', 'a', encoding="utf-8") as f:
    read_data = f.write(str(gpu_memory_used))