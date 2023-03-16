################################################################################
# Configurable Parameters
################################################################################
# Execution parameters.
########################################   

# Flag to enable GPU use.
use_gpu = True

# TensorFlow device names of GPU and CPU.
gpu_device = '/job:localhost/replica:0/task:0/device:GPU:1'
cpu_device = '/device:CPU:0'

# Flag to enable measurement of GPU memory usage.
# Automatically diabled if not using the GPU.
enable_gpu_memory_tracking = False


########################################   
# Important tunable parameters.
########################################   
# These are the values used if grid search is disabled.

# Maximum number of optimization iterations.
N = 10

# Number of device 'pixels', i.e. square regions of constant height,
# in each direction.
pixelsX = 19
pixelsY = pixelsX

# Coefficient used for differentiable thresholding annealing.
# At each optimization step, the coefficient of the sigmoid function used to
# force admissable solutions is increased by the increment N / sigmoid_update.
sigmoid_update = 40.0

# Learning rate provided to Keras optimizer.
learning_rate = 8E-1


########################################   
# Configuration for hyperparameter search.
########################################

# Flag to enable hyperparameter grid search.
enable_hyperparameter_gridsearch = True

# Values to use in hyperparameter grid search.
# Stored as a dict. Each dict key is the name of a tunable hyperparameter, i.e.
# 'N', and its value is a list of values to try for that hyperparameter.
param_grid = {'N': [100, 200],
              'sigmoid_update': [20.0, 40.0, 80.0],
              'learning_rate': [2E-1, 4E-1, 8E-1],
              'initial_height': [0, 1, 2, 3, 4, 5] }


########################################   
# Source parameters.
########################################   

# List of wavelengths over which to optimize (nm). thetas, phis, pte and ptm
# each correspond to wavelengths to create a lit of sources, each characterized
# by a wavelength, angle, and polarization.
wavelengths = [120000.0]

# Set of polar angles over which to optimize (radians). 
thetas = [0.0]

# Set of azimuthal angles over which to optimize (radians).
phis = [0.0]

# Set of TE polarization component magnitudes over which to optimize. A
# magnitude of 0.0 means no TE component. Under normal incidence, the TE
# polarization is parallel to the y-axis.
pte = [1.0]

# Set of TM polarization component magnitudes over which to optimize. A
# magnitude of 0.0 means no TM component. Under normal incidence, the TM
# polarization is parallel to the x-axis.
ptm = [0.0]

# Number of sources over which to optimize.
batchSize = len(wavelengths)


########################################   
# Device parmeters.
########################################   

# Relative permittivity of the non-vacuum, constituent material of the device
# layers.
erd = 12.04

# Relative permittivity of the substrate layer.
ers = erd

# Minimum and maximum allowed relative permittivity in the topology
# optimization.
eps_min = 1.0
eps_max = erd

# Number of device layers, including the substrate layer.
Nlay = 6

# Thickness of each layer (nm). L[0] corresponds to layer closest to the
# source, and L[-1] to the substrate layer.
L = [50000.0, 50000.0, 50000.0, 50000.0, 50000.0, 1200000.0] * Nlay

# Length of each pixel in the x direction (nm).
Lx = 20000.0
#Lx = 5000000.0 / pixelsX

# Length of each pixel in the y direction (nm).
Ly = Lx

f = 0.0 # Focal distance (nm)


########################################   
# Solver parameters.
########################################   

# Number of spatial harmonics used by RCWA in each transverse direction.
PQ = [5,5]

# Number of sample points within each pixel in the transverse directions.
# Does not really matter because all pixels are uniform, but the RCWA solver
# requires a minimum value here.
Nx = 16
Ny = Nx

# Upsampling rate (per pixel) used when simulating scattering from the device.
upsample = 11


########################################   
# Problem parameters.
########################################

# Weight of first loss term (incentivize intensity at target).
w_l1 = 1.0

# Weight of second loss term (disincentivize intensity elsewhere).
w_l2 = 0.0

# Starting value of coefficient used for differential thresholding of
# solutions.
sigmoid_coeff = 1.0

# Radius of focal spot, in units of Nx or Ny.
focal_spot_radius = 10

# Flag to enable random initial guesses for optimization.
enable_random_init = False

# If random_init is false, initializes all pixels to this starting height.
initial_height = 0

# Flag to enable debug statements.
enable_debug = False

# Flag to enable normal runtime status print statements.
enable_print = True

# Flag to enable benchmarking.
enable_timing = True


########################################   
# Logging parameters.
########################################

# Flag to enable logging.
enable_logging = True

# File name to use for logging.
log_filename = 'nearfield_' + str(pixelsX) + 'x' + str(pixelsY) + '.log'

################################################################################
# Configure Devices
################################################################################

import sys
import tensorflow as tf
sys.path.append('./rcwa_tf/src/')
sys.path.append('./src/')
import tf_utils

# Limit GPU memory growth.
tf_utils.config_gpu_memory_usage()

# Choose the device to run on.
tfDevice = gpu_device if use_gpu else cpu_device

# Measure GPU memory usage.
if use_gpu and enable_gpu_memory_tracking:
    gpu_memory_init = tf_utils.gpu_memory_info()


################################################################################
# Dependencies
################################################################################

import numpy as np
import time
import solver
import solver_metasurface


################################################################################
# Loss Function Definition
################################################################################

def loss_function(h, params):

    # Generate permittivity and permeability distributions.
    ER_t, UR_t = solver_metasurface.generate_layered_metasurface(h, params)

    # Simulate the system.
    outputs = solver.simulate(ER_t, UR_t, params)

    # First loss term: maximize sum of electric field magnitude within some
    # radius of the desired focal point.
    r = params['focal_spot_radius']
    field = outputs['ty'][:, :, :, np.prod(params['PQ']) // 2, 0]
    focal_plane = solver.propagate(params['input'] * field,
        params['propagator'], params['upsample'])
    index = (params['pixelsX'] * params['upsample']) // 2
    l1 = tf.math.reduce_sum(
        tf.abs(focal_plane[0, index-r:index+r, index-r:index+r]))

    # Second loss term: minimize sum of electric field magnitude elsewhere.
    l2 = tf.math.reduce_sum(tf.abs(focal_plane[0, :, :])) - l1

    # Final loss: (negative) field intensity at focal point + field intensity
    # elsewhere.
    return -params['w_l1']*l1 + params['w_l2']*l2


################################################################################
# Main Routine
################################################################################

# This context gurantees everythign is executed on the chosen device.
with tf.device(tfDevice):
    
    if enable_print:
        print()
        print('-------')
        print('Start of metasurface optimization routine.')
        print('Executing on device ' + tfDevice + ' ...')
        print('-------')
    
    # Initialize and populate dict of user-configurable parameters.
    user_params = {}
    user_params['enable_hyperparameter_gridsearch'] = enable_hyperparameter_gridsearch
    user_params['N'] = N
    user_params['sigmoid_update'] = sigmoid_update
    user_params['learning_rate'] = learning_rate
    user_params['param_grid'] = param_grid
    user_params['wavelengths'] = wavelengths
    user_params['thetas'] = thetas
    user_params['phis'] = phis
    user_params['pte'] = pte
    user_params['ptm'] = ptm
    user_params['batchSize'] = batchSize
    user_params['pixelsX'] = pixelsX
    user_params['pixelsY'] = pixelsY
    user_params['erd'] = erd
    user_params['ers'] = ers
    user_params['eps_min'] = eps_min
    user_params['eps_max'] = eps_max
    user_params['Nlay'] = Nlay
    user_params['L'] = L
    user_params['Lx'] = Lx
    user_params['Ly'] = Ly
    user_params['f'] = f
    user_params['PQ'] = PQ
    user_params['Nx'] = Nx
    user_params['Ny'] = Ny
    user_params['upsample'] = upsample
    user_params['w_l1'] = w_l1
    user_params['w_l2'] = w_l2
    user_params['sigmoid_coeff'] = sigmoid_coeff
    user_params['focal_spot_radius'] = focal_spot_radius
    user_params['enable_random_init'] = enable_random_init
    user_params['initial_height'] = initial_height
    user_params['enable_print'] = enable_print
    user_params['enable_timing'] = enable_timing
    user_params['enable_debug'] = enable_debug
    user_params['enable_logging'] = enable_logging
    user_params['log_filename'] = log_filename
    
    # Set loss function.
    user_params['loss_function'] = loss_function

    # Get start time.
    if enable_timing:
        start_time = time.time()
    
    
    # If we're doing a hyperparameter search...
    if enable_hyperparameter_gridsearch:
        
        # Perform the hyperparameter grid search.
        results = solver_metasurface.hyperparameter_gridsearch(user_params)
        
        # Get list of evaluation scores.
        scores = [r['eval_score'] for r in results]

        # Select hyperparameters corresponding to best evaluation score.
        r = results[np.argmax(scores)]
        h = r['h']
        loss = r['loss']
        focal_plane = r['focal_plane']
        eval_score = r['eval_score']
        params = r['params']

        if enable_print: print('Best hyperparameters: ' + str(r['hyperparameters']))
        if enable_print: print('With evaluation score: ' + str(eval_score))
    
    
    # Otherwise, if we're just doing a single optimization run...
    else:
        
        # Optimize.
        h, loss, params = solver_metasurface.optimize_device(user_params)
    
    if enable_timing:
        print('Completed tests in ' + str(time.time() - start_time) + ' s.')
    
    if use_gpu and enable_gpu_memory_tracking:
        gpu_memory_final = tf_utils.gpu_memory_info()
        gpu_memory_used = [gpu_memory_final[1][0] - gpu_memory_init[1][0],
            gpu_memory_final[1][1] - gpu_memory_init[1][1]]
        print('Memory used on each GPU(MiB): ' + str(gpu_memory_used))
