'''
solver_metasurface_pt.py

Functions implementing optimization algorithm for COPILOT metalens devices,
using PyTorch differentiable implementation of RCWA.

The important user-facing functions here are -

optimize_device: use to optimize a single device for some given parameters.

hyperparameter_gridsearch: use to optimize many devices for a given grid of
   algorithm hyperparameters.
'''

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import itertools
import json
import gc

import solver_pt
import rcwa_utils_pt


def generate_layered_metasurface(h, params):
    '''
    Generates permittivity/permeability for a multilayer metasurface design,
    based on a height representation of the metasurface.

    Args:
        h: A `torch.Tensor` of shape `(pixelsX, pixelsY)` specifying the
        metasurface height at each unit cell. Each entry in this tensor should
        be a float in [0,params['Nlay']-1].
        
        params: A `dict` containing simulation and optimization settings.

    Returns:
        ER_t: A `torch.Tensor` of shape 
        `(batchSize, pixelsX, pixelsY, Nlayer, Nx, Ny)` specifying the relative
        permittivity distribution of the unit cell.

        UR_t: A `torch.Tensor` of shape
        `(batchSize, pixelsX, pixelsY, Nlayer, Nx, Ny)` specifying the relative
        permeability distribution of the unit cell.

    '''

    # Retrieve simulation size parameters.
    batchSize = params['batchSize']
    pixelsX = params['pixelsX']
    pixelsY = params['pixelsY']
    Nlay = params['Nlay']
    Nx = params['Nx']
    Ny = params['Ny']

    # Initialize relative permeability.
    materials_shape = (batchSize, pixelsX, pixelsY, Nlay, Nx, Ny)
    UR_t = params['urd'] * torch.ones(materials_shape)
    
    # Limit optimization range.
    h = torch.clamp(h, min = 0, max = Nlay-1) 
    
    # Convert height representation of to stacked representation.
    z = diff_height_to_stacked(h, params)
    
    # Repeat entries in z so that it has the shape
    # (batchSize, pixelsX, pixelsY, 1, Nx, Ny).
    z = z[None, :, :, :, None, None]
    z = torch.tile(z, (batchSize, 1, 1, 1, Nx, Ny))

    # Build substrate layer and concatenate along the layers dimension.
    layer_shape = (batchSize, pixelsX, pixelsY, 1, Nx, Ny)
    ER_substrate = params['ers'] * torch.ones(layer_shape, dtype = torch.float32)
    ER_t = torch.cat([z, ER_substrate], dim = 3)

    # Cast to complex for subsequent calculations.
    ER_t = ER_t.type(torch.complex64)
    UR_t = UR_t.type(torch.complex64)

    return ER_t, UR_t


def diff_height_to_stacked(h, params):
    '''
    Performs a differentiable transformation from the continuous height
    representation of a metasurface pixel to a continous stacked 
    representation. This is achieved via differential thresholding based on a
    sigmoid function.

    The HEIGHT REPRESENTATION of a metasurface is a tensor containing floats
    representing the material height at each metasurface pixel.

    The STACKED REPRESENTAION of a metasurface is a 3D tensor specifying a
    float relative permittivity of the device at each pixel and on each
    layer. This does not include the substrate layer.

    As params['sigmoid_coeff'] is increased, the thresholding becomes more
    strict, until eventually the metasurface is restricted to be 'admissable' -
    that is, each position in the stacked representation may take on only one
    of the two allowed values.
    
    Args:
        h: A `torch.Tensor` of shape `(pixelsX, pixelsY)` and type float
            containing heights of each pixel.
        
        params: A `dict` containing simulation and optimization settings.
    
    Returns:
        z: A `torch.Tensor` of shape `(pixelsX, pixelsY, Nlay-1)` and type float
            containing relative permittivity of the device at each pixel and on
            each non-substrate layer.

    '''
    
    Nlay = params['Nlay']
    z = torch.stack( [diff_threshold(h, thresh=Nlay-1-i,
                                  coeff=params['sigmoid_coeff'],
                                  offset=Nlay-2-i,
                                  output_scaling = [params['eps_min'],params['eps_max']]) for i in range(Nlay-1) ] )

    return torch.permute(z, [1,2,0])


def diff_threshold(x, coeff=1, thresh=1, offset=0, output_scaling=[0,1]):
    '''
    Performs a differentiable thresholding operation on the input, based on a
    sigmoid funciton.
    
    Args:
        x: Float input to be thresholded. Can be a single number or a tensor
            of any dimension.
        
        coeff: Float coefficient determining steepness of thresholding.
        
        thresh: Float thresholding cutoff, i.e. where in x the step should
            occur.
        
        offset: Float minimum value assumed to occur in x. This value is
            subtracted from x first before the operation is applied, such that
            the sigmoid cutoff occurs halfway between in_offset and thresh.
            
        output_scaling: Float list of length 2 specifying limits to which
            output should be renormalized.
        
        Both offsets should be < thresh, and coeff should be >= 0.
    
    Returns:
        x: Thresholded input.

    '''
    
    x_new = torch.sigmoid(coeff * (x - (offset + (thresh - offset)/2)) )
    x_new = output_scaling[0] + (output_scaling[1] - output_scaling[0]) * x_new
    return x_new


def get_substrate_layer(params):
    '''
    Generates a tensor representing the substrate layer of the device.
    
    Args:
        params: A `dict` containing simulation and optimization settings.
    
    Returns:
        ER_substrate: A `torch.Tensor` of shape
            `(batchSize, pixelsX, pixelsY, 1, Nx, Ny)' specifying the relative
            permittivity distribution of the unit cell in the substrate layer.

    '''
    
    # Retrieve simulation size parameters.
    batchSize = params['batchSize']
    pixelsX = params['pixelsX']
    pixelsY = params['pixelsY']
    Nlay = params['Nlay']
    Nx = params['Nx']
    Ny = params['Ny']
    
    # Build and return substrate layer.
    layer_shape = (batchSize, pixelsX, pixelsY, 1, Nx, Ny)
    ER_substrate = params['ers'] * torch.ones(layer_shape, dtype = torch.float32)
    
    return ER_substrate


def init_layered_metasurface(params, initial_height=0):
    '''
    Generates an initial guess for optimization of a multilayer metasurface.

    The provided guess is a height representation of the metasurface.

    If params['random_init'] == 0, returns a initial guess with zero height
    at all pixels. Otherwise, returns an initial guess with all pixels
    at the height specified by initial_height.
    
    Args:
        params: A `dict` containing simulation and optimization settings.

        initial_height: A float in the range [0, Nlay-1] specifying the an
            initial guess for the height at each pixel.
        
    Returns:
        init: A `np.array of shape `(pixelsX, pixelsY)` specifying an initial
        guess for the height of the metasurface at each pixel.
    
    '''
    
    if params['enable_random_init']:
        init = torch.rand((params['pixelsX'], params['pixelsY']))
        return init * (params['Nlay'] - 1)
    else:
        return torch.ones((params['pixelsX'], params['pixelsY'])) * initial_height


def display_layered_metasurface(ER_t, params):
    '''
    Displays stacked representation of a metasurface.

    Args:
        ER_t: A `torch.Tensor` of shape
        `(batchSize, pixelsX, pixelsY, Nlayer, Nx, Ny)` specifying the relative
        permittivity distribution of the unit cell.
        
        params: A `dict` containing simulation and optimization settings.

    Returns: None

    '''
    
    # Display the permittiviy profile.
    norm = colors.Normalize(vmin=params['eps_min'], vmax=params['eps_max'])
    images=[]
    fig, axes = plt.subplots(params['Nlay'], 1, figsize=(12,12))
    for l in range(params['Nlay']):
        img = torch.permute(torch.squeeze(ER_t[0,:,:,l,:,:]),(0,2,1,3))
        img = torch.real(torch.reshape(img, (params['pixelsX']*params['Nx'],params['pixelsY']*params['Ny'])))
        images.append(axes[l].matshow(img.detach().cpu().numpy(), interpolation='nearest'))
        axes[l].get_xaxis().set_visible(False)
        axes[l].get_yaxis().set_visible(False)
        images[l].set_norm(norm)
        
    plt.show()


def evaluate_solution(focal_plane, params):
    '''
    Generates an evaluation score of a metasurface solution which can be used
    to compare a solution to others.
    
    Args:
        focal_plane: A `torch.Tensor` of shape
            `(batchSize, pixelsX * upsample, pixelsY * upsample)` describing 
            electric field intensity on the focal plane.
    
        params: A `dict` containing simulation and optimization settings.
    
    Returns:
        eval_score: Float evaluation score in range [0, inf).

    '''
    
    r = params['focal_spot_radius']
    index = (params['pixelsX'] * params['upsample']) // 2

    eval_score = torch.sum(torch.abs(focal_plane[0, index-r:index+r, index-r:index+r]) )

    return float(eval_score)


def optimize_device(user_params):
    '''
    Produces an optimized layered metasurface design for some given device and
    optimization parameters.

    Args:
        user_params: A `dict` containing simulation and optimization settings.
            As opposed to dicts named simply 'params' elsewhere in this code,
            'user_params' contains only parameters which are able to be
            directly configured by a user, and not those derived parameters
            calculated by the RCWA solver.

    Returns:
        h: A `torch.Tensor` of shape `(pixelsX, pixelsY)` and type float
            containing heights of each pixel in the optimized design.

        loss: A 'torch.Tensor' of shape `(N+1)` and type float containing
            containing calculated loss at each optimization iteration.

        params: A `dict` containing simulation and optimization settings.
            The same as the provided user_params, but also contains
            derived parameters calculated by the RCWA solver.
            
        focal_plane: A `torch.Tensor` of shape `(batchSize, params['upsample'] * pixelsX,
            params['upsample'] * pixelsY)` and dtype `torch.complex64` specifying the 
            the electric fields at the output plane.

    '''
    
    params = solver_pt.initialize_params(wavelengths=user_params['wavelengths'],
                                  thetas=user_params['thetas'],
                                  phis=user_params['phis'],
                                  pte=user_params['pte'],
                                  ptm=user_params['ptm'],
                                  pixelsX=user_params['pixelsX'],
                                  pixelsY=user_params['pixelsY'],
                                  erd=user_params['erd'],
                                  ers=user_params['ers'],
                                  PQ=user_params['PQ'],
                                  Lx=user_params['Lx'],
                                  Ly=user_params['Ly'],
                                  L=user_params['L'],
                                  Nx=16,
                                  eps_min=1.0,
                                  eps_max=user_params['erd'])

    # Merge with the user-provided parameter dictionary.
    params['N'] = user_params['N']
    params['sigmoid_coeff'] = user_params['sigmoid_coeff']
    params['sigmoid_update'] = user_params['sigmoid_update']
    params['learning_rate'] = user_params['learning_rate']
    params['focal_spot_radius'] = user_params['focal_spot_radius']
    params['enable_random_init'] = user_params['enable_random_init']
    params['initial_height'] = user_params['initial_height']
    params['enable_debug'] = user_params['enable_debug']
    params['enable_print'] = user_params['enable_print']
    params['enable_logging'] = user_params['enable_logging']
    params['log_filename_prefix'] = user_params['log_filename_prefix']
    params['log_filename_extension'] = user_params['log_filename_extension']
    params['parameter_string'] = user_params['parameter_string']
    params['loss_function'] = user_params['loss_function']

    # Define the free-space propagator and input field distribution
    # for the metasurface.
    params['f'] = user_params['f'] * 1E-9
    params['upsample'] = user_params['upsample']
    params['propagator'] = solver_pt.make_propagator(params, params['f'])
    params['input'] = solver_pt.define_input_fields(params)

    # Get the initial height representation of the metasurface.
    h = torch.autograd.Variable(init_layered_metasurface(params, initial_height=params['initial_height']),
                                requires_grad=True)

    # Define optimizer.
    opt = torch.optim.Adam([h], lr=params['learning_rate'])
    loss = np.zeros(params['N']+1)

    # Optimize.
    if params['enable_print']: print('Optimizing... ', end="")
    for i in range(params['N']):
        if params['enable_print']: print(str(i) + ', ', end="")
        opt.zero_grad()
        l = params['loss_function'](h, params)
        l.backward()
        opt.step()
        loss[i] = l

        # Anneal sigmoid coefficient.
        params['sigmoid_coeff'] += (params['sigmoid_update'] / params['N'])
    if params['enable_print']: print('Done.')

    # Round off to a final, admissable, solution.
    # Do a final range clip.
    h = torch.clamp(h, min=0, max=params['Nlay']-1)

    # Round heights to nearest integer.
    h = torch.round(h)

    # Get final loss.
    loss[-1] = params['loss_function'](h, params)
    
    # Get scattering pattern of final solution.
    ER_t, UR_t = generate_layered_metasurface(h, params)
    outputs = solver_pt.simulate(ER_t, UR_t, params)
    field = outputs['ty'][:, :, :, np.prod(params['PQ']) // 2, 0]
    focal_plane = solver_pt.propagate(params['input'] * field, params['propagator'], params['upsample'])
    
    # Get the evaluation score of the resulting solution.
    eval_score = evaluate_solution(focal_plane, params)

    # Log result.
    if params['enable_logging']:
        hp_names = ['N', 'sigmoid_update', 'learning_rate', 'initial_height']
        hyperparams = [user_params[name] for name in hp_names]
        result = {'hyperparameter_names': list(hp_names),
            'hyperparameters': hyperparams,
            'h': h,
            'loss': loss,
            'focal_plane': focal_plane,
            'eval_score': eval_score,
            'params': params }
        
        log_result(result, params['log_filename_prefix'] + params['parameter_string'] + user_params['log_filename_extension'])
    
    return h, loss, params, focal_plane


def hyperparameter_gridsearch(user_params):
    '''
    Runs a grid search for good hyperparameters for layered metasurface
    optimization.

    Args:
        user_params: A `dict` containing simulation and optimization settings.
            As opposed to dicts named simply 'params' elsewhere in this code,
            'user_params' contains only parameters which are able to be
            directly configured by a user.

            Specifically, the entry param_grid is used to configure ranges for
            the grid search.

    Returns:
        results: A list of `dict`s, each of which corresponds to one
            run of metasurface optimization for some selected hyperparameters.
            Each dict contains a list of the used hyperparameters, as well as
            the height representation of the reulting metasurface, evaluation
            score assigned to that metasurface, the loss curve for the
            optimization run, and the focal plane scatter pattern produced
            by the optimized device.
    '''
    
    # Allocate list of results.
    # Each entry is a dictionary containing the list of hyperparameters used,
    # height representation of the resulting metasurface, focal plane intensity
    # pattern produced by the metasurface, evaluation score of the metasurface,
    # and list of optimization losses for that run.
    results = []
    
    # Get dimensions of grid.
    hp_grid = user_params['param_grid'].values()
    hp_names = user_params['param_grid'].keys()
    
    if user_params['enable_print']: print('Beginning hyperparameter grid search...')
    
    # Iterate over the grid.
    for hyperparams in itertools.product(*hp_grid):
        
        if user_params['enable_print']: print('\nTrying hyperparameters: ' + str(list(hp_names)))
        if user_params['enable_print']: print(hyperparams)

        # Update parameter list dict with selected parameters.
        for i, name in enumerate(hp_names):
            user_params[name] = hyperparams[i]
            
        # Update log file name.
        hyperparameter_string = '-'.join([h + str(v) for (h,v) in zip(hp_names,hyperparams)])
        user_params['parameter_string'] = hyperparameter_string

        # Run optimization with selected parameters.
        h, loss, params, focal_plane = optimize_device(user_params)
        
        # Get the evaluation score of the resulting solution.
        eval_score = evaluate_solution(focal_plane, params)
        
        # Save result.
        result = {'hyperparameter_names': list(hp_names),
            'hyperparameters': hyperparams,
            'h': h,
            'loss': loss,
            'focal_plane': focal_plane,
            'eval_score': eval_score,
            'params': params }
        results.append(result)

    return results

                           
def log_result(result, log_filename):
    '''
    Writes the result of a single optimization run to an output file.
    
    Args:
        result: A dict which corresponds to one run of metasurface optimization for 
            some selected hyperparameters. It should contain the keys:
            
            hyperparameter_names: A list of string names of the hyperparameters being optimized.
            hyperparameters: A list of hyperparameter values used for this run, corresponding to
                those in hyperparameter_names.
               
            h: A list of shape `(pixelsX, pixelsY)` and type float
                containing heights of each pixel in the optimized design.

            loss: A list of shape `(N+1)` and type float containing
                containing calculated loss at each optimization iteration.
                
            focal_plane: A list of shape `(batchSize, params['upsample'] * pixelsX,
                params['upsample'] * pixelsY)` and dtype float specifying the 
                the real part of electric fields at the output plane.
            
            eval_score: Float evaluation score of the solution in range [0, inf).

            params: A `dict` containing simulation and optimization settings.
        
        log_filename: A string specifying the relative path of the file to write result to.
            File is created if it does not exist and overwritten if it does.

    Returns:
        None
    '''
    
    # Open log file in write mode.
    with open(log_filename, 'w', encoding="utf-8") as f:
        
        # Get json representation of results dict and write to log file.
        json.dump(make_result_loggable(result), f)

        
def make_result_loggable(result):
    '''
    Prepares the results dictionary of an optimization run for writing to an output file.
    
    Args:
        result: A dict which corresponds to one run of metasurface optimization for 
            some selected hyperparameters, structured as in log_result, except that
            h, loss, and focal_plane may be 'torch.Tensor's.

    Returns:
        loggable_result: A dict with the same contents as result but with element
            types converted such that it can be passed to log_result.
    '''
    
    # Modify result dict to only include necessary elements
    # and ensure that they are all json serializable.
    loggable_result = {'hyperparameter_names': result['hyperparameter_names'],
                       'hyperparameters': result['hyperparameters'],
                       'h': result['h'].detach().cpu().numpy().tolist(),
                       'loss': result['loss'].tolist(),
                       'focal_plane': result['focal_plane'].type(torch.float32).detach().cpu().numpy().tolist(),
                       'eval_score': result['eval_score'] }
    
    return loggable_result


def load_result(log_filename):
    '''
    Read results of an optimization run from a file.
    
    Args:
        log_filename: A string specifying the relative path to the output file to be read.
    
    Returns:
        result: A dict containing all information about the recorded optimization run,
            structured as in log_result.
    '''
    
    # Open log file in read mode.
    with open(log_filename, 'r', encoding="utf-8") as f:

        # Read json representation of results from log file.
        result = json.load(f)
        result['h'] = torch.tensor(result['h'], dtype=torch.float32)
        result['loss'] = np.array(result['loss'])
        result['focal_plane'] = torch.tensor(result['focal_plane'], dtype=torch.float32)
        return result
