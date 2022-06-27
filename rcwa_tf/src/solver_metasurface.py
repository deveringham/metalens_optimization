import tensorflow as tf
import numpy as np
import rcwa_utils
import tensor_utils
import matplotlib.pyplot as plt
from matplotlib import colors


def generate_copilot_metasurface_singlelayer(z, params):
    '''
    Generates permittivity/permeability for a discrete, single-layer metasurface design.
    The metasurface either has material or not at each grid position, and a substrate layer underneath.

    Version 2: Uses a simplified representation of the surface, where z contains not float heights
    but rather a float relative permittivity at each position. Regularization terms in the
    loss function and/or differentiable thresholding in the optimization step enforce that these
    adhere arbitrary floats converge to either one of two admissable values.

    Args:
        z: A `tf.Tensor` of shape `(pixelsX, pixelsY)` specifying the relative permittivity
        at each unit cell. Each entry in this tensor should be an float in [1,params['erd']].
        
        params: A `dict` containing simulation and optimization settings.

    Returns:
        ER_t: A `tf.Tensor` of shape `(batchSize, pixelsX, pixelsY, Nlayer, Nx, Ny)`
        specifying the relative permittivity distribution of the unit cell.

        UR_t: A `tf.Tensor` of shape `(batchSize, pixelsX, pixelsY, Nlayer, Nx, Ny)`
        specifying the relative permeability distribution of the unit cell.
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
    UR = params['urd'] * np.ones(materials_shape)
    
    # Sigmoid pass.
    if params['thresholding_enabled']:
        z = tf.math.sigmoid(params['sigmoid_coeff'] * (z - (1 + (params['erd']-1)/2)) )
        z = 1 + (params['erd'] - 1) * z

    # Limit the optimization ranges.
    z = tf.clip_by_value(z, clip_value_min = params['eps_min'], clip_value_max = params['eps_max'])

    # Repeat entries in z so that it has the shape (batchSize, pixelsX, pixelsY, 1, Nx, Ny)
    z = z[tf.newaxis, :, :, tf.newaxis, tf.newaxis, tf.newaxis]
    z = tf.tile(z, multiples = (batchSize, 1, 1, 1, Nx, Ny))

    # Build substrate layer and concatenate along the layers dimension.
    layer_shape = (batchSize, pixelsX, pixelsY, 1, Nx, Ny)
    ER_substrate = params['ers'] * tf.ones(layer_shape, dtype = tf.float32)
    ER_t = tf.concat(values = [z, ER_substrate], axis = 3)

    # Cast to complex for subsequent calculations.
    ER_t = tf.cast(ER_t, dtype = tf.complex64)
    UR_t = tf.convert_to_tensor(UR, dtype = tf.float32)
    UR_t = tf.cast(UR_t, dtype = tf.complex64)

    return ER_t, UR_t


def init_copilot_metasurface(params):
    '''
    Generates an initial guess for the metasurface to be optimized.
    If params['random_init'] == 0, returns a initial guess of a metasurface with relative permittivity 
    equal to params['eps_min'] at all pixels. Otherwise, randomly initializes each pixel's permittivity 
    to some value in [params['eps_min'],params['eps_max']].
    '''
    
    if params['random_init']:
        init = np.random.rand(params['pixelsX'], params['pixelsY'])
        return init * (params['eps_max'] - params['eps_min']) + params['eps_min']
    else:
        return params['eps_min'] * np.ones(shape=(params['pixelsX'], params['pixelsY']))


def display_metasurface(ER_t, params):
    '''
    Displays stacked layer representation of a metasurface, in terms of its relative permittivity
    at each grid location.

    Args:
        ER_t: A `tf.Tensor` of shape `(batchSize, pixelsX, pixelsY, Nlayer, Nx, Ny)`
        specifying the relative permittivity distribution of the unit cell.
        
        params: A `dict` containing simulation and optimization settings.

    Returns: None
    '''
    
    # Display the permittiviy profile.
    norm = colors.Normalize(vmin=params['eps_min'], vmax=params['eps_max'])
    images=[]
    fig, axes = plt.subplots(params['Nlay'], 1, figsize=(12,12))
    for l in range(params['Nlay']):
        img = tf.transpose(tf.squeeze(ER_t[0,:,:,l,:,:]),[0,2,1,3])
        img = tf.math.real(tf.reshape(img, (params['pixelsX']*params['Nx'],params['pixelsY']*params['Ny'])))
        images.append(axes[l].matshow(img, interpolation='nearest'))
        images[l].set_norm(norm)
        
    fig.colorbar(images[0], ax=axes, orientation='horizontal', fraction=.1)
    plt.show()

    
def pixel_to_stacked(z, params):
    
    # Retrieve simulation size parameters.
    batchSize = params['batchSize']
    pixelsX = params['pixelsX']
    pixelsY = params['pixelsY']
    Nlay = params['Nlay']
    Nx = params['Nx']
    Ny = params['Ny']

    # Initialize relative permeability.
    materials_shape = (batchSize, pixelsX, pixelsY, Nlay, Nx, Ny)
    UR = params['urd'] * np.ones(materials_shape)
    
    # Repeat entries in z so that it has the shape (batchSize, pixelsX, pixelsY, 1, Nx, Ny)
    z = z[tf.newaxis, :, :, tf.newaxis, tf.newaxis, tf.newaxis]
    z = tf.tile(z, multiples = (batchSize, 1, 1, 1, Nx, Ny))

    # Build substrate layer and concatenate along the layers dimension.
    layer_shape = (batchSize, pixelsX, pixelsY, 1, Nx, Ny)
    ER_substrate = params['ers'] * tf.ones(layer_shape, dtype = tf.float32)
    ER_t = tf.concat(values = [z, ER_substrate], axis = 3)

    # Cast to complex for subsequent calculations.
    ER_t = tf.cast(ER_t, dtype = tf.complex64)
    UR_t = tf.convert_to_tensor(UR, dtype = tf.float32)
    UR_t = tf.cast(UR_t, dtype = tf.complex64)
    
    return ER_t, UR_t


# Old versions

'''
def generate_copilot_metasurface(z, params):
    ''
    Generates permittivity/permeability for a discrete, multilayered metasurface design.
    The metasurface has a discrete height at each grid position, and a substrate layer underneath.

    Version 1: 
    pixelsX and pixelsY give the number of unit cells in the design. Each unit cell comprises a single
    'pixel' of the discrete metasurface, and is characterized by a single height.

    Args:
        z: A `tf.Tensor` of shape `(pixelsX, pixelsY)` specifying the metasurface height
        at each unit cell. Each entry in this tensor should be an float in [0,Nlay-2].
        During construction of the metasurface, these float heights are rounded off to the nearest
        integer in [0,Nlay-2]. The physical thickness of these layers is configured in params.

        params: A `dict` containing simulation and optimization settings.

    Returns:
        ER_t: A `tf.Tensor` of shape `(batchSize, pixelsX, pixelsY, Nlayer, Nx, Ny)`
        specifying the relative permittivity distribution of the unit cell.

        UR_t: A `tf.Tensor` of shape `(batchSize, pixelsX, pixelsY, Nlayer, Nx, Ny)`
        specifying the relative permeability distribution of the unit cell.
    ''

    # Debug using GradientTape
    #with tf.GradientTape() as tape:

    # Retrieve simulation size parameters.
    batchSize = params['batchSize']
    pixelsX = params['pixelsX']
    pixelsY = params['pixelsY']
    Nlay = params['Nlay']
    Nx = params['Nx']
    Ny = params['Ny']
    Lx = params['Lx']
    Ly = params['Ly']

    # Initialize relative permeability.
    materials_shape = (batchSize, pixelsX, pixelsY, Nlay, Nx, Ny)
    UR = params['urd'] * np.ones(materials_shape)

    # Limit the optimization ranges.
    z = tf.clip_by_value(z, clip_value_min = 0, clip_value_max = Nlay-1)

    # Round the pixel heights to the nearest integer.
    z = tf.math.round(z)

    # Repeat entries in z so that it has the shape (batchSize, pixelsX, pixelsY, Nlay-1, Nx, Ny)
    z = z[tf.newaxis, :, :, tf.newaxis, tf.newaxis, tf.newaxis]
    z = tf.tile(z, multiples = (batchSize, 1, 1, Nlay-1, Nx, Ny))

    # Clever shit.
    z_off = tf.range(start=Nlay-1, limit=0, delta=-1, dtype=tf.float32)
    z_off = z_off[tf.newaxis, tf.newaxis, tf.newaxis, :, tf.newaxis, tf.newaxis]
    z_off = tf.tile(z_off, multiples = (batchSize, pixelsX, pixelsY, 1, Nx, Ny))
    z_lay = tf.clip_by_value(z_off - z, clip_value_min = 0, clip_value_max = 1)

    # Define the cartesian cross section.
    dx = Lx / Nx # grid resolution along x
    dy = Ly / Ny # grid resolution along y
    xa = np.linspace(0, Nx - 1, Nx) * dx # x axis array
    xa = xa - np.mean(xa) # center x axis at zero
    ya = np.linspace(0, Ny - 1, Ny) * dy # y axis vector
    ya = ya - np.mean(ya) # center y axis at zero
    [y_mesh, x_mesh] = np.meshgrid(ya,xa)

    # Convert to tensors and expand and tile to match the simulation shape.
    y_mesh = tf.convert_to_tensor(y_mesh, dtype = tf.float32)
    y_mesh = y_mesh[tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, :, :]
    y_mesh = tf.tile(y_mesh, multiples = (batchSize, pixelsX, pixelsY, 1, 1, 1))
    x_mesh = tf.convert_to_tensor(x_mesh, dtype = tf.float32)
    x_mesh = x_mesh[tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, :, :]
    x_mesh = tf.tile(x_mesh, multiples = (batchSize, pixelsX, pixelsY, 1, 1, 1))

    # Magic from solver.generate_rectangular_resonators.
    #r = z_lay * Nx / params['Lx']
    
    # Find zeros
    #r1 = 1 - tf.abs((x_mesh * r) - (y_mesh * r)) - tf.abs((x_mesh * r) + (y_mesh * r))

    # Build device layer.
    #ER_r1 = tf.math.sigmoid(params['sigmoid_coeff'] * r1)
    #ER_t = 1 + (params['erd'] - 1) * ER_r1
    
    # Find argument to sigmoid function.
    sigmoid_arg = z - 0.5

    # Build device layer.
    ER_r1 = tf.math.sigmoid(params['sigmoid_coeff'] * sigmoid_arg)
    ER_t = 1 + (params['erd'] - 1) * ER_r1

    # Build substrate layer and concatenate along the layers dimension.
    layer_shape = (batchSize, pixelsX, pixelsY, 1, Nx, Ny)
    ER_substrate = params['ers'] * tf.ones(layer_shape, dtype = tf.float32)
    ER_t = tf.concat(values = [ER_t, ER_substrate], axis = 3)

    # Cast to complex for subsequent calculations.
    ER_t = tf.cast(ER_t, dtype = tf.complex64)
    UR_t = tf.convert_to_tensor(UR, dtype = tf.float32)
    UR_t = tf.cast(UR_t, dtype = tf.complex64)

    #print('Gradient (in generate_copilot_metasurface): '  + str( tape.gradient(ER_t, z) ))

    return ER_t, UR_t
'''

'''
def generate_copilot_metasurface_singlelayer(z, params):
    ''
    Generates permittivity/permeability for a discrete, single-layer metasurface design.
    The metasurface either has material or not at each grid position, and a substrate layer underneath.

    Version 1: 
    pixelsX and pixelsY give the number of unit cells in the design. Each unit cell comprises a single
    'pixel' of the discrete metasurface, and is characterized by a single bool (material/empty).

    Args:
        z: A `tf.Tensor` of shape `(pixelsX, pixelsY)` specifying the metasurface height
        at each unit cell. Each entry in this tensor should be an float in [0,1].
        During construction of the metasurface, these float heights are rounded off to the nearest
        integer in [0,1].

        params: A `dict` containing simulation and optimization settings.

    Returns:
        ER_t: A `tf.Tensor` of shape `(batchSize, pixelsX, pixelsY, Nlayer, Nx, Ny)`
        specifying the relative permittivity distribution of the unit cell.

        UR_t: A `tf.Tensor` of shape `(batchSize, pixelsX, pixelsY, Nlayer, Nx, Ny)`
        specifying the relative permeability distribution of the unit cell.
    ''

    # Debug using GradientTape
    with tf.GradientTape() as tape:
        
        #print('Gradient of z wrt z: '  + str( tape.gradient(z, z) ))

        # Retrieve simulation size parameters.
        batchSize = params['batchSize']
        pixelsX = params['pixelsX']
        pixelsY = params['pixelsY']
        Nlay = params['Nlay']
        Nx = params['Nx']
        Ny = params['Ny']
        Lx = params['Lx']
        Ly = params['Ly']

        # Initialize relative permeability.
        materials_shape = (batchSize, pixelsX, pixelsY, Nlay, Nx, Ny)
        UR = params['urd'] * np.ones(materials_shape)

        # Limit the optimization ranges.
        z = tf.clip_by_value(z, clip_value_min = 0, clip_value_max = Nlay-1)

        # Repeat entries in z so that it has the shape (batchSize, pixelsX, pixelsY, 1, Nx, Ny)
        z = z[tf.newaxis, :, :, tf.newaxis, tf.newaxis, tf.newaxis]
        z = tf.tile(z, multiples = (batchSize, 1, 1, 1, Nx, Ny))

        # Use sigmoid to 'round off' z into pixel regions.
        sigmoid_arg = z - 0.5
        
        #print('Gradient of sigmoid_arg wrt z: '  + str( tape.gradient(sigmoid_arg, z) ))
        
        ER_r1 = tf.math.sigmoid(params['sigmoid_coeff'] * sigmoid_arg)
        
        #print('Gradient of ER_r1 wrt z: '  + str( tape.gradient(ER_r1, z) ))
        
        ER_t = 1 + (params['erd'] - 1) * ER_r1
        
        #print('Gradient of ER_t wrt z: '  + str( tape.gradient(ER_t, z) ))

        # Build substrate layer and concatenate along the layers dimension.
        layer_shape = (batchSize, pixelsX, pixelsY, 1, Nx, Ny)
        ER_substrate = params['ers'] * tf.ones(layer_shape, dtype = tf.float32)
        ER_t = tf.concat(values = [ER_t, ER_substrate], axis = 3)

        # Cast to complex for subsequent calculations.
        ER_t = tf.cast(ER_t, dtype = tf.complex64)
        UR_t = tf.convert_to_tensor(UR, dtype = tf.float32)
        UR_t = tf.cast(UR_t, dtype = tf.complex64)

        #print('Gradient of ER_t wrt z: '  + str( tape.gradient(ER_t, z) ))

    return ER_t, UR_t
'''
    