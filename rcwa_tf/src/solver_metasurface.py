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
        at each unit cell. Each entry in this tensor should be a float in [1,params['erd']].
        
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
    z = z[tf.newaxis, :, :, :, tf.newaxis, tf.newaxis]
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


def generate_copilot_metasurface_multilayer(z, device, params):
    '''
    Generates permittivity/permeability for a discrete, multilayer metasurface design.
    The metasurface either has material or not at each grid position, and a substrate layer underneath.

    Version 1: Top layer of design is treated separately because it is the only layer being optimized.
    

    Args:
        z: A `tf.Tensor` of shape `(pixelsX, pixelsY, 1)` specifying the relative permittivity
        at each unit cell. Each entry in this tensor should be a float in [1,params['erd']]. This is
        the top layer of the design, which is currenlty being optimized.
        
        device: A `tf.Tensor` of shape `(batchsize, pixelsX, pixelsY, Nlay-1, Nx, Ny)` specifying the 
        relative permittivity at each unit cell. This is the portion of the design which is not being 
        optimized - in the first iteration, only the substrate layer.
        
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

    # Repeat entries in z so that it has the shape (batchSize, pixelsX, pixelsY, Nlay-1, Nx, Ny)
    z = z[tf.newaxis, :, :, :, tf.newaxis, tf.newaxis]
    z = tf.tile(z, multiples = (batchSize, 1, 1, 1, Nx, Ny))

    # Build substrate layer and concatenate along the layers dimension.
    #layer_shape = (batchSize, pixelsX, pixelsY, 1, Nx, Ny)
    #ER_substrate = params['ers'] * tf.ones(layer_shape, dtype = tf.float32)
    ER_t = tf.concat(values = [z, device], axis = 3)

    # Cast to complex for subsequent calculations.
    ER_t = tf.cast(ER_t, dtype = tf.complex64)
    UR_t = tf.convert_to_tensor(UR, dtype = tf.float32)
    UR_t = tf.cast(UR_t, dtype = tf.complex64)
    
    return ER_t, UR_t


def generate_copilot_metasurface(h, params):
    '''
    Generates permittivity/permeability for a discrete, single-layer metasurface design.
    The metasurface either has material or not at each grid position, and a substrate layer underneath.

    Version 3: Uses a height representation of the surface, where h contains float heights
    representing the material height at each metasurface pixel. A differentiable thresholding step limits
    these heights and converts them to the stacked representation.

    Args:
        h: A `tf.Tensor` of shape `(pixelsX, pixelsY)` specifying the metasurface height
        at each unit cell. Each entry in this tensor should be a float in [0,params['Nlay']-1].
        
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
    
    # Limit optimization range.
    h = tf.clip_by_value(h, clip_value_min = 0, clip_value_max = Nlay-1)
    
    # Convert height representation of the metasurface to stacked representation.
    z = diff_height_to_stacked(h, params)
    
    # Repeat entries in z so that it has the shape (batchSize, pixelsX, pixelsY, 1, Nx, Ny)
    z = z[tf.newaxis, :, :, :, tf.newaxis, tf.newaxis]
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


def get_substrate_layer(params):
    '''
    Generates a tensor representing the substrate layer of the device.
    
    Args:
        params: A `dict` containing simulation and optimization settings.
    
    Returns:
        ER_substrate: A `tf.Tensor` of shape `(batchSize, pixelsX, pixelsY, 1, Nx, Ny)`
        specifying the relative permittivity distribution of the unit cell in the
        substrate layer.
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
    ER_substrate = params['ers'] * tf.ones(layer_shape, dtype = tf.float32)
    
    return ER_substrate


def init_copilot_metasurface(params, single_layer=False):
    '''
    Generates an initial guess for the metasurface to be optimized.
    If params['random_init'] == 0, returns a initial guess of a metasurface with relative permittivity 
    equal to params['eps_min'] at all pixels. Otherwise, randomly initializes each pixel's permittivity 
    to some value in [params['eps_min'],params['eps_max']].
    
    Args:
        params: A `dict` containing simulation and optimization settings.
        
    Returns:
        init: A `tf.Tensor` of shape `(pixelsX, pixelsY, Nlay-1)` specifying an initial guess for the relative
        permittivity of the metasurface at each pixel and non-substrate layer.
    
    '''
    
    layers = 1 if single_layer else params['Nlay']-1
    
    if params['random_init']:
        init = np.random.rand(params['pixelsX'], params['pixelsY'], layers)
        return init * (params['eps_max'] - params['eps_min']) + params['eps_min']
    else:
        return params['eps_min'] * np.ones(shape=(params['pixelsX'], params['pixelsY'], layers))


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

    
def pixel_to_stacked(z, params, add_substrate=True, cast=True):
    
    # Retrieve simulation size parameters.
    batchSize = params['batchSize']
    pixelsX = params['pixelsX']
    pixelsY = params['pixelsY']
    Nlay = params['Nlay']
    Nx = params['Nx']
    Ny = params['Ny']
    
    # Initialize relative permeability.
    materials_shape = (batchSize, pixelsX, pixelsY, Nlay, Nx, Ny)
    UR_t = params['urd'] * np.ones(materials_shape)
    UR_t = tf.convert_to_tensor(UR_t, dtype = tf.float32)
    
    # Repeat entries in z so that it has the shape (batchSize, pixelsX, pixelsY, Nlay-1, Nx, Ny)
    ER_t = z[tf.newaxis, :, :, :, tf.newaxis, tf.newaxis]
    ER_t = tf.tile(ER_t, multiples = (batchSize, 1, 1, 1, Nx, Ny))
    
    if (add_substrate):
        # Build substrate layer and concatenate along the layers dimension.
        layer_shape = (batchSize, pixelsX, pixelsY, 1, Nx, Ny)
        ER_substrate = params['ers'] * tf.ones(layer_shape, dtype = tf.float32)
        ER_t = tf.concat(values = [ER_t, ER_substrate], axis = 3)

    if (cast):
        # Cast to complex for subsequent calculations.
        ER_t = tf.cast(ER_t, dtype = tf.complex64)
        UR_t = tf.cast(UR_t, dtype = tf.complex64)
    
    return ER_t, UR_t


def diff_threshold(x, coeff=1, thresh=1, offset=0, output_scaling=[0,1]):
    '''
    Performs a differentiable thresholding operation on the input, based on a sigmoid funciton.
    
    Args:
        x: Float input to be thresholded. Can be a single number or a tensor of any dimension.
        
        coeff: Float coefficient determining steepness of thresholding.
        
        thresh: Float thresholding cutoff, i.e. where in x the step should occur.
        
        offset: Float minimum value assumed to occur in x. This value is subtracted from x
            first before the operation is applied, such that the sigmoid cutoff occurs
            halfway between in_offset and thresh.
            
        output_scaling: Float list of length 2 specifying limits to which output should
            be renormalized.
        
        Both offsets should be < thresh, and coeff should be >= 0.
    
    Returns:
        x: Thresholded input.
    '''
    
    x_new = tf.math.sigmoid(coeff * (x - (offset + (thresh - offset)/2)) )
    x_new = output_scaling[0] + (output_scaling[1] - output_scaling[0]) * x_new
    return x_new


def diff_height_to_stacked(h, params):
    '''
    Performs a differentiable transformation from the continuous height representation of a
    metasurface pixel (i.e., a float representing the height of the surface at that pixel) to a
    continous stacked representation (i.e., a list of floats representing relative permittivity of
    the surface at each layer, at that pixel). Based on a differential thresholding using a
    sigmoid function.
    
    Args:
        h: Float height of the pixel. Can also be a tensor of any dimension.
        
        params: A `dict` containing simulation and optimization settings.
    
    Returns:
        z: Stacked representation of the pixel.
    '''
    
    #scaling = [params['eps_min'], params['eps_max']]
    
    #z = tf.stack([diff_threshold(h, thresh=3, coeff=params['sigmoid_coeff'], offset=2, output_scaling=scaling),
    #              diff_threshold(h, thresh=2, coeff=params['sigmoid_coeff'], offset=1, output_scaling=scaling),
    #              diff_threshold(h, thresh=1, coeff=params['sigmoid_coeff'], offset=0, output_scaling=scaling)],
    #            )
    
    Nlay = params['Nlay']
    z = tf.stack( [diff_threshold(h, thresh=Nlay-1-i,
                                  coeff=params['sigmoid_coeff'],
                                  offset=Nlay-2-i,
                                  output_scaling = [params['eps_min'],params['eps_max']]) for i in range(Nlay-1) ] )

    return tf.transpose(z, perm=[1,2,0])


def evaluate_solution(focal_plane, params):
    '''
    Generates an evaluation score of a metasurface solution which can be used
    to compare this solution to others.
    
    Args:
        focal_plane: A `tf.Tensor` of shape `(batchSize, pixelsX * upsample, pixelsY * upsample)` describing 
            electric field intensity on the focal plane.
    
        params: A `dict` containing simulation and optimization settings.
    
    Returns:
        eval: Float evaluation score in range [0, inf).
    '''
    
    r = params['focal_spot_radius']
    index = (params['pixelsX'] * params['upsample']) // 2
    return tf.math.reduce_sum(tf.abs(focal_plane[0, index-r:index+r, index-r:index+r])).numpy()

        
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
    