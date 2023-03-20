'''
solver_pt.py

Core implementation of differentiable RCWA, using PyTorch to take
advantage of automatic differentiation.

Contains all functions for configuration of input parameters,
simulation of fields within device, and propagation of fields
away from its transmission surface.

This file is a modified version of the original by
Copyright (c) 2020 Shane Colburn, University of Washington

Modifications made by Dylan Everingham, 2022
for performance optimization and conversion to use PyTorch.
'''

import torch
import numpy as np
import rcwa_utils_pt
import json
from torch.utils.checkpoint import checkpoint

def initialize_params(wavelengths = [632.0],
                      thetas = [0.0],
                      phis = [0.0],
                      pte = [1.0],
                      ptm = [0.0],
                      pixelsX = 1,
                      pixelsY = 1,
                      erd = 6.76,
                      ers = 2.25,
                      PQ = [11, 11],
                      Lx = 0.7 * 632.0,
                      Ly = 0.7 * 632.0,
                      L = [632.0, 632.0],
                      Nx = 512,
                      eps_min = 1.0,
                      eps_max = 12.11,
                      blur_radius = 100.0):
  '''
    Initializes simulation parameters and hyperparameters.
    Args:
        wavelengths: A `list` of dtype `float` and length `batchSize` specifying
        the set of wavelengths over which to optimize.

        thetas: A `list` of dtype `float` and length `batchSize` specifying
        the set of polar angles over which to optimize.

        phis: A `list` of dtype `float` and length `batchSize` specifying the 
        set of azimuthal angles over which to optimize.

        pte: A `list` of dtype `float` and length `batchSize` specifying the set
        of TE polarization component magnitudes over which to optimize. A 
        magnitude of 0.0 means no TE component. Under normal incidence, the TE 
        polarization is parallel to the y-axis.

        ptm: A `list` of dtype `float` and length `batchSize` specifying the set
        of TM polarization component magnitudes over which to optimize. A 
        magnitude of 0.0 means no TM component. Under normal incidence, the TM 
        polarization is parallel to the x-axis.

        pixelsX: An `int` specifying the x dimension of the metasurface in
        pixels that are of width `params['Lx']`.

        pixelsY: An `int` specifying the y dimension of the metasurface in
        pixels that are of width `params['Ly']`.

        erd: A `float` specifying the relative permittivity of the non-vacuum,
        constituent material of the device layer for shape optimizations.

        ers: A `float` specifying the relative permittivity of the substrate
        layer.

        PQ: A `list` of dtype `int` and length 2 specifying the number of 
        Fourier harmonics in the x and y directions. The numbers should be odd
        values.

        Lx: A `float` specifying the unit cell pitch in the x direction in
        micrometers.

        Ly: A `float` specifying the unit cell pitch in the y direction in
        micrometers.

        L: A `list` of dtype `float` specifying the layer thicknesses in 
        micrometers.

        Nx: An `int` specifying the number of sample points along the x 
        direction in the unit cell.

        eps_min: A `float` specifying the minimum allowed permittivity for 
        topology optimizations.

        eps_max: A `float` specifying the maximum allowed permittivity for 
        topology optimizations.

        blur_radius: A `float` specifying the radius of the blur function with 
        which a topology optimized permittivity density should be convolved.
        
    Returns:
        params: A `dict` containing simulation and optimization settings.
  '''

  # Define the `params` dictionary.
  params = dict({})

  # Units and tensor dimensions.
  params['micrometers'] = 1E-6
  params['degrees'] = np.pi / 180
  params['batchSize'] = len(wavelengths)
  params['pixelsX'] = pixelsX
  params['pixelsY'] = pixelsY
  params['Nlay'] = len(L)

  # Simulation tensor shapes.
  batchSize = params['batchSize']

  # Batch parameters (wavelength, incidence angle, and polarization).
  lam0 = params['micrometers'] * torch.tensor(wavelengths, dtype=torch.float32)
  lam0 = lam0[:, None, None, None, None, None]
  lam0 = torch.tile(lam0, (1, pixelsX, pixelsY, 1, 1, 1))
  params['lam0'] = lam0

  theta = params['degrees'] * torch.tensor(thetas, dtype=torch.float32)
  theta = theta[:, None, None, None, None, None]
  theta = torch.tile(theta, (1, pixelsX, pixelsY, 1, 1, 1))
  params['theta'] = theta

  phi = params['degrees'] * torch.tensor(phis, dtype=torch.float32)
  phi = phi[:, None, None, None, None, None]
  phi = torch.tile(phi, (1, pixelsX, pixelsY, 1, 1, 1))
  params['phi'] = phi

  pte = torch.tensor(pte, dtype=torch.complex64)
  pte = pte[:, None, None, None]
  pte = torch.tile(pte, (1, pixelsX, pixelsY, 1))
  params['pte'] = pte

  ptm = torch.tensor(ptm, dtype=torch.complex64)
  ptm = ptm[:, None, None, None]
  ptm = torch.tile(ptm, (1, pixelsX, pixelsY, 1))
  params['ptm'] = ptm

  # Device parameters.
  params['ur1'] = 1.0 # permeability in reflection region
  params['er1'] = 1.0 # permittivity in reflection region
  params['ur2'] = 1.0 # permeability in transmission region
  params['er2'] = 1.0 # permittivity in transmission region
  params['urd'] = 1.0 # permeability of device
  params['erd'] = erd # permittivity of device
  params['urs'] = 1.0 # permeability of substrate
  params['ers'] = ers # permittivity of substrate
  params['Lx'] = Lx * params['micrometers'] # period along x
  params['Ly'] = Ly * params['micrometers'] # period along y
  L = torch.tensor(L, dtype=torch.complex64)
  L = L[None, None, None, :, None, None]
  params['L'] = L * params['micrometers']
  params['length_min'] = 0.1
  params['length_max'] = 2.0

  # RCWA parameters.
  params['PQ'] = PQ # number of spatial harmonics along x and y
  params['Nx'] = Nx # number of point along x in real-space grid
  if params['PQ'][1] == 1:
    params['Ny'] = 1
  else:
    params['Ny'] = int(np.round(params['Nx'] * params['Ly'] / params['Lx'])) # number of point along y in real-space grid

  # Coefficient for the argument of torch.sigmoid() when generating
  # permittivity distributions with geometric parameters.
  params['sigmoid_coeff'] = 1000.0

  # Polynomial order for rectangular resonators definition.
  params['rectangle_power'] = 200

  # Allowed permittivity range.
  params['eps_min'] = eps_min
  params['eps_max'] = eps_max

  # Upsampling for Fourier optics propagation.
  params['upsample'] = 1

  # Duty Cycle limits for gratings.
  params['duty_min'] = 0.1
  params['duty_max'] = 0.9

  # Permittivity density blur radius.
  params['blur_radius'] = blur_radius * params['micrometers']

  return params


def make_propagator(params, f):
  '''
    Pre-computes the band-limited angular spectrum propagator for modelling
    free-space propagation for the distance and sampling as specified in `params`.

    Args:
        params: A `dict` containing simulation and optimization settings.

        f: A `float` specifying the focal length, or distance to propagate, in
        meters.
    Returns:
        propagator: a `torch.Tensor` of shape `(batchSize, params['upsample'] * pixelsX,
        params['upsample'] * pixelsY)` and dtype `torch.complex64` defining the 
        reciprocal space, band-limited angular spectrum propagator.
  '''

  # Simulation tensor shape.
  batchSize = params['batchSize']
  pixelsX = params['pixelsX']
  pixelsY = params['pixelsY']
  upsample = params['upsample']

  # Propagator definition.
  k = 2 * np.pi / params['lam0'][:, 0, 0, 0, 0, 0]
  k = k[:, None, None]
  samp = params['upsample'] * pixelsX
  k = torch.tile(k, (1, 2 * samp - 1, 2 * samp - 1))
  k = k.type(torch.complex64)  
  k_xlist_pos = 2 * np.pi * np.linspace(0, 1 / (2 *  params['Lx'] / params['upsample']), samp)  
  front = k_xlist_pos[-(samp - 1):]
  front = -front[::-1]
  k_xlist = torch.tensor(np.hstack((front, k_xlist_pos)), dtype = torch.float32)
  k_x = torch.kron(k_xlist, torch.ones((2 * samp - 1, 1)))
  k_x = k_x[None, :, :]
  k_y = torch.permute(k_x, (0, 2, 1))
  k_x = k_x.type(torch.complex64)
  k_x = torch.tile(k_x, (batchSize, 1, 1))
  k_y = k_y.type(torch.complex64)
  k_y = torch.tile(k_y, (batchSize, 1, 1))
  k_z_arg = torch.square(k) - (torch.square(k_x) + torch.square(k_y))
  k_z = torch.sqrt(k_z_arg)
  
  # Cast to double precision to accommodate long focal lengths.
  propagator_arg = 1j * k_z * f
  propagator = torch.exp(propagator_arg)

  # Limit transfer function bandwidth to prevent aliasing.
  kx_limit = 2 * np.pi * (((1 / (pixelsX * params['Lx'])) * f) ** 2 + 1) ** (-0.5) / params['lam0'][:, 0, 0, 0, 0, 0]
  kx_limit = kx_limit.type(torch.complex64)
  ky_limit = kx_limit
  kx_limit = kx_limit[:, None, None]
  ky_limit = ky_limit[:, None, None]

  # Apply the antialiasing filter.
  ellipse_kx = torch.real(torch.square(k_x / kx_limit) + torch.square(k_y / k)) <= 1
  ellipse_ky = torch.real(torch.square(k_x / k) + torch.square(k_y / ky_limit)) <= 1   
  propagator = propagator * ellipse_kx * ellipse_ky
  
  return propagator


def propagate(field, propagator, upsample):
  '''
    Propagates a batch of input fields to a parallel output plane using the 
    band-limited angular spectrum method.

    Args:
        field: A `torch.Tensor` of shape `(batchSize, params['upsample'] * pixelsX,
        params['upsample'] * pixelsY)` and dtype `torch.complex64` specifying the 
        input electric fields to be diffracted to the output plane.

        propagator: a `torch.Tensor` of shape `(batchSize, params['upsample'] * pixelsX,
        params['upsample'] * pixelsY)` and dtype `torch.complex64` defining the 
        reciprocal space, band-limited angular spectrum propagator.

        upsample: An odd-valued `int` specifying the factor by which the
        transverse field data stored in `field` should be upsampled.
        
    Returns:
        out: A `torch.Tensor` of shape `(batchSize, params['upsample'] * pixelsX,
        params['upsample'] * pixelsY)` and dtype `torch.complex64` specifying the 
        the electric fields at the output plane.
  '''

  batchSize, _, m = field.shape
  n = upsample * m

  field_real = torch.real(field)
  field_imag = torch.imag(field)

  # Add an extra channels dimension for torch.nn.interpolate, and remove afterwards.
  field_real = field_real[None, :, :, :]
  field_imag = field_imag[None, :, :, :]
  field_real = torch.nn.functional.interpolate(field_real, size=[n,n], mode='nearest')
  field_imag = torch.nn.functional.interpolate(field_imag, size=[n,n], mode='nearest')
  field_real = field_real[0, :, :, :]
  field_imag = field_imag[0, :, :, :]
  field = field_real.type(torch.complex64) + 1j * field_imag.type(torch.complex64)

  # To pad total image to have dimension 2n - 1, have to pad with (n-1)/2 on each side.
  field = torch.nn.functional.pad(field, [(n-1) // 2, -((n-1) // -2), (n-1) // 2, -((n-1) // -2)])

  # Apply the propagator in Fourier space.
  field_freq = torch.fft.fftshift(torch.fft.fft2(field), dim = (1,2))
  field_filtered = torch.fft.ifftshift(field_freq * propagator, dim = (1,2))
  out = torch.fft.ifft2(field_filtered)
    
  # Crop back down to n x n matrices.
  out = out[:, (n-1) // 2 : n-1-((n-1) // -2), (n-1) // 2 : n-1-((n-1) // -2)]

  return out


def define_input_fields(params):
  '''
    Given the batch of input conditions with different wavelengths and incidence
    angles, this gives the input source fields incident on the metasurface.

    Args:
        params: A `dict` containing simulation and optimization settings.
    Returns:
        A `torch.Tensor` of shape `(batchSize, pixelsX, pixelsY)` and dtype 
        `torch.complex64` specifying the source fields injected onto a metasurface
        at the input.
  '''

  # Define the cartesian cross section.
  pixelsX = params['pixelsX']
  pixelsY = params['pixelsY']
  dx = params['Lx'] # grid resolution along x
  dy = params['Ly'] # grid resolution along y
  xa = torch.linspace(0, pixelsX - 1, pixelsX) * dx # x axis array
  xa = xa - torch.mean(xa) # center x axis at zero
  ya = torch.linspace(0, pixelsY - 1, pixelsY) * dy # y axis vector
  ya = ya - torch.mean(ya) # center y axis at zero
  [y_mesh, x_mesh] = torch.meshgrid(ya, xa, indexing='ij')
  x_mesh = x_mesh[None, :, :]
  y_mesh = y_mesh[None, :, :]

  # Extract the batch of wavelengths and input thetas.
  lam_phase_test = params['lam0'][:, 0, 0, 0, 0, 0]
  lam_phase_test = lam_phase_test[:, None, None]
  theta_phase_test = params['theta'][:, 0, 0, 0, 0, 0]
  theta_phase_test = theta_phase_test[:, None, None]

  # Apply a linear phase ramp based on the wavelength and thetas.
  phase_def = 2 * np.pi * torch.sin(theta_phase_test) * x_mesh / lam_phase_test
  phase_def = phase_def.type(torch.complex64)

  return torch.exp(1j * phase_def)


def solver_step1(ER_t, UR_t, PQ):

  ERC = rcwa_utils_pt.convmat(ER_t, PQ[0], PQ[1])
  URC = rcwa_utils_pt.convmat(UR_t, PQ[0], PQ[1])
  
  return ERC, URC


def solver_step2(batchSize, pixelsX, pixelsY, Nlay, Lx, Ly,
                        theta, phi, lam0, er1, er2, ur1, ur2, PQ):  
  
  I = torch.eye(np.prod(PQ), dtype = torch.complex64)
  I = I[None, None, None, None, :, :]
  I = torch.tile(I, (batchSize, pixelsX, pixelsY, Nlay, 1, 1))
  
  Z = torch.zeros((np.prod(PQ), np.prod(PQ)), dtype = torch.complex64)
  Z = Z[None, None, None, None, :, :]
  Z = torch.tile(Z, (batchSize, pixelsX, pixelsY, Nlay, 1, 1))
    
  n1 = np.sqrt(er1)
  n2 = np.sqrt(er2)

  k0 = 2 * np.pi / lam0
  k0 = k0.type(torch.complex64)
  
  kinc_x0 = n1 * torch.sin(theta) * torch.cos(phi)
  kinc_x0 = kinc_x0.type(torch.complex64)
  
  kinc_y0 = n1 * torch.sin(theta) * torch.sin(phi)
  kinc_y0 = kinc_y0.type(torch.complex64)
  
  kinc_z0 = n1 * torch.cos(theta)
  kinc_z0 = kinc_z0.type(torch.complex64)
  kinc_z0 = kinc_z0[:, :, :, 0, :, :]

  # Unit vectors
  T1 = np.transpose([2 * np.pi / Lx, 0])
  T2 = np.transpose([0, 2 * np.pi / Ly])
  p_max = np.floor(PQ[0] / 2.0)
  q_max = np.floor(PQ[1] / 2.0)
  p = torch.linspace(-p_max, p_max, PQ[0], dtype = torch.complex64) # indices along T1
  p = p[None, None, None, None, :, None]
  p = torch.tile(p, (1, pixelsX, pixelsY, Nlay, 1, 1))
  q = torch.linspace(-q_max, q_max, PQ[1], dtype = torch.complex64) # indices along T2
  q = q[None, None, None, None, None, :]
  q = torch.tile(q, (1, pixelsX, pixelsY, Nlay, 1, 1))

  # Build Kx and Ky matrices
  kx_zeros = torch.zeros(PQ[1], dtype = torch.complex64)
  kx_zeros = kx_zeros[None, None, None, None, None, :]
  ky_zeros = torch.zeros(PQ[0], dtype = torch.complex64)
  ky_zeros = ky_zeros[None, None, None, None, :, None]
  kx = kinc_x0 - 2 * np.pi * p / (k0 * Lx) - kx_zeros
  ky = kinc_y0 - 2 * np.pi * q / (k0 * Ly) - ky_zeros

  kx_T = torch.transpose(kx, dim0 = 4, dim1 = 5)
  KX = torch.reshape(kx_T, (batchSize, pixelsX, pixelsY, Nlay, np.prod(PQ)))
  KX = torch.diag_embed(KX, dim1 = -2, dim2 = -1)

  ky_T = torch.transpose(ky, dim0 = 4, dim1 = 5)
  KY = torch.reshape(ky_T, (batchSize, pixelsX, pixelsY, Nlay, np.prod(PQ)))
  KY = torch.diag_embed(KY, dim1 = -2, dim2 = -1)

  KZref = torch.matmul(torch.conj(ur1 * I), torch.conj(er1 * I))
  KZref = KZref - torch.matmul(KX, KX) - torch.matmul(KY, KY)
  KZref = torch.sqrt(KZref)
  KZref = -torch.conj(KZref)

  KZtrn = torch.matmul(torch.conj(ur2 * I), torch.conj(er2 * I))
  KZtrn = KZtrn - torch.matmul(KX, KX) - torch.matmul(KY, KY)
  KZtrn = torch.sqrt(KZtrn)
  KZtrn = torch.conj(KZtrn)

  return I, Z, k0, kinc_x0, kinc_y0, kinc_z0, KX, KY, KZref, KZtrn


def solver_step3(I, Z, KX, KY):

  KZ = I - torch.matmul(KX, KX) - torch.matmul(KY, KY)
  KZ = torch.sqrt(KZ)
  KZ = torch.conj(KZ)

  Q_free_00 = torch.matmul(KX, KY)
  Q_free_01 = I - torch.matmul(KX, KX)
  Q_free_10 = torch.matmul(KY, KY) - I
  Q_free_11 = -torch.matmul(KY, KX)
  Q_free_row0 = torch.cat([Q_free_00, Q_free_01], dim = 5)
  Q_free_row1 = torch.cat([Q_free_10, Q_free_11], dim = 5)
  Q_free = torch.cat([Q_free_row0, Q_free_row1], dim = 4)

  W0_row0 = torch.cat([I, Z], dim = 5)
  W0_row1 = torch.cat([Z, I], dim = 5)
  W0 = torch.cat([W0_row0, W0_row1], dim = 4)

  LAM_free_row0 = torch.cat([1j * KZ, Z], dim = 5)
  LAM_free_row1 = torch.cat([Z, 1j * KZ], dim = 5)
  LAM_free = torch.cat([LAM_free_row0, LAM_free_row1], dim = 4)

  V0 = torch.matmul(Q_free, torch.linalg.inv(LAM_free))

  return V0, W0


def solver_step4(batchSize, pixelsX, pixelsY, PQ):
  
  SG = dict({})
  SG_S11 = torch.zeros((2 * np.prod(PQ), 2 * np.prod(PQ)), dtype = torch.complex64)
  SG['S11'] = rcwa_utils_pt.expand_and_tile_pt(SG_S11, batchSize, pixelsX, pixelsY)

  SG_S12 = torch.eye(2 * np.prod(PQ), dtype = torch.complex64)
  SG['S12'] = rcwa_utils_pt.expand_and_tile_pt(SG_S12, batchSize, pixelsX, pixelsY)

  SG_S21 = torch.eye(2 * np.prod(PQ), dtype = torch.complex64)
  SG['S21'] = rcwa_utils_pt.expand_and_tile_pt(SG_S21, batchSize, pixelsX, pixelsY)

  SG_S22 = torch.zeros((2 * np.prod(PQ), 2 * np.prod(PQ)), dtype = torch.complex64)
  SG['S22'] = rcwa_utils_pt.expand_and_tile_pt(SG_S22, batchSize, pixelsX, pixelsY)

  return SG


def solver_step5(L, Nlay, ERC, URC, k0, KX, KY, V0, W0, SG):
    
  # Build the eigenvalue problem.
  P_00 = torch.matmul(KX, torch.linalg.inv(ERC))
  P_00 = torch.matmul(P_00, KY)

  P_01 = torch.matmul(KX, torch.linalg.inv(ERC))
  P_01 = torch.matmul(P_01, KX)
  P_01 = URC - P_01

  P_10 = torch.matmul(KY, torch.linalg.inv(ERC))
  P_10 = torch.matmul(P_10, KY) - URC

  P_11 = torch.matmul(-KY, torch.linalg.inv(ERC))
  P_11 = torch.matmul(P_11, KX)

  P_row0 = torch.cat([P_00, P_01], dim = 5)
  P_row1 = torch.cat([P_10, P_11], dim = 5)
  P = torch.cat([P_row0, P_row1], dim = 4)

  Q_00 = torch.matmul(KX, torch.linalg.inv(URC))
  Q_00 = torch.matmul(Q_00, KY)

  Q_01 = torch.matmul(KX, torch.linalg.inv(URC))
  Q_01 = torch.matmul(Q_01, KX)
  Q_01 = ERC - Q_01

  Q_10 = torch.matmul(KY, torch.linalg.inv(URC))
  Q_10 = torch.matmul(Q_10, KY) - ERC

  Q_11 = torch.matmul(-KY, torch.linalg.inv(URC))
  Q_11 = torch.matmul(Q_11, KX)

  Q_row0 = torch.cat([Q_00, Q_01], dim = 5)
  Q_row1 = torch.cat([Q_10, Q_11], dim = 5)
  Q = torch.cat([Q_row0, Q_row1], dim = 4)

  # Compute eignmodes for the layers in each pixel for the whole batch.
  OMEGA_SQ = torch.matmul(P, Q)
  #LAM, W = rcwa_utils_pt.eig_general(OMEGA_SQ)
  LAM, W = checkpoint(rcwa_utils_pt.eig_general, OMEGA_SQ)
  LAM = torch.sqrt(LAM)
  LAM = torch.diag_embed(LAM, dim1 = -2, dim2 = -1)

  V = torch.matmul(Q, W)
  V = torch.matmul(V, torch.linalg.inv(LAM))

  # Scattering matrices for the layers in each pixel for the whole batch.
  W_inv = torch.linalg.inv(W)
  V_inv = torch.linalg.inv(V)
  A = torch.matmul(W_inv, W0) + torch.matmul(V_inv, V0)
  B = torch.matmul(W_inv, W0) - torch.matmul(V_inv, V0)
  X = torch.matrix_exp(-LAM * k0 * L)
  #X = checkpoint(torch.matrix_exp, -LAM * k0 * L)

  S = dict({})
  A_inv = torch.linalg.inv(A)
  S11_left = torch.matmul(X, B)
  S11_left = torch.matmul(S11_left, A_inv)
  S11_left = torch.matmul(S11_left, X)
  S11_left = torch.matmul(S11_left, B)
  S11_left = A - S11_left
  S11_left = torch.linalg.inv(S11_left)

  S11_right = torch.matmul(X, B)
  S11_right = torch.matmul(S11_right, A_inv)
  S11_right = torch.matmul(S11_right, X)
  S11_right = torch.matmul(S11_right, A)
  S11_right = S11_right - B
  S['S11'] = torch.matmul(S11_left, S11_right)

  S12_right = torch.matmul(B, A_inv)
  S12_right = torch.matmul(S12_right, B)
  S12_right = A - S12_right
  S12_left = torch.matmul(S11_left, X)
  S['S12'] = torch.matmul(S12_left, S12_right)

  S['S21'] = S['S12']
  S['S22'] = S['S11']

  # Update the global scattering matrices.
  for l in range(Nlay):
    S_layer = dict({})
    S_layer['S11'] = S['S11'][:, :, :, l, :, :]
    S_layer['S12'] = S['S12'][:, :, :, l, :, :]
    S_layer['S21'] = S['S21'][:, :, :, l, :, :]
    S_layer['S22'] = S['S22'][:, :, :, l, :, :]
    SG = rcwa_utils_pt.redheffer_star_product(SG, S_layer)

  return S, SG


def solver_step6(I, Z, KX, KY, KZref, KZtrn, W0, V0, er1, ur1):

  # Eliminate layer dimension for tensors as they are unchanging on this dimension.
  KX = KX[:, :, :, 0, :, :]
  KY = KY[:, :, :, 0, :, :]
  KZref = KZref[:, :, :, 0, :, :]
  KZtrn = KZtrn[:, :, :, 0, :, :]
  Z = Z[:, :, :, 0, :, :]
  I = I[:, :, :, 0, :, :]
  W0 = W0[:, :, :, 0, :, :]
  V0 = V0[:, :, :, 0, :, :]

  Q_ref_00 = torch.matmul(KX, KY)
  Q_ref_01 = ur1 * er1 * I - torch.matmul(KX, KX)
  Q_ref_10 = torch.matmul(KY, KY) - ur1 * er1 * I
  Q_ref_11 = -torch.matmul(KY, KX)
  Q_ref_row0 = torch.cat([Q_ref_00, Q_ref_01], dim = 4)
  Q_ref_row1 = torch.cat([Q_ref_10, Q_ref_11], dim = 4)
  Q_ref = torch.cat([Q_ref_row0, Q_ref_row1], dim = 3)

  W_ref_row0 = torch.cat([I, Z], dim = 4)
  W_ref_row1 = torch.cat([Z, I], dim = 4)
  W_ref = torch.cat([W_ref_row0, W_ref_row1], dim = 3)

  LAM_ref_row0 = torch.cat([-1j * KZref, Z], dim = 4)
  LAM_ref_row1 = torch.cat([Z, -1j * KZref], dim = 4)
  LAM_ref = torch.cat([LAM_ref_row0, LAM_ref_row1], dim = 3)

  V_ref = torch.matmul(Q_ref, torch.linalg.inv(LAM_ref))

  W0_inv = torch.linalg.inv(W0)
  V0_inv = torch.linalg.inv(V0)
  A_ref = torch.matmul(W0_inv, W_ref) + torch.matmul(V0_inv, V_ref)
  A_ref_inv = torch.linalg.inv(A_ref)
  B_ref = torch.matmul(W0_inv, W_ref) - torch.matmul(V0_inv, V_ref)

  SR = dict({})
  SR['S11'] = torch.matmul(-A_ref_inv, B_ref)
  SR['S12'] = 2 * A_ref_inv
  SR_S21 = torch.matmul(B_ref, A_ref_inv)
  SR_S21 = torch.matmul(SR_S21, B_ref)
  SR['S21'] = 0.5 * (A_ref - SR_S21)
  SR['S22'] = torch.matmul(B_ref, A_ref_inv)
    
  return I, Z, KX, KY, KZref, KZtrn, W0, V0, W_ref, SR


def solver_step7(I, Z, KX, KY, KZref, KZtrn, W0, V0, er2, ur2):
  
  Q_trn_00 = torch.matmul(KX, KY)
  Q_trn_01 = ur2 * er2 * I - torch.matmul(KX, KX)
  Q_trn_10 = torch.matmul(KY, KY) - ur2 * er2 * I
  Q_trn_11 = -torch.matmul(KY, KX)
  Q_trn_row0 = torch.cat([Q_trn_00, Q_trn_01], dim = 4)
  Q_trn_row1 = torch.cat([Q_trn_10, Q_trn_11], dim = 4)
  Q_trn = torch.cat([Q_trn_row0, Q_trn_row1], dim = 3)

  W_trn_row0 = torch.cat([I, Z], dim = 4)
  W_trn_row1 = torch.cat([Z, I], dim = 4)
  W_trn = torch.cat([W_trn_row0, W_trn_row1], dim = 3)

  LAM_trn_row0 = torch.cat([1j * KZtrn, Z], dim = 4)
  LAM_trn_row1 = torch.cat([Z, 1j * KZtrn], dim = 4)
  LAM_trn = torch.cat([LAM_trn_row0, LAM_trn_row1], dim = 3)

  V_trn = torch.matmul(Q_trn, torch.linalg.inv(LAM_trn))

  W0_inv = torch.linalg.inv(W0)
  V0_inv = torch.linalg.inv(V0)
  A_trn = torch.matmul(W0_inv, W_trn) + torch.matmul(V0_inv, V_trn)
  A_trn_inv = torch.linalg.inv(A_trn)
  B_trn = torch.matmul(W0_inv, W_trn) - torch.matmul(V0_inv, V_trn)

  ST = dict({})
  ST['S11'] = torch.matmul(B_trn, A_trn_inv)
  ST_S12 = torch.matmul(B_trn, A_trn_inv)
  ST_S12 = torch.matmul(ST_S12, B_trn)
  ST['S12'] = 0.5 * (A_trn - ST_S12)
  ST['S21'] = 2 * A_trn_inv
  ST['S22'] = torch.matmul(-A_trn_inv, B_trn)
    
  return W_trn, ST


def solver_step8(SG, SR, ST):
  
  SG = rcwa_utils_pt.redheffer_star_product(SR, SG)
  SG = rcwa_utils_pt.redheffer_star_product(SG, ST)
  
  return SG


def solver_step9(batchSize, pixelsX, pixelsY, PQ, pte, ptm, kinc_x0, kinc_y0, kinc_z0, W_ref):
    
  # Compute mode coefficients of the source.
  delta = torch.zeros((batchSize, pixelsX, pixelsY, np.prod(PQ)), dtype = torch.float32)
  delta[:, :, :, int(np.prod(PQ) / 2.0)] = 1

  # Incident wavevector.
  kinc_x0_pol = torch.real(kinc_x0[:, :, :, 0, 0])
  kinc_y0_pol = torch.real(kinc_y0[:, :, :, 0, 0])
  kinc_z0_pol = torch.real(kinc_z0[:, :, :, 0])
  kinc_pol = torch.cat([kinc_x0_pol, kinc_y0_pol, kinc_z0_pol], dim = 3)

  # Calculate TE and TM polarization unit vectors.
  firstPol = True
  for pol in range(batchSize):
    if (kinc_pol[pol, 0, 0, 0] == 0.0 and kinc_pol[pol, 0, 0, 1] == 0.0):
      ate_pol = np.zeros((1, pixelsX, pixelsY, 3))
      ate_pol[:, :, :, 1] = 1
      ate_pol = torch.tensor(ate_pol, dtype = torch.float32)
    else:
      # Calculation of `ate` for oblique incidence.
      n_hat = np.zeros((1, pixelsX, pixelsY, 3))
      n_hat[:, :, :, 0] = 1
      n_hat = torch.tensor(n_hat, dtype = torch.float32)
      kinc_pol_iter = kinc_pol[pol, :, :, :]
      kinc_pol_iter = kinc_pol_iter[None, :, :, :]
      ate_cross = torch.cross(n_hat, kinc_pol_iter)
      ate_pol =  ate_cross / torch.linalg.norm(ate_cross, dim = 3, keepdim = True)

    if firstPol:
      ate = ate_pol
      firstPol = False
    else:
      ate = torch.cat([ate, ate_pol], dim = 0)

  atm_cross = torch.cross(kinc_pol, ate)
  atm = atm_cross / torch.linalg.norm(atm_cross, dim = 3, keepdim = True)
  ate = ate.type(torch.complex64)
  atm = atm.type(torch.complex64)

  # Decompose the TE and TM polarization into x and y components.
  EP = pte * ate + ptm * atm
  EP_x = EP[:, :, :, 0]
  EP_x = EP_x[:, :, :, None]
  EP_y = EP[:, :, :, 1]
  EP_y = EP_y[:, :, :, None]

  esrc_x = EP_x * delta
  esrc_y = EP_y * delta

  esrc = torch.cat([esrc_x, esrc_y], dim = 3)
  esrc = esrc[:, :, :, :, None]

  W_ref_inv = torch.linalg.inv(W_ref)
 
  return W_ref_inv, esrc


def solver_step10(PQ, KX, KY, KZref, KZtrn, SG, W_ref, W_trn, W_ref_inv, esrc):
    
  csrc = torch.matmul(W_ref_inv, esrc)

  # Compute tranmission and reflection mode coefficients.
  cref = torch.matmul(SG['S11'], csrc)
  ctrn = torch.matmul(SG['S21'], csrc)
  eref = torch.matmul(W_ref, cref)
  etrn = torch.matmul(W_trn, ctrn)

  rx = eref[:, :, :, 0 : np.prod(PQ), :]
  ry = eref[:, :, :, np.prod(PQ) : 2 * np.prod(PQ), :]
  tx = etrn[:, :, :, 0 : np.prod(PQ), :]
  ty = etrn[:, :, :, np.prod(PQ) : 2 * np.prod(PQ), :]

  # Compute longitudinal components.
  KZref_inv = torch.linalg.inv(KZref)
  KZtrn_inv = torch.linalg.inv(KZtrn)
  rz = torch.matmul(KX, rx) + torch.matmul(KY, ry)
  rz = torch.matmul(-KZref_inv, rz)
  tz = torch.matmul(KX, tx) + torch.matmul(KY, ty)
  tz = torch.matmul(-KZtrn_inv, tz)
 
  return rx, ry, rz, tx, ty, tz


def solver_step11(batchSize, pixelsX, pixelsY, PQ, kinc_z0, KZref, KZtrn, ur1, ur2, rx, ry, rz, tx, ty, tz):

  rx2 = torch.real(rx) ** 2 + torch.imag(rx) ** 2
  ry2 = torch.real(ry) ** 2 + torch.imag(ry) ** 2
  rz2 = torch.real(rz) ** 2 + torch.imag(rz) ** 2
  R2 = rx2 + ry2 + rz2
  R = torch.real(-KZref / ur1) / torch.real(kinc_z0 / ur1)
  R = torch.matmul(R, R2)
  R = torch.reshape(R, (batchSize, pixelsX, pixelsY, PQ[0], PQ[1]))
  REF = torch.sum(R, dim = [3, 4])

  tx2 = torch.real(tx) ** 2 + torch.imag(tx) ** 2
  ty2 = torch.real(ty) ** 2 + torch.imag(ty) ** 2
  tz2 = torch.real(tz) ** 2 + torch.imag(tz) ** 2
  T2 = tx2 + ty2 + tz2
  T = torch.real(KZtrn / ur2) / torch.real(kinc_z0 / ur2)
  T = torch.matmul(T, T2)
  T = torch.reshape(T, (batchSize, pixelsX, pixelsY, PQ[0], PQ[1]))
  TRN = torch.sum(T, dim = [3, 4])
  
  return R, REF, T, TRN


def simulate(ER_t, UR_t, params):
  '''
    Calculates the transmission/reflection coefficients for a unit cell with a
    given permittivity/permeability distribution and the batch of input conditions 
    (e.g., wavelengths, wavevectors, polarizations) for a fixed real space grid
    and number of Fourier harmonics.

    Args:
        ER_t: A `torch.Tensor` of shape `(batchSize, pixelsX, pixelsY, Nlayer, Nx, Ny)`
        and dtype `torch.complex64` specifying the relative permittivity distribution
        of the unit cell.

        UR_t: A `torch.Tensor` of shape `(batchSize, pixelsX, pixelsY, Nlayer, Nx, Ny)`
        and dtype `torch.complex64` specifying the relative permeability distribution
        of the unit cell.

        params: A `dict` containing simulation and optimization settings.
    Returns:
        outputs: A `dict` containing the keys {'rx', 'ry', 'rz', 'R', 'ref', 
        'tx', 'ty', 'tz', 'T', 'TRN'} corresponding to the computed reflection/tranmission
        coefficients and powers.
  '''

  # Extract parameters from the `params` dictionary.
  batchSize = params['batchSize']
  pixelsX = params['pixelsX']
  pixelsY = params['pixelsY']
  L = params['L']
  Nlay = params['Nlay']
  Lx = params['Lx']
  Ly = params['Ly']
  theta = params['theta']
  phi = params['phi']
  pte = params['pte']
  ptm = params['ptm']
  lam0 = params['lam0']
  er1 = params['er1']
  er2 = params['er2']
  ur1 = params['ur1']
  ur2 = params['ur2']
  PQ = params['PQ']

  ### Step 1: Build convolution matrices for the permittivity and permeability ###
  ERC, URC = solver_step1( ER_t, UR_t, PQ)
  
  ### Step 2: Wave vector expansion ###
  I, Z, k0, kinc_x0, kinc_y0, kinc_z0, KX, KY, KZref, KZtrn = solver_step2(batchSize, pixelsX, pixelsY, Nlay, Lx, Ly, theta, phi, lam0, er1, er2, ur1, ur2, PQ)

  ### Step 3: Free Space ###
  V0, W0 = solver_step3(I, Z, KX, KY)

  ### Step 4: Initialize Global Scattering Matrix ###
  SG = solver_step4(batchSize, pixelsX, pixelsY, PQ)
  
  ### Step 5: Calculate eigenmodes ###
  S, SG = solver_step5(L, Nlay, ERC, URC, k0, KX, KY, V0, W0, SG)
  
  ### Step 6: Reflection side ###
  I, Z, KX, KY, KZref, KZtrn, W0, V0, W_ref, SR = solver_step6(I, Z, KX, KY, KZref, KZtrn, W0, V0, er1, ur1)
  
  ### Step 7: Transmission side ###
  W_trn, ST = solver_step7(I, Z, KX, KY, KZref, KZtrn, W0, V0, er2, ur2)
  
  ### Step 8: Compute global scattering matrix ###
  SG = solver_step8(SG, SR, ST)
  
  ### Step 9: Compute source parameters ###
  W_ref_inv, esrc = solver_step9(batchSize, pixelsX, pixelsY, PQ, pte, ptm, kinc_x0, kinc_y0, kinc_z0, W_ref)
  
  ### Step 10: Compute reflected and transmitted fields ###
  rx, ry, rz, tx, ty, tz = solver_step10(PQ, KX, KY, KZref, KZtrn, SG, W_ref, W_trn, W_ref_inv, esrc)
  
  ### Step 11: Compute diffraction efficiences ###
  R, REF, T, TRN = solver_step11(batchSize, pixelsX, pixelsY, PQ, kinc_z0, KZref, KZtrn, ur1, ur2, rx, ry, rz, tx, ty, tz)

  # Store the transmission/reflection coefficients and powers in a dictionary.
  outputs = dict({})
  outputs['rx'] = rx
  outputs['ry'] = ry
  outputs['rz'] = rz
  outputs['R'] = R
  outputs['REF'] = REF
  outputs['tx'] = tx
  outputs['ty'] = ty
  outputs['tz'] = tz
  outputs['T'] = T
  outputs['TRN'] = TRN

  return outputs