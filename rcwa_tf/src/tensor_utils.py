# Copyright (c) 2020, Shane Colburn, University of Washington
# This file is part of rcwa_tf
# Written by Shane Colburn (Email: scolbur2@uw.edu)

import tensorflow as tf
import numpy as np
import json

def expand_and_tile_np(array, batchSize, pixelsX, pixelsY):
  '''
    Expands and tile a numpy array for a given batchSize and number of pixels.
    Args:
        array: A `np.ndarray` of shape `(Nx, Ny)`.
    Returns:
        A `np.ndarray` of shape `(batchSize, pixelsX, pixelsY, Nx, Ny)` with
        the values from `array` tiled over the new dimensions.
  '''
  array = array[np.newaxis, np.newaxis, np.newaxis, :, :]
  return np.tile(array, reps = (batchSize, pixelsX, pixelsY, 1, 1))

def expand_and_tile_tf(tensor, batchSize, pixelsX, pixelsY):
  '''
    Expands and tile a `tf.Tensor` for a given batchSize and number of pixels.
    Args:
        tensor: A `tf.Tensor` of shape `(Nx, Ny)`.
    Returns:
        A `tf.Tensor` of shape `(batchSize, pixelsX, pixelsY, Nx, Ny)` with
        the values from `tensor` tiled over the new dimensions.
  '''
  tensor = tensor[tf.newaxis, tf.newaxis, tf.newaxis, :, :]
  return tf.tile(tensor, multiples = (batchSize, pixelsX, pixelsY, 1, 1))


def check_eig_tf(A, e, v, eps=1e-4):
    l = tf.linalg.matvec(A,v)
    r = v*e
    mask = tf.abs(l-r) > eps
    return (tf.reduce_sum(tf.cast(mask, dtype=tf.uint8)) == 0).numpy()


@tf.custom_gradient
def eig_general(A, eps = 1E-6):
  '''
    Computes the eigendecomposition of a batch of matrices, the same as 
    `tf.eig()` but assumes the input shape also has extra dimensions for pixels
    and layers. This function also provides the reverse mode gradient of the 
    eigendecomposition as derived in 10.1109/ICASSP.2017.7952140. This applies 
    for general, complex matrices that do not have to be self adjoint. This 
    result gives the exact reverse mode gradient for nondegenerate eigenvalue 
    problems. To extend to the case of degenerate eigenvalues common in RCWA, we
    approximate the gradient by a Lorentzian broadening technique that 
    introduces a small error but stabilizes the calculation. This is based on
    10.1103/PhysRevX.9.031041.
    Args:
        A: A `tf.Tensor` of shape `(batchSize, pixelsX, pixelsY, Nlayers, Nx, 
        Ny)` and dtype `tf.complex64` where the last two dimensions define 
        matrices for which we will calculate the eigendecomposition of their 
        reverse mode gradients.

        eps: A `float` defining a regularization parameter used in the 
        denominator of the Lorentzian broadening calculation to enable reverse
        mode gradients for degenerate eigenvalues.

    Returns:
        A `Tuple(List[tf.Tensor, tf.Tensor], tf.Tensor)`, where the `List`
        specifies the eigendecomposition as computed by `tf.eig()` and the 
        second element of the `Tuple` gives the reverse mode gradient of the
        eigendecompostion of the input argument `A`.
  '''

  # Perform the eigendecomposition.
  eigenvalues, eigenvectors = tf.eig(A)

  #with open("tf.txt", 'w', encoding="utf-8") as f: 
  #  json.dump(tf.math.real(A).numpy().tolist(),f)

  '''
  batch0 = eigenvectors.shape[0]
  batch1 = eigenvectors.shape[1]
  batch2 = eigenvectors.shape[2]
  batch3 = eigenvectors.shape[3]
  rows = eigenvectors.shape[-2]
  cols = eigenvectors.shape[-1]
  
  tf_good = []
  for b0 in range(batch0):
    for b1 in range(batch1):
      for b2 in range(batch2):
        for b3 in range(batch3):
          for i in range(rows):
            tf_good.append(check_eig_tf(A[b0,b1,b2,b3,:,:], eigenvalues[b0,b1,b2,b3,i], eigenvectors[b0,b1,b2,b3,:,i]))

  print('Eig Check (True is pass):')
  print((np.sum(tf_good) - len(tf_good)) == 0)
  '''
    
  # Referse mode gradient calculation.
  def grad(grad_D, grad_U):

    # Use the pre-computed eigendecomposition.
    nonlocal eigenvalues, eigenvectors
    D = eigenvalues
    U = eigenvectors

    # Convert eigenvalues gradient to a diagonal matrix.
    grad_D = tf.linalg.diag(grad_D)

    # Extract the tensor dimensions for later use.
    batchSize, pixelsX, pixelsY, Nlay, dim, _ = A.shape

    # Calculate intermediate matrices.
    I = tf.eye(num_rows = dim, dtype = tf.complex64)
    D = tf.reshape(D, shape = (batchSize, pixelsX, pixelsY, Nlay, dim, 1))
    shape_di = (batchSize, pixelsX, pixelsY, Nlay, dim, 1)
    shape_dj = (batchSize, pixelsX, pixelsY, Nlay, 1, dim)
    E = tf.ones(shape = shape_di, dtype = tf.complex64) * tf.linalg.adjoint(D)
    E = E - D * tf.ones(shape = shape_dj, dtype = tf.complex64)
    E = tf.linalg.adjoint(D) - D

    # Lorentzian broadening.
    F = E / (E ** 2 + eps)
    F = F - I * F

    # Compute the reverse mode gradient of the eigendecomposition of A.
    grad_A = tf.math.conj(F) *  tf.linalg.matmul(tf.linalg.adjoint(U), grad_U)
    grad_A = grad_D + grad_A
    grad_A = tf.linalg.matmul(grad_A, tf.linalg.adjoint(U))
    grad_A = tf.linalg.matmul(tf.linalg.inv(tf.linalg.adjoint(U)), grad_A)
    return grad_A

  return [eigenvalues, eigenvectors], grad
