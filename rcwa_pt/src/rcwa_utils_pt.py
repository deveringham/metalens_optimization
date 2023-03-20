'''
rcwa_utils_py.pt

Contains some mathematical functions used in the PyTorch implementation
of RCWA.

This file is a modified version of the original by
Copyright (c) 2020 Shane Colburn, University of Washington

Modifications made by Dylan Everingham, 2022
for performance optimization and conversion to use PyTorch.
'''

import torch
import numpy as np

    
def convmat(A, P, Q):
    '''
    This function computes a convolution matrix for a real space matrix `A` that
    represents either a relative permittivity or permeability distribution for a
    set of pixels, layers, and batch.
    Args:
        A: A `torch.Tensor` of dtype `complex` and shape `(batchSize, pixelsX, 
        pixelsY, Nlayers, Nx, Ny)` specifying real space values on a Cartesian
        grid.

        P: A positive and odd `int` specifying the number of spatial harmonics 
        along `T1`.

        Q: A positive and odd `int` specifying the number of spatial harmonics 
        along `T2`.
    Returns:
        A `torch.Tensor` of dtype `complex` and shape `(batchSize, pixelsX, 
        pixelsY, Nlayers, P * Q, P * Q)` representing a stack of convolution 
        matrices based on `A`.   
    '''
    
    # Determine the shape of A.
    batchSize, pixelsX, pixelsY, Nlayers, Nx, Ny = A.shape

    # Compute indices of spatial harmonics.
    NH = P * Q # total number of harmonics.
    p_max = int(np.floor(P / 2))
    q_max = int(np.floor(P / 2))

    # Indices along T1 and T2.
    p = np.linspace(-p_max, p_max, P)
    q = np.linspace(-q_max, q_max, Q)
    
    # Compute array indices of the center harmonic.
    p0 = int(np.floor(Nx / 2))
    q0 = int(np.floor(Ny / 2))
    
    # Fourier transform the real space distributions.
    A = torch.fft.fftshift(torch.fft.fft2(A), dim = (4,5)) / (Nx * Ny)
    
    # Build the matrix.
    firstCoeff = True
    for qrow in range(Q):
        for prow in range(P):
            for qcol in range(Q):
                for pcol in range(P):
                    pfft = int(p[prow] - p[pcol])
                    qfft = int(q[qrow] - q[qcol])

                    # Sequentially concatenate Fourier coefficients.
                    value = A[:, :, :, :, p0 + pfft, q0 + qfft]
                    value = value[:, :, :, :, None, None]
                    if firstCoeff:
                        firstCoeff = False
                        C = value
                    else:
                        C = torch.cat([C, value], dim = 5)
    
    # Reshape the coefficients tensor into a stack of convolution matrices.
    convMatrixShape = (batchSize, pixelsX, pixelsY, Nlayers, P * Q, P * Q)      
    matrixStack = torch.reshape(C, convMatrixShape)
    
    return matrixStack


def redheffer_star_product(SA, SB):
    '''
    This function computes the redheffer star product of two block matrices, 
    which is the result of combining the S-parameter of two systems.
    Args:
        SA: A `dict` of `torch.Tensor` values specifying the block matrix 
        corresponding to the S-parameters of a system. `SA` needs to have the 
        keys ('S11', 'S12', 'S21', 'S22'), where each key maps to a `torch.Tensor`
        of shape `(batchSize, pixelsX, pixelsY, 2*NH, 2*NH)`, where NH is the 
        total number of spatial harmonics.

        SB: A `dict` of `torch.Tensor` values specifying the block matrix 
        corresponding to the S-parameters of a second system. `SB` needs to have
        the keys ('S11', 'S12', 'S21', 'S22'), where each key maps to a
        `torch.Tensor` of shape `(batchSize, pixelsX, pixelsY, 2*NH, 2*NH)`, where 
        NH is the total number of spatial harmonics.
    Returns:
        A `dict` of `torch.Tensor` values specifying the block matrix 
        corresponding to the S-parameters of the combined system. `SA` needs 
        to have the keys ('S11', 'S12', 'S21', 'S22'), where each key maps to
        a `torch.Tensor` of shape `(batchSize, pixelsX, pixelsY, 2*NH, 2*NH),
        where NH is the total number of spatial harmonics.  
    '''
    # Define the identity matrix.
    batchSize, pixelsX, pixelsY, dim, _ = SA['S11'].shape
    I = torch.eye(dim, dtype = torch.complex64)
    I = I[None, None, None, :, :]
    I = torch.tile(I, (batchSize, pixelsX, pixelsY, 1, 1))
    
    # Calculate S11.
    S11 = torch.linalg.inv(I - torch.matmul(SB['S11'], SA['S22']))
    S11 = torch.linalg.matmul(S11, SB['S11'])
    S11 = torch.matmul(SA['S12'], S11)
    S11 = SA['S11'] + torch.matmul(S11, SA['S21'])
    
    # Calculate S12.
    S12 = torch.linalg.inv(I - torch.matmul(SB['S11'], SA['S22']))
    S12 = torch.matmul(S12, SB['S12'])
    S12 = torch.matmul(SA['S12'], S12)
    
    # Calculate S21.
    S21 = torch.linalg.inv(I - torch.matmul(SA['S22'], SB['S11']))
    S21 = torch.matmul(S21, SA['S21'])
    S21 = torch.matmul(SB['S21'], S21)
    
    # Calculate S22.
    S22 = torch.linalg.inv(I - torch.matmul(SA['S22'], SB['S11']))
    S22 = torch.matmul(S22, SA['S22'])
    S22 = torch.matmul(SB['S21'], S22)
    S22 = SB['S22'] + torch.matmul(S22, SB['S12'])
    
    # Store S parameters in an output dictionary.
    S = dict({})
    S['S11'] = S11
    S['S12'] = S12
    S['S21'] = S21
    S['S22'] = S22
    
    return S

class EigCustomGrad(torch.autograd.Function):
    '''
    Computes the eigendecomposition of a batch of matrices, the same as 
    `torch.linalg.eig()` but assumes the input shape also has extra dimensions for pixels
    and layers. This function also provides the reverse mode gradient of the 
    eigendecomposition as derived in 10.1109/ICASSP.2017.7952140. This applies 
    for general, complex matrices that do not have to be self adjoint. This 
    result gives the exact reverse mode gradient for nondegenerate eigenvalue 
    problems. To extend to the case of degenerate eigenvalues common in RCWA, we
    approximate the gradient by a Lorentzian broadening technique that 
    introduces a small error but stabilizes the calculation. This is based on
    10.1103/PhysRevX.9.031041.
    Args:
        A: A `torch.Tensor` of shape `(batchSize, pixelsX, pixelsY, Nlayers, Nx, 
        Ny)` and dtype `torch.complex64` where the last two dimensions define 
        matrices for which we will calculate the eigendecomposition of their 
        reverse mode gradients.

        eps: A `float` defining a regularization parameter used in the 
        denominator of the Lorentzian broadening calculation to enable reverse
        mode gradients for degenerate eigenvalues.

    Returns:
        A `Tuple(List[torch.Tensor, torch.Tensor], torch.Tensor)`, where the `List`
        specifies the eigendecomposition as computed by `torch.linalg.eig()` and the 
        second element of the `Tuple` gives the reverse mode gradient of the
        eigendecompostion of the input argument `A`.
  '''

    @staticmethod
    def forward(ctx, A, eps = 1E-6):
        
        # Perform the eigendecomposition.
        eigenvalues, eigenvectors = torch.linalg.eig(A)
        
        # Sort the eigenvalues in non-descencing order, according to their real parts.
        indices = torch.real(eigenvalues).argsort()
        
        # Apply the ordering to the (complex) eigenvalues and their eigenvectors.
        eigenvalues = eigenvalues.gather(dim=-1, index=indices)
        indices = indices[:,:,:,:,None,:]
        indices = indices.repeat((1,1,1,1,indices.shape[-1],1))
        eigenvectors = eigenvectors.gather(dim=-1, index=indices)
        
        # Save the decomposition for the backwards pass.
        ctx.save_for_backward(A, eigenvalues, eigenvectors)
        ctx.eps = eps
        
        return eigenvalues, eigenvectors

    @staticmethod
    def backward(ctx, grad_D, grad_U):
        
        # Get the pre-computed eigendecomposition.
        A, D, U = ctx.saved_tensors
        
        # Referse mode gradient calculation.
        # Convert eigenvalues gradient to a diagonal matrix.
        grad_D = torch.diag_embed(grad_D, dim1 = -2, dim2 = -1)

        # Extract the tensor dimensions for later use.
        batchSize, pixelsX, pixelsY, Nlay, dim, _ = A.shape

        # Calculate intermediate matrices.
        I = torch.eye(dim, dtype = torch.complex64)
        D = torch.reshape(D, (batchSize, pixelsX, pixelsY, Nlay, dim, 1))
        shape_di = (batchSize, pixelsX, pixelsY, Nlay, dim, 1)
        shape_dj = (batchSize, pixelsX, pixelsY, Nlay, 1, dim)
        E = torch.ones(shape_di, dtype = torch.complex64) * torch.conj(torch.transpose(D, -2, -1))
        E = E - D * torch.ones(shape_dj, dtype = torch.complex64)
        E = torch.conj(torch.transpose(D, -2, -1)) - D

        # Lorentzian broadening.
        F = E / (E ** 2 + ctx.eps)
        F = F - I * F

        # Compute the reverse mode gradient of the eigendecomposition of A.
        grad_A = torch.conj(F) *  torch.matmul(torch.conj(torch.transpose(U, -2, -1)), grad_U) ### Error here
        grad_A = grad_D + grad_A
        grad_A = torch.matmul(grad_A, torch.conj(torch.transpose(U, -2, -1)))
        grad_A = torch.matmul(torch.linalg.inv(torch.conj(torch.transpose(U, -2, -1))), grad_A)
        
        return grad_A


# Produces a differentiable function usable by PyTorch.
eig_general = EigCustomGrad.apply


def expand_and_tile_np(array, batchSize, pixelsX, pixelsY):
  '''
    Expands and tiles a numpy array for a given batchSize and number of pixels.
    Args:
        array: A `np.ndarray` of shape `(Nx, Ny)`.
    Returns:
        A `np.ndarray` of shape `(batchSize, pixelsX, pixelsY, Nx, Ny)` with
        the values from `array` tiled over the new dimensions.
  '''
  array = array[np.newaxis, np.newaxis, np.newaxis, :, :]
  return np.tile(array, reps = (batchSize, pixelsX, pixelsY, 1, 1))


def expand_and_tile_pt(tensor, batchSize, pixelsX, pixelsY):
  '''
    Expands and tiles a `torch.Tensor` for a given batchSize and number of pixels.
    Args:
        tensor: A `torch.Tensor` of shape `(Nx, Ny)`.
    Returns:
        A `torch.Tensor` of shape `(batchSize, pixelsX, pixelsY, Nx, Ny)` with
        the values from `tensor` tiled over the new dimensions.
  '''
  tensor = tensor[None, None, None, :, :]
  return torch.tile(tensor, (batchSize, pixelsX, pixelsY, 1, 1))