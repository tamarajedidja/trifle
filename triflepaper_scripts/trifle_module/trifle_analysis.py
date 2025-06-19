#!/usr/bin/env python
# coding: utf-8

# #### MODULE FOR TRIFLE ANALYSIS ---------------------------------------
# Version d.d. 09-01-2024 by TJ de Kloe

# In[ ]:
from numpy.linalg import det, cond, LinAlgError
import numpy as np

def matrix_properties(matrix, tolerance=1e-10):
    """
    Check if a matrix is full rank, invertible, and well-conditioned.
    
    Parameters:
    - matrix (np.ndarray): The input matrix to analyze.
    - tolerance (float): Threshold for determinant and rank-based checks.

    Returns:
    - dict: A dictionary with keys:
        - 'is_full_rank' (bool): Whether the matrix is full rank.
        - 'determinant' (float): Determinant of the matrix.
        - 'is_invertible' (bool): Whether the matrix is invertible.
        - 'condition_number' (float): Condition number of the matrix.
        - 'is_well_conditioned' (bool): Whether the matrix is well-conditioned.
    """
    properties = {}

    # Rank of the matrix
    rank = np.linalg.matrix_rank(matrix)
    properties['is_full_rank'] = rank == min(matrix.shape)
    properties['rank'] = rank

    # Determinant of the matrix
    try:
        determinant = det(matrix)
        properties['determinant'] = determinant
        properties['is_invertible'] = np.abs(determinant) > tolerance
    except LinAlgError:
        properties['determinant'] = None
        properties['is_invertible'] = False

    # Condition number of the matrix
    try:
        condition_number = cond(matrix)
        properties['condition_number'] = condition_number
        properties['is_well_conditioned'] = condition_number < 1e10
    except LinAlgError:
        properties['condition_number'] = None
        properties['is_well_conditioned'] = False

    # Print matrix properties as text
    print("\nMatrix Properties:")
    print(f"- Shape: {matrix.shape}")
    print(f"- Full Rank: {'Yes' if properties['is_full_rank'] else 'No'}")
    print(f"- Rank: {rank}")
    if properties['determinant'] is not None:
        print(f"- Determinant: {properties['determinant']:.4e}")
    else:
        print("- Determinant: Not computable")
    print(f"- Invertible: {'Yes' if properties['is_invertible'] else 'No'}")
    if properties['condition_number'] is not None:
        print(f"- Condition Number: {properties['condition_number']:.4e}")
        print(f"- Well-Conditioned: {'Yes' if properties['is_well_conditioned'] else 'No'}")
    else:
        print("- Condition Number: Not computable")

    return properties
#%%
def run_tICA(T, modelorder, seed):
    from sklearn.decomposition import FastICA
    import sys
    
    # THIS FUNCTION RUNS TEMPORAL ICA
    # Inputs are: 
    # - T: Spatial time series (DIM: time x parcels, t x parcels)
    if T.shape[1] > T.shape[0]:
        sys.exit(' Make sure that your input has dimensions time x parcels\nExiting.')
    
    # Variance normalise
    #T -= np.nanmean(T,axis=0)
    #T /= np.nanstd(T, axis=0)
    
    #Algorithm "parallel" is faster but converges less often
    tfm_ica = FastICA(n_components=modelorder, max_iter=100000, tol=1e-4, fun="logcosh", algorithm="deflation", random_state=seed, whiten=True) 
    #print("Whitened input data for tica")
    # 2) Fit temporal ICA
    tfms    = tfm_ica.fit_transform(T)
    M       = tfm_ica.mixing_
    B       = tfms.T
    return M, B, tfm_ica 

# In[1]:
    
def pseudoinverse(A):
    import numpy as np
    """
    Compute the Moore-Penrose pseudoinverse of a matrix.
    - Uses A^+ = (A^T A)^-1 A^T for matrices with full column rank.
    - Falls back to SVD-based computation otherwise.
    
    Parameters:
        A (numpy.ndarray): Input matrix.
        
    Returns:
        numpy.ndarray: The pseudoinverse of matrix A.
    """
    # Check if the matrix has full column rank
    rank = np.linalg.matrix_rank(A)
    num_columns = A.shape[1]
    
    if rank == num_columns:  # Full column rank
        print("Matrix has full column rank. Using (A^T A)^-1 A^T formula.")
        AtA = A.T @ A
        AtA_inv = np.linalg.inv(AtA)
        A_pinv = AtA_inv @ A.T
    else:
        print("Matrix does not have full column rank. Using SVD-based pseudoinverse.")
        # SVD-based pseudoinverse
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        S_inv = np.array([1/s if s > 1e-15 else 0 for s in S])  # Tolerance for singular values
        A_pinv = Vt.T @ np.diag(S_inv) @ U.T
    
    return A_pinv

#%%
# COMPUTE M_t/ f
def compute_tvM(S, M, B):
    # THIS FUNCTION COMPUTES THE M_t matrix 
    # Inputs are:
    # - S: Parcels (DIM: voxels x spatial components/parcels, i.e. v x d)
    # - M: TFM Mixing matrix (DIM: spatial components/parcels by temporal components, i.e. d x k)
    # - B: TFM time-series (DIM: temporal components by frames/time, i.e. k x t)
    
    ## CONCEPTUAL OVERVIEW OF THE DATA:
    # X_{v x t} = S_{v x d} * M_{d x k} * B_{k * t}
    
    ## LOAD PACKAGES
    import numpy as np
    import numpy.linalg as npl
    
    ## ALGEBRAIC COMPUTATION OF M_T
    # X_{v x t} = S_{v x d} * M_{d x k} * B_{k x t} 
    # pinv(S) = A
    # A * X * BT * (B*BT)^-1 = M_t
    # (B*BT)^-1 = Z
    # A * X * B^T * Z = M_t
    Xr    = np.dot(np.dot(S,M),B) # Compute noiseless reconstruction of the data
    A     = pseudoinverse(S)      # Compute A, the pseudo inverse of S
    BT    = np.transpose(B)       # Compute BT, the transpose of B
    BBT   = np.dot(B,BT)          # Compute BBT, B times its' transpose
    Z     = npl.inv(BBT)          # Compute Z, the inverse of BBT
    
    # Compute shapes:
    v     = S.shape[0]            # Amount of voxels
    d     = S.shape[1]            # Amount of spatial components
    k     = M.shape[1]            # Amount of temporal components
    N_t   = B.shape[1]            # Number of timepoints
    
    # Compute M_t (from the noiseless reconstruction of X)
    M_t          = np.zeros([d,k, N_t])
    for t in range(N_t):
        M_t[:,:,t]=np.dot(   np.dot(   np.transpose(np.matrix(np.dot(A,Xr[:,t]))),  np.matrix(BT[t,:])), Z)
   
    # Compute M by summing over the time dimension of M_t
    Mtsum = np.dot(np.dot(np.dot(A, Xr), BT), Z)
    
    ## COMPUTE F 
    # ---------------------------------------------------------
    f = N_t*M_t
    
    return Xr, M_t, Mtsum, f


# In[ ]:




