#!/usr/bin/env python
# coding: utf-8

# #### MODULE FOR TRIFLE ANALYSIS ---------------------------------------
# Version d.d. 10-06-2025 by TJ de Kloe

from numpy.linalg import det, cond, LinAlgError
import numpy as np

def run_tICA(T, modelorder, seed):
    """
    Run Temporal ICA using FastICA on spatial ICA time series.

    Parameters:
    - T (array): time × spatial components matrix
    - modelorder (int): number of temporal components (TFMs)
    - seed (int): random state

    Returns:
    - M (array): mixing matrix (spatial components × TFMs)
    - B (array): temporal components (TFMs × time)
    - tfm_ica (FastICA object): fitted ICA object
    """
    from sklearn.decomposition import FastICA
    import sys

    if T.shape[0] < T.shape[1]:
        sys.exit("Expected T with shape (time x spatial components). Try transposing.\nExiting.")

    tfm_ica  	= FastICA(n_components=modelorder, max_iter=100000, tol=1e-4,fun="logcosh", algorithm="deflation", random_state=seed, whiten=True)
    tfms  	= tfm_ica.fit_transform(T)
    M  		= tfm_ica.mixing_
    B  		= tfms.T
    return M, B, tfm_ica

def pseudoinverse(A, verbose=True):
    """
    Compute the Moore-Penrose pseudoinverse of a matrix.
    Uses A⁺ = (AᵀA)⁻¹ Aᵀ if full column rank, otherwise SVD.

    Parameters:
        A (ndarray): Input matrix
        verbose (bool): Print method used

    Returns:
        ndarray: Pseudoinverse of A
    """
    rank = np.linalg.matrix_rank(A)
    num_columns = A.shape[1]

    if rank == num_columns:
        if verbose:
            print("Matrix has full column rank. Using (AᵀA)⁻¹ Aᵀ.")
        AtA = A.T @ A
        A_pinv = np.linalg.inv(AtA) @ A.T
    else:
        if verbose:
            print("Matrix does not have full column rank. Using SVD.")
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        S_inv = np.array([1/s if s > 1e-15 else 0 for s in S])
        A_pinv = Vt.T @ np.diag(S_inv) @ U.T

    return A_pinv


def matrix_properties(matrix, tolerance=1e-10, verbose=True):
    """
    Check and optionally print diagnostics for a matrix:
    - Full rank
    - Invertibility (via determinant)
    - Condition number

    Parameters:
    - matrix (np.ndarray)	: Input square matrix
    - tolerance (float) 	: Threshold for determinant-based invertibility
    - verbose (bool) 		: If True, print matrix diagnostics

    Returns:
    - dict: {
        'rank': int,
        'is_full_rank' 		: bool,
        'determinant' 		: float or None,
        'is_invertible'		: bool,
        'condition_number'	: float or None,
        'is_well_conditioned'	: bool
      }
    """
    import numpy as np

    props  			= {}
    rank  			= np.linalg.matrix_rank(matrix)
    props['rank'] 		= rank
    props['is_full_rank']	= rank == min(matrix.shape)

    # Determinant
    try:
        det_val  		= np.linalg.det(matrix)
        props['determinant']  	= det_val
        props['is_invertible'] 	= abs(det_val) > tolerance
    except np.linalg.LinAlgError:
        props['determinant']  	= None
        props['is_invertible'] 	= False

    # Condition number
    try:
        cond_val  		= np.linalg.cond(matrix)
        props['condition_number']= cond_val
        props['is_well_conditioned']= cond_val < 1e10
    except np.linalg.LinAlgError:
        props['condition_number']= None
        props['is_well_conditioned']= False

    # Optional print
    if verbose:
        print("\nMatrix Properties:")
        print(f"- Shape: {matrix.shape}")
        print(f"- Full Rank: {'Yes' if props['is_full_rank'] else 'No'}")
        print(f"- Rank: {rank}")
        if props['determinant'] is not None:
            print(f"- Determinant: {props['determinant']:.4e}")
        else:
            print("- Determinant: Not computable")
        print(f"- Invertible: {'Yes' if props['is_invertible'] else 'No'}")
        if props['condition_number'] is not None:
            print(f"- Condition Number: {props['condition_number']:.2e}")
            print(f"- Well-Conditioned: {'Yes' if props['is_well_conditioned'] else 'No'}")
        else:
            print("- Condition Number: Not computable")

    return props


def compute_tvM(S, M, B, check_condition=False):
    """
    Compute the time-varying mixing matrix M_t, a static reconstruction Mtsum, and the final f matrix used in TRIFLE.

    ---------------------------------------------------------
    CONCEPTUAL OVERVIEW OF THE DATA
    ---------------------------------------------------------
    The reconstructed data is a product of the ICA output such that: 
        X = S · M · B
    where:
        - X is (voxels × time)
        - S is (voxels × spatial components d)
        - M is (d × temporal components k)
        - B is (k × time)

    We compute a dynamic version of M, denoted M_t:
        M_t[:,:,t] = A @ X[:,t] @ B[t,:]^T @ (B @ B^T)^-1
    where:
        - A = pseudoinverse of S
        - B[t,:]^T is the time-specific row of B, transposed
        - (B @ B^T)^-1 is the (co)variance scaling term Z
    ---------------------------------------------------------

    Parameters:
        S (array) 	: spatial ICA spatial maps (voxels × components d)
        M (array) 	: temporal ICA mixing matrix (d × k)
        B (array) 	: temporal ICA TFM time series (k × t)
        check_condition (bool): if True, print matrix condition stats

    Returns:
        Xr (array) 	: reconstructed data
        M_t (array) 	: time-varying mixing matrix (d × k × t)
        Mtsum (array) 	: time-averaged M_t
        f (array) 	: N_t × (d × k) matrix 
    """
    import numpy.linalg as npl

    Xr  	 	= S @ M @ B  		# Compute noiseless reconstruction of the data
    A  		 	= pseudoinverse(S) 	# Compute A, the pseudo inverse of S
    BT  	 	= B.T			# Compute BT, the transpose of B
    BBT  	 	= B @ BT		# Compute BBT, B times its' transpose

    props = matrix_properties(BBT)
    if not props['is_invertible'] or not props['is_well_conditioned']:
        warnings.warn("BBT is poorly conditioned or not invertible. Results may be unstable.")

    Z  		 	= npl.inv(BBT) 		# Compute Z, the inverse of BBT
    v, d  	 	= S.shape
    k, N_t  	 	= B.shape

    M_t  	 	= np.zeros((d, k, N_t))
    for t in range(N_t):
        M_t[:, :, t]  	= (A @ Xr[:, t]).reshape(-1, 1) @ BT[t, :].reshape(1, -1) @ Z

    Mtsum  	 	= A @ Xr @ BT @ Z
    f  		 	= N_t * M_t
    return Xr, M_t, Mtsum, f




