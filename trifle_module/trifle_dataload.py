#!/usr/bin/env python

# #### TRIFLE MODULE FOR LOADING DATA ---------------------------------------
# Version d.d. 10-06-2025 by TJ de Kloe

## LOAD MODULES 
import numpy as np
import nibabel as nib
from pathlib import Path
import scipy.stats as ss

np.seterr(invalid='ignore')

def load_X(Xfilename):
    """Load preprocessed 4D fMRI data and reshape to 2D (voxels x time)."""
    try:
        data_img  	= nib.load(Xfilename)
        data4d  	= data_img.get_fdata()
    except Exception as e:
        raise RuntimeError(f"Could not load X from {Xfilename}: {e}")

    if data4d.ndim 	!= 4:
        raise ValueError(f"{Xfilename} is not a 4D image.")
        
    data2d  		= data4d.reshape((-1, data4d.shape[3]))
    print(f"The 4D shape of X is {data4d.shape}")
    print(f"The reshaped 2D shape of X is {data2d.shape}")
    return data2d, data2d.shape

def load_S(Sfilename):
    """Load 4D spatial ICA map and reshape to 2D (voxels x components)."""
    try:
        spatial4d  	= nib.load(Sfilename).get_fdata()
    except Exception as e:
        raise RuntimeError(f"Could not load S from {Sfilename}: {e}")

    if spatial4d.ndim 	!= 4:
        raise ValueError(f"{Sfilename} is not a 4D image.")

    spatial2d  		= spatial4d.reshape((-1, spatial4d.shape[3]))
    print(f"The 4D shape of S is {spatial4d.shape}")
    print(f"The reshaped 2D shape of S is {spatial2d.shape}")
    return spatial2d, spatial2d.shape

def mask(maskfile, data2d, spatialmap2d):
    """Apply a brain mask to both data and spatial map."""
    try:
        mask_data  	= nib.load(maskfile).get_fdata()
    except Exception as e:
        raise RuntimeError(f"Could not load mask from {maskfile}: {e}")

    if mask_data.ndim 	!= 3:
        raise ValueError(f"{maskfile} is not a 3D mask.")

    flat_mask 		= mask_data.reshape(-1)
    brain_voxels 	= flat_mask != 0

    X  			= data2d[brain_voxels, :]
    S  			= spatialmap2d[brain_voxels, :]

    print(f"Original mask shape: {mask_data.shape}")
    print(f"Number of brain voxels: {np.sum(brain_voxels)}")
    print(f"Masked X shape: {X.shape}")
    print(f"Masked S shape: {S.shape}")
    return brain_voxels, X, X.shape, S, S.shape

def load_TMB(Tfilename, Mfilename, Bfilename):
    """Load and z-score temporal components T, M, B."""
    T  			= np.loadtxt(Tfilename)
    Tz			= ss.zscore(T, axis=0)
    print(f"T shape: {T.shape}")

    M  			= np.loadtxt(Mfilename)
    print(f"M shape: {M.shape}")

    B  			= np.loadtxt(Bfilename).T  # Transpose to get k x time
    Bz  		= ss.zscore(B, axis=1)
    print(f"B shape: {B.shape}")
    print(f"Bz shape: {Bz.shape}")

    return T, Tz, M, B, Bz

