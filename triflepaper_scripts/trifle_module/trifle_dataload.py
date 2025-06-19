#!/usr/bin/env python
# coding: utf-8

# #### TRIFLE MODULE FOR LOADING DATA ---------------------------------------
# Version d.d. 09-01-2024 by TJ de Kloe

# In[ ]:
def load_X(Xfilename):
    # Load packages
    import nibabel as nib
    import numpy as np 
    import sys
    np.seterr(invalid='ignore')
    
    # Load data 
    try:
        dataImg        = nib.load(Xfilename)
        data4d         = dataImg.get_fdata()
        data4d_shape   = data4d.shape
        print('The 4d shape of the data is', data4d_shape)
        
        data2d             = np.reshape(data4d,(np.prod(data4d_shape[0:3]),data4d_shape[3]))
        data2d_shape       = data2d.shape
        print('The 2d shape of the data is', data2d_shape) 
    except:
        sys.exit('Cannot open ' + Xfilename | '\nExiting.')
    if len(data4d.shape) != 4:
        sys.exit(Xfilename + ' is not a 4D image\nExiting.')
    
    # Reshape data to 2D

    return data2d, data2d_shape


# In[ ]:

def load_S(Sfilename):
    import nibabel as nib
    import numpy as np 
    np.seterr(invalid='ignore')
    
    # Load data 
    spatialmap4d        = nib.load(Sfilename) # Load data in 4D (x,y,z,t)
    spatialmap4d        = spatialmap4d.get_fdata()
    spatialmap4d_shape  = spatialmap4d.shape
    print('The 4d shape of S is', spatialmap4d_shape)
    
    # Reshape data to 2D
    spatialmap2d        = np.reshape(spatialmap4d,(np.prod(spatialmap4d_shape[0:3]),spatialmap4d_shape[3])) # Reshape to 2D (v,t)
    spatialmap2d_shape  = spatialmap2d.shape
    print('The 2d shape of S is', spatialmap2d_shape) 
    
    return spatialmap2d, spatialmap2d_shape


# In[ ]:

def mask(maskfile, data2d, spatialmap2d):
    # Import packages
    import nibabel as nib
    import numpy as np 
    
    # Mask data 
    mask3d              = nib.load(maskfile)
    mask3d              = mask3d.get_fdata()
    print('The original shape of the mask is', mask3d.shape)
    mask                = np.reshape(mask3d, [np.prod(mask3d.shape)])
    print('The shape of the reshaped mask is', mask.shape)
    brain_voxels        = mask!= 0
    print('The amount of brain voxels is', np.sum(brain_voxels))
    X                   = data2d[brain_voxels,:]
    X_shape             = X.shape
    S                   = spatialmap2d[brain_voxels,:] 
    S_shape             = S.shape
    print('The shape of the masked data (X) is', X_shape)
    print('The shape of the masked spatial map (S) is', S_shape) 
    
    return brain_voxels, X, X_shape, S, S_shape


# In[ ]:

def load_TMB(Tfilename, Mfilename, Bfilename):
    # Import packages
    import numpy as np 
    import scipy.stats as ss
    
    ## Import layer 1 time series and Z-score:
    # --------------------------------------------------------------
    T                   = np.loadtxt(Tfilename)
    Tz                  = ss.zscore(T, axis=0) 
    print('The shape of T is', T.shape)
    
    ## LOAD M (Weights, d x k) AND B (tIC, k x t) from layer 2 output (tfmpub folders) and z-score B
    # --------------------------------------------------------------
    M                   = np.loadtxt(Mfilename) # TFM core mixing matrix
    print('The shape of M is', M.shape)
    B                   = np.loadtxt(Bfilename) # TFM timeseries
    B                   = np.transpose(B)
    print('The shape of B is', B.shape)
    Bz                  = ss.zscore(B, axis=1)
    print('The size of Bz is:', np.shape(Bz))
    
    return T, Tz, M, B, Bz

