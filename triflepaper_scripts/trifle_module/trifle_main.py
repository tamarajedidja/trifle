#!/usr/bin/env python
# coding: utf-8

# #### MAIN SCRIPT FOR RUNNING TRIFLE ---------------------------------------
# Version 09-01-2024 by TJ de Kloe

# In[1]:
def main(Xfilename, Sfilename, maskfilename, Tfilename, Mfilename, Bfilename):
    # TV-TFM module 
    import sys
    import os
    trifle_module_path = "/home/mrstats/tamdklo/code/trifle_module"
    sys.path.append(os.path.abspath(trifle_module_path)) 
    import trifle_dataload as trifle_dataload
    import trifle_analysis as trifle_analysis 
    ## LOAD X (data, v x t) 
    # cleaned, demeaned, normalised
    # with "dataload_module" 
    # -------------------------------------------------------------- 
    data2d, data2d_shape = trifle_dataload.load_X(Xfilename)
    
    ## LOAD S (parcels, v x d) 
    # --------------------------------------------------------------
    spatialmap2d, spatialmap2d_shape = trifle_dataload.load_S(Sfilename)

    ## MASK WITH DR MASK
    # --------------------------------------------------------------
    brain_voxels, X, X_shape, S, S_shape = trifle_dataload.mask(maskfilename, data2d, spatialmap2d)
    
    ## LOAD STAGE 1 TIME SERIES (T), MIXING MATRIX (M) & TFM TIME SERIES (B) 
    # and z-score time series
    # --------------------------------------------------------------
    T, Tz, M, B, Bz = trifle_dataload.load_TMB(Tfilename, Mfilename, Bfilename)
    
    ## TV-TFMS
    # ---------------------------------------------------------
    # COMPUTE TIME VARYING MATRIX, with "trifles module"
    Xr, M_t, Mtsum, f  = trifle_analysis.compute_tvM(S, M, Bz)
    
    return X, S, Tz, M, Bz, Xr, M_t, f


# In[ ]:




