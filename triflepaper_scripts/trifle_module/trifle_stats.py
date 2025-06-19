#!/usr/bin/env python
# coding: utf-8

# #### MODULE FOR TRIFLE STATS ---------------------------------------
# Version d.d. 09-01-2024 by TJ de Kloe

# In[1]:


## LOAD MODULES 
import numpy as np
import numpy.linalg as npl
import pandas as pd
import scipy
import scipy.stats as ss
import statsmodels.api as sm
import nipype.interfaces.fsl as fsl 
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:

def into_trials(onsets, regressors, TR, epochsize, N_t, f_tfm):
    # GOAL:  Slice time series and task regressors into trials. 
    # INPUT: 
    # 1. onsets (regressors x trials) *in seconds*; ensure chronological order of regressors columnwise (visual stim in column before response eg)
    # 2. regressors (regressors x time points): the onsets convolved with the HRF 
    # 3. TR: *in seconds*
    # 4. f_tfm: time-varying mixing matrix of 1 specific TFM, hence of size: spatial parcels (d) x timepoints 
    
    # Import packages
    import numpy as np 
    import pandas as pd 
    
    # Get sizes
    Nreg             = regressors.shape[0];   
    Ntrials          = onsets.shape[1]; 

    # Get onsets in frames (instead of time) 
    onsets_frames    = onsets/TR; 
    mask_nans        = np.isnan(onsets); sumnans = np.sum(mask_nans); 
    onsets_frames[mask_nans] = np.nan; 
    onsets_frames    = pd.DataFrame(onsets_frames)
    onsets_frames    = onsets_frames.dropna(axis=1); 

    # Define start and end of epochs
    epoch_start       = (np.round(onsets_frames.iloc[0,:],0)).astype(int); epoch_start = epoch_start.to_numpy()
    epoch_ends        = (epoch_start + epochsize).astype(int)

    # Initialize output dataframes
    Ndel              = np.sum(epoch_ends > N_t); Ndel = Ndel + sumnans
    f_tfm_epochs      = np.zeros([f_tfm.shape[0], Ntrials-Ndel, epochsize])
    regressors_epochs = np.zeros([Nreg, Ntrials-Ndel, epochsize])
    
    # f_tfm into trials 
    for di in range(f_tfm.shape[0]):
        for ti in range(Ntrials-Ndel):
            f_tfm_epochs[di,ti,:] = f_tfm[di,epoch_start[ti]:epoch_ends[ti]]
        del ti
    del di

    # regressors into trials 
    for reg_idx in range(Nreg):
        for trial_idx in range(Ntrials-Ndel):
            regressors_epochs[reg_idx, trial_idx, :] = regressors[reg_idx, epoch_start[trial_idx]:epoch_ends[trial_idx]]
        del trial_idx
        
    ## POSSIBLY CHANGE TO: WITHIN TRIALS AND BETWEEN TRIALS DESIGN >> SEPERATE REGS?
    
    return f_tfm_epochs, regressors_epochs, Ndel

# In[ ]:

def subtract_staticM(static_M, maxcor, f_epochs):
    import numpy as np 
    
    # GOAL: Slice time series and task regressors into trials. 
    # INPUT: 
    # 1. static_M: M, shape: d, parcels x k,tfms
    # 2. maxcor: correlation value of the TFM time series to the task 
    # 3. f_epochs: f slices into epochs with "into_trials", shape: d x trials x epochsize
    
    ## RECODE IF COR TFM WAS NEGATIVE (WITH VISUAL REGRESSOR)
    # --------------------------------------------------
    if maxcor < 0:
        f_epochs = - f_epochs
        static_M = - static_M
    
    Nt             = f_epochs.shape[1]
    
    ## SUBTRACT M
    # --------------------------------------------------
    step1          = np.squeeze(np.dstack([static_M]*Nt))
    m              = np.squeeze(np.dstack([step1]*f_epochs.shape[2] ))
         
    f_epochs_minM  = np.add(f_epochs,-m) 
    f_average      = np.mean(f_epochs_minM,1)
    f_se           = np.std(f_epochs_minM,1)/np.sqrt(Nt)
    
    return f_epochs_minM, f_average, f_se


def trialaverage(f_epochs):
    from scipy.stats import sem
    # GOAL: Slice time series and task regressors into trials. 
    # INPUT: 
    # 1. static_M: M, shape: d, parcels x 1 (tfm of interest)
    # 2. maxcor: correlation value of the TFM time series to the task 
    # 3. f_epochs: f slices into epochs with "into_trials", shape: d x trials x epochsize
    
    f_average      = np.mean(f_epochs,axis=1)
    f_std          = np.std(f_epochs,axis=1)
    f_sem          = sem(f_epochs,axis=1)
    
    return f_average, f_std, f_sem

# ###### Additional functions for exploratory analyses:

# In[ ]:

def r2fisherz(r):
    import math
    z = .5*(math.log(1+r) - math.log(1-r))
    return z 

def fisherz2r(z):
    import math
    r_z = (math.exp(2 * z) - 1) / (math.exp(2 * z) + 1)
    return r_z

def run_cors2d(regressors, timeseries):
    import scipy.stats as ss
    import numpy as np 
    
    # GOAL: Correlate 2D (e.g. DR, of TFM) timeseries to task 
    # INPUT: 
    # 1. regressors, shape: regressors x time
    # 2. timeseries, shape: parcels x time 
    # OUTPUT:
    # 1. Correlation matrix
    # 2. P-values matrix 
    d             = timeseries.shape[0] 
    Nreg          = regressors.shape[0]
    cor           = np.zeros([Nreg,d]);
    pvals         = np.zeros([Nreg,d]);
    
    for d_idx in range(d):
        for reg_idx in range(Nreg):
            cor[reg_idx,d_idx], pvals[reg_idx,d_idx] = ss.pearsonr(timeseries[d_idx,:], regressors[reg_idx,:])        
    return cor, pvals

def run_cors3d(regressors, timeseries):
    import scipy.stats as ss
    import numpy as np 
    
    # GOAL: Correlate 3D (e.g. DR, of TFM) timeseries to task (reshape into 2D + correlate)
    # INPUT: 
    # 1. regressors, shape: regressors x time
    # 2. timeseries, shape: parcels x time 
    # OUTPUT:
    # 1. Correlation matrix
    # 2. P-values matrix 
    d              = timeseries.shape[0] 
    Nreg           = regressors.shape[0]
    dim1           = np.prod(timeseries.shape[0:2])
    nFrames        = timeseries.shape[2]
    timeseries2D   = np.reshape(timeseries, (dim1,nFrames), order='F')
    cor            = np.zeros([Nreg,dim1]);
    pvals          = np.zeros([Nreg,dim1]);
    
    for d_idx in range(dim1):
        for reg_idx in range(Nreg):
            cor[reg_idx,d_idx], pvals[reg_idx,d_idx] = ss.pearsonr(timeseries2D[d_idx,:], regressors[reg_idx,:])
    return cor, pvals

