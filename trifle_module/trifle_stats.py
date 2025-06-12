#!/usr/bin/env python

# #### TRIFLE MODULE FOR STATS ---------------------------------------
# Version d.d. 10-06-2025 by TJ de Kloe

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

def into_trials(onsets, regressors, TR, epochsize, N_t, f_tfm):
    """
    Slice time series and task regressors into trial-based epochs.

    Parameters:
    - onsets (np.ndarray)	: Onset times in seconds, shape (regressors × trials).
    - regressors (np.ndarray)	: HRF-convolved task regressors, shape (regressors × time).
    - TR (float)		: Repetition time in seconds.
    - epochsize (int)		: Number of TRs per epoch.
    - N_t (int)			: Total number of time points.
    - f_tfm (np.ndarray)	: TFM-specific f matrix (d × timepoints).

    Returns:
    - f_tfm_epochs (np.ndarray)	: f matrix sliced into trials, shape (d × valid_trials × epochsize).
    - regressors_epochs (np.ndarray): regressor time series sliced into trials, shape (regressors × valid_trials × epochsize).
    - Ndel (int) 		: Number of excluded trials due to NaNs or overflow.
    """
    import numpy as np
    import pandas as pd

    Nreg  			= regressors.shape[0]
    Ntrials  			= onsets.shape[1]

    # Convert onsets to frame indices
    onsets_frames  		= onsets / TR
    mask_nans  			= np.isnan(onsets_frames)
    sumnans  			= np.sum(mask_nans)
    onsets_frames[mask_nans]  	= np.nan
    onsets_frames_df  		= pd.DataFrame(onsets_frames)
    onsets_frames_df  		= onsets_frames_df.dropna(axis=1)

    epoch_start  		= np.round(onsets_frames_df.iloc[0, :]).astype(int).to_numpy()
    epoch_end  			= (epoch_start + epochsize).astype(int)

    # Filter out epochs that exceed the number of time points
    valid_mask  		= epoch_end <= N_t
    epoch_start  		= epoch_start[valid_mask]
    epoch_end  			= epoch_end[valid_mask]
    valid_trials  		= len(epoch_start)
    Ndel  			= Ntrials - valid_trials

    # Initialise output
    f_tfm_epochs  		= np.zeros((f_tfm.shape[0], valid_trials, epochsize))
    regressors_epochs  		= np.zeros((Nreg, valid_trials, epochsize))

    # Slice f_tfm into epochs
    for i, (start, end) in enumerate(zip(epoch_start, epoch_end)):
        f_tfm_epochs[:, i, :]  	= f_tfm[:, start:end]

    # Slice regressors into epochs
    for reg in range(Nreg):
        for i, (start, end) in enumerate(zip(epoch_start, epoch_end)):
            regressors_epochs[reg, i, :] = regressors[reg, start:end]

    return f_tfm_epochs, regressors_epochs, Ndel

def subtract_staticM(static_M, maxcor, f_epochs):
    """
    Subtract the static mixing matrix M from each time point of the f-epochs.

    Parameters:
    - static_M (np.ndarray) 	: Static M matrix, shape (d, k)
    - maxcor (float) 	 	: Correlation of the TFM time series with the task
    - f_epochs (np.ndarray)	: f matrix sliced into epochs, shape (d, trials, epochsize)

    Returns:
    - f_epochs_minM (np.ndarray): f with M removed, same shape as input
    - f_average (np.ndarray)	: Average over trials, shape (d, epochsize)
    - f_se (np.ndarray)		: Standard error over trials, shape (d, epochsize)
    """
    import numpy as np

    if maxcor < 0:
        f_epochs  		= -f_epochs
        static_M  		= -static_M

    Nt              		= f_epochs.shape[1]
    
    ## SUBTRACT M
    # --------------------------------------------------
    step1           		= np.squeeze(np.dstack([static_M]*Nt))
    m               		= np.squeeze(np.dstack([step1]*f_epochs.shape[2] ))
         
    f_epochs_minM   		= np.add(f_epochs,-m) 
    f_average       		= np.mean(f_epochs_minM,1)
    f_se            		= np.std(f_epochs_minM,1)/np.sqrt(Nt)
    
    return f_epochs_minM, f_average, f_se


from scipy.stats import sem

def trialaverage(f_epochs):
    """
    Compute the trial-averaged time course and error metrics.

    Parameters:
    - f_epochs (np.ndarray) 	: f sliced into epochs, shape (d, trials, epochsize)

    Returns:
    - f_average (np.ndarray) 	: Mean across trials, shape (d, epochsize)
    - f_std (np.ndarray) 	: Standard deviation across trials, shape (d, epochsize)
    - f_sem (np.ndarray) 	: Standard error of the mean, shape (d, epochsize)
    """
    f_average = np.mean(f_epochs, axis=1)
    f_std     = np.std(f_epochs, axis=1)
    f_sem     = sem(f_epochs, axis=1)

    return f_average, f_std, f_sem

def r2fisherz(r):
    import math
    z = .5*(math.log(1+r) - math.log(1-r))
    return z 

def fisherz2r(z):
    import math
    r_z = (math.exp(2 * z) - 1) / (math.exp(2 * z) + 1)
    return r_z

def run_cors2d(regressors, timeseries):
    """
    Compute Pearson correlations between task regressors and 2D time series data.

    Parameters:
    - regressors (np.ndarray) 	: Task regressors, shape (n_regressors, time)
    - timeseries (np.ndarray) 	: Timeseries data, shape (n_sources, time)

    Returns:
    - cor (np.ndarray) 		: Correlation coefficients, shape (n_regressors, n_sources)
    - pvals (np.ndarray) 	: Two-tailed p-values, same shape
    """
    import numpy as np
    from scipy.stats import pearsonr

    n_regressors, n_sources  	= regressors.shape[0], timeseries.shape[0]
    cor    			= np.zeros((n_regressors, n_sources))
    pvals  			= np.zeros((n_regressors, n_sources))

    for r in range(n_regressors):
        for s in range(n_sources):
            cor[r, s], pvals[r, s] = pearsonr(regressors[r], timeseries[s])

    return cor, pvals

def run_cors3d(regressors, timeseries):
    """
    Compute Pearson correlations between each regressor and a 3D time series 
    (e.g. d × trials × time) reshaped to 2D.

    Parameters:
    - regressors (np.ndarray): Array of shape (n_regressors, time)
    - timeseries (np.ndarray): Array of shape (d, trials, time)

    Returns:
    - cor (np.ndarray): Correlation coefficients, shape (n_regressors, d * trials)
    - pvals (np.ndarray): Corresponding p-values, same shape
    """
    import numpy as np
    from scipy.stats import pearsonr

    n_regressors = regressors.shape[0]
    d, trials, time = timeseries.shape

    # Reshape to 2D: (d × trials, time)
    flat_timeseries = timeseries.reshape((d * trials, time), order='F')

    # Prepare outputs
    cor   = np.zeros((n_regressors, d * trials))
    pvals = np.zeros((n_regressors, d * trials))

    # Correlate each source-trial with each regressor
    for r in range(n_regressors):
        for s in range(d * trials):
            cor[r, s], pvals[r, s] = pearsonr(regressors[r], flat_timeseries[s])

    return cor, pvals


