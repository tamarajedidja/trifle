{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### MODULE FOR TRIFLE ANALYSIS ---------------------------------------\n",
    "Version d.d. 09-01-2024 by TJ de Kloe"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def run_tICA(T):\n",
    "    from sklearn.decomposition import FastICA\n",
    "    import numpy as np \n",
    "    import sys\n",
    "    import errno\n",
    "    np.seterr(invalid='ignore')\n",
    "    \n",
    "    # THIS FUNCTION RUNS TEMPORAL ICA\n",
    "    # Inputs are: \n",
    "    # - T: Spatial time series (DIM: time x parcels, t x parcels)\n",
    "    if T.shape[1] < T.shape[0]:\n",
    "        sys.exit(infile + ' Make sure that your input has dimensions time x parcels\\nExiting.')\n",
    "    \n",
    "    # Variance normalise\n",
    "    T -= np.nanmean(T,axis=0)\n",
    "    T /= np.nanstd(T, axis=0)\n",
    "    \n",
    "    #Algorithm \"parallel\" is faster but converges less often\n",
    "    tfm_ica = FastICA(n_components=3, max_iter=20000, tol=0.00001, fun=\"logcosh\", algorithm=\"deflation\", random_state=None) \n",
    "    \n",
    "    # 2) Fit temporal ICA\n",
    "    tfms    = tfm_ica.fit_transform(T)\n",
    "    M       = tfm_ica.mixing_\n",
    "    B       = tfms.T\n",
    "    return M, B "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "# COMPUTE M_t/ f\n",
    "def compute_tvM(S, M, B):\n",
    "    # THIS FUNCTION COMPUTES THE M_t matrix \n",
    "    # Inputs are:\n",
    "    # - S: Parcels (DIM: voxels x spatial components/parcels, i.e. v x d)\n",
    "    # - M: TFM Mixing matrix (DIM: spatial components/parcels by temporal components, i.e. d x k)\n",
    "    # - B: TFM time-series (DIM: temporal components by frames/time, i.e. k x t)\n",
    "    \n",
    "    ## CONCEPTUAL OVERVIEW OF THE DATA:\n",
    "    # X_{v x t} = S_{v x d} * M_{d x k} * B_{k * t}\n",
    "    \n",
    "    ## LOAD PACKAGES\n",
    "    import numpy as np\n",
    "    import numpy.linalg as npl\n",
    "    \n",
    "    ## ALGEBRAIC COMPUTATION OF M_T\n",
    "    # X_{v x t} = S_{v x d} * M_{d x k} * B_{k x t} \n",
    "    # pinv(S) = A\n",
    "    # A * X * BT * (B*BT)^-1 = M_t\n",
    "    # (B*BT)^-1 = Z\n",
    "    # A * X * B^T * Z = M_t\n",
    "    Xr    = np.dot(np.dot(S,M),B) # Compute noiseless reconstruction of the data\n",
    "    A     = npl.pinv(S)     # Compute A, the pseudo inverse of S\n",
    "    BT    = np.transpose(B)       # Compute BT, the transpose of B\n",
    "    BBT   = np.dot(B,BT)          # Compute BBT, B times its' transpose\n",
    "    Z     = npl.inv(BBT)    # Compute Z, the inverse of BBT\n",
    "    \n",
    "    # Compute shapes:\n",
    "    v     = S.shape[0]            # Amount of voxels\n",
    "    d     = S.shape[1]            # Amount of spatial components\n",
    "    k     = M.shape[1]            # Amount of temporal components\n",
    "    N_t   = B.shape[1]            # Number of timepoints\n",
    "    \n",
    "    # Compute M_t (from the noiseless reconstruction of X)\n",
    "    M_t          = np.zeros([d,k, N_t])\n",
    "    for t in range(N_t):\n",
    "        M_t[:,:,t]=np.dot(   np.dot(   np.transpose(np.matrix(np.dot(A,Xr[:,t]))),  np.matrix(BT[t,:])), Z)\n",
    "   \n",
    "    # Compute M by summing over the time dimension of M_t\n",
    "    Mtsum = np.dot(np.dot(np.dot(A, Xr), BT), Z)\n",
    "    \n",
    "    ## COMPUTE F \n",
    "    # ---------------------------------------------------------\n",
    "    f = N_t*M_t\n",
    "    \n",
    "    return Xr, M_t, Mtsum, f"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
