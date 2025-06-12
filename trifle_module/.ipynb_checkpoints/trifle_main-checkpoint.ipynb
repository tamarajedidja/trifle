{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### MAIN SCRIPT FOR RUNNING TRIFLE ---------------------------------------\n",
    "Version 09-01-2024 by TJ de Kloe"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "def main(Xfilename, Sfilename, maskfilename, Tfilename, Mfilename, Bfilename):\n",
    "    ## LOAD X (data, v x t) \n",
    "    # cleaned, demeaned, normalised\n",
    "    # with \"dataload_module\" \n",
    "    # -------------------------------------------------------------- \n",
    "    data2d = trifle_dataload.load_X(Xfilename)\n",
    "    \n",
    "    ## LOAD S (parcels, v x d) \n",
    "    # --------------------------------------------------------------\n",
    "    spatialmap2d = trifle_dataload.load_S(Sfilename)\n",
    "\n",
    "    ## MASK WITH DR MASK\n",
    "    # --------------------------------------------------------------\n",
    "    X, S = trifle_dataload.mask(maskfilename, data2d, spatialmap2d)\n",
    "    \n",
    "    ## LOAD STAGE 1 TIME SERIES (T), MIXING MATRIX (M) & TFM TIME SERIES (B) \n",
    "    # and z-score time series\n",
    "    # --------------------------------------------------------------\n",
    "    T, Tz, M, B, Bz = trifle_dataload.load_TMB(Tfilename, Mfilename, Bfilename)\n",
    "    \n",
    "    ## TV-TFMS\n",
    "    # ---------------------------------------------------------\n",
    "    # COMPUTE TIME VARYING MATRIX, with \"trifles module\"\n",
    "    Xr, M_t, Mtsum, f  = trifle_analysis.computeM_t(S, M, Bz)\n",
    "    \n",
    "    return X, S, Tz, M, Bz, Xr, M_t, f"
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
