#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MASTERSCRIPT FOR THE SMITH 20 PARCELLATION FOR THE FOLLOWING MANUSCRIPT: 
    "de Kloe, T. J., Fazal, Z., Kohn, N., Norris, D. G., Menon, R. S., Llera, A., & Beckmann, C. F. (2025). Time-Resolved Instantaneous Functional Loci Estimation (TRIFLE): Estimating Time-Varying Allocation of Spatially Overlapping Sources in Functional Magnetic Resonance Imaging. Imaging Neuroscience."
    
@author: Tamara Jedidja de Kloe, 2024

"""

#%% FUNCTIONS
# --------------------------------------------------------------
def load_taskdesign(filename_task, Nt, DERyesno):
    # Load task design
    design_full       = np.loadtxt(filename_task)
    design_full_z     = ss.zscore(design_full, axis=0)
    names_task        = ['Visual', 'Motor']
    names_task_der    = ['Visual', 'Der. Visual', 'Motor', 'Der. Motor']
    
    if DERyesno == 'no':
        task          = np.zeros([Nt,2])
        task[:,0]     = design_full_z[:,0]
        task[:,1]     = design_full_z[:,2]
        task_pd       = pd.DataFrame(task, columns = names_task)       

    elif DERyesno == 'yes':
        #Including derivatives
        task          = np.zeros([Nt,4])
        task[:,:4]    = design_full_z[:,:4]
        task_pd       = pd.DataFrame(task, columns = names_task_der)
 
    return task, task_pd

#%% LOAD PACKAGES
# --------------------------------------------------------------
import os.path as op
import os
import sys
from pathlib import Path
import argparse
import numpy as np 
import seaborn as sns
import scipy.stats as ss
import pandas as pd 
import matplotlib.pyplot as plt
import pickle
import statsmodels.api as sm
from patsy import dmatrices
from statsmodels.stats.multitest import multipletests

trifle_module_path = os.path.join(os.path.dirname(__file__), "trifle_module")
sys.path.append(os.path.abspath(trifle_module_path))
import trifle_main
import trifle_stats

#%% STUDY SPECIFICS
# --------------------------------------------------------------
# PARAMETERS
TR                  = .206
Nt                  = 3000                      #total number of timepoints per task run 
Nt_trial            = 60                        #amount of time points per epoch 
Nd                  = 20; Nk=15                 #spatial and temporal model order
Nses                = 40; Npp = 14              #ses and pps
framerange          = np.arange(0, Nt, 1)       #session frame range 
timerange           = np.arange(0, TR*Nt, TR)   #session time range
timerange_m         = timerange[:Nt_trial]      #epoch time range
ar_lag              = 5                         #autoregressive lag

## NAMES 
names_task          = ['Visual', 'Motor']
names_S20           = ['NWD (1)', 'Sensorimotor*','Auditory*','NWD (2)','NWD (3)',
                   'Medial visual (V1)*','DMN*','ECN*', 'Cerebellum*','Lateral visual (V3)*',
                   'NWD (IFG and Amygdala)', 'FP (Left)*' , 'FP (Right)*', 'NWD (4)', 'NWD (Insula network)', 
                   'Primary visual (V2)*','NWD (Hippocampal)','NWD (5)', 'NWD (6)', 'NWD (7)']
names_S10           = ['Medial Visual', 'Primary Visual', 'Lateral Visual', 'DMN', 'Cerebellum', 
                   'Sensorimotor', 'Auditory','Executive Control', 'Frontoparietal Left', 'Frontoparietal Right']
names_tfms          = ['TFM1', 'TFM2', 'TFM3', 'TFM4', 'TFM5', 'TFM6', 'TFM7', 'TFM8', 'TFM9', 'TFM10', 'TFM11', 'TFM12', 'TFM13', 'TFM14', 'TFM15']
names_subses        = ['03_01', '03_02', '03_03', '04_01', '04_02', '04_03', '05_01', '05_02', '05_03','06_01', '06_02', '06_03', '07_01', '07_02', '07_03', '08_01', '08_02', '08_03','09_01', '09_02', '09_03', '10_01', '10_02', '11_01', '11_02', '11_03','12_01', '12_02', '12_03', '13_01', '13_02', '13_03', '15_01', '15_02','16_01', '16_02', '16_03', '17_01', '17_02', '17_03']
sessions            = list(names_subses)
tfms                = list(names_tfms)

## PLOTTING SPECS
import matplotlib as mpl                        
mpl.rc('font', family='Futura Md BT') 
mpl.rcParams.update({'font.size': 11})
#Palette
cmap                = sns.diverging_palette(220, 10, as_cmap=True)
from matplotlib.colors import LinearSegmentedColormap
cmap_m              = LinearSegmentedColormap.from_list(name='test', colors=['blue','white','orange'])
#Colors
mycolors            = ['#A6381A','#2A9D8F', '#EFC560','#3C6A89'] 
colors_4            = ['#A6381A','#2A9D8F', '#EFC560','#3C6A89'] 
network_colors      = ["#2A9D8F", '#EFC560', '#C97459', '#3C6A89', '#A3589C', '#67AD80','#C75964', '#AD64B8', '#AD64B8', '#A6381A'] #'#e2472b'] #SM, AU, MedVis-V1, DMN, ECN, CER, LaterVis-V3, FP left, FP right, PrimVis-Vis 2, (Hipp)

# ROOT FOLDER PARSER
#TODO 

# PATHS
#experiment_dir     = Path(args.data_root).resolve()
experiment_dir      = Path('/project/3013060.04/TK_data/')
fmridata_dir        = experiment_dir / 'derivative-menon' / 'melodic'
stage1_dir          = experiment_dir / 'dr'
designs_dir         = experiment_dir / 'glm' / 'designs'

# Filenames
Xfilename           = 'filtered_func_data_denoised_norm2_unitvariance.nii.gz'
Sfilename           = 'dr_stage2_subject00000.nii.gz'
maskfile_name       = 'mask.nii.gz'

filenames_X         = {}; filenames_S = {}; filenames_mask = {}
filenames_T         = {}; filenames_M = {}; filenames_B = {}
filenames_task      = {} 

for sub_ses in sessions:
    subses_X        = Path(f"sub-{sub_ses}.ica")
    subses_stage1   = Path(f"DR_sub-{sub_ses}_s20.dr")
    stage2_dir      = stage1_dir / f"TFM_sub-{sub_ses}_dr_s20_c15.ica"
    subses_task     = Path(f"{sub_ses}.txt")
    
    filenames_X[sub_ses]     = fmridata_dir / subses_X / Xfilename
    filenames_S[sub_ses]     = stage1_dir / subses_stage1 / Sfilename
    filenames_mask[sub_ses]  = stage1_dir / subses_stage1 / maskfile_name
    filenames_M[sub_ses]     = stage2_dir / "melodic_unmix"
    filenames_B[sub_ses]     = stage2_dir / "melodic_mix"
    filenames_T[sub_ses]     = stage1_dir / subses_stage1 / "dr_stage1_subject00000.txt"
    filenames_task[sub_ses]  = designs_dir / subses_task

## IMPORT TASK DESIGNS
# ---------------------------------------------------------
task_dict           = {}; task_pd_dict = {}
for ses in sessions:
    task_dict[ses], task_pd_dict[ses] = load_taskdesign(filenames_task[ses], Nt, 'no')

#%% RUN MAIN (DATALOAD AND TRIFLE LAYER 3)
# ---------------------------------------------------------
X_dict  = {}
S_dict  = {}
T_dict  = {}; Tz_dict  = {}
M_dict  = {}
B_dict  = {}; Bz_dict   = {}
Xr_dict = {}
Mt_dict = {}
f_dict  = {}

for ses in sessions:
    X_dict[ses], S_dict[ses], Tz_dict[ses], M_dict[ses], Bz_dict[ses], Xr_dict[ses], Mt_dict[ses], f_dict[ses] = trifle_main.main(filenames_X[ses], filenames_S[ses], filenames_mask[ses], filenames_T[ses], filenames_M[ses], filenames_B[ses])

#%%#%% IMPORT PREVIOUSLY CREATED DATAFRAMES
# ---------------------------------------------------------
#TODO: remove import
#X_dict         = pickle.load(open('/project/3013060.04/TK_data/results_S20/X_dict.pickle',"rb")); 
S_dict          = pickle.load(open('/project/3013060.04/TK_data/results_S20/S_dict.pickle',"rb")); 
Tz_dict         = pickle.load(open('/project/3013060.04/TK_data/results_S20/Tz_dict.pickle',"rb")); 
M_dict          = pickle.load(open('/project/3013060.04/TK_data/results_S20/M_dict.pickle',"rb")); 
Bz_dict         = pickle.load(open('/project/3013060.04/TK_data/results_S20/Bz_dict.pickle',"rb")); 
#Xr_dict        = pickle.load(open('/project/3013060.04/TK_data/results_S20/Xr_dict.pickle',"rb")); 
Mt_dict         = pickle.load(open('/project/3013060.04/TK_data/results_S20/Mt_dict.pickle',"rb")); 
f_dict          = pickle.load(open('/project/3013060.04/TK_data/results_S20/f_dict.pickle',"rb")); 
#task_dict      = pickle.load(open('/project/3013060.04/TK_data/results_S20/task_dict.pickle',"rb")); 
#task_pd_dict   = pickle.load(open('/project/3013060.04/TK_data/results_S20/task_pd_dict.pickle',"rb")); 

#%% IDENITFY TFM TIMESERIES MOST STRONGLY RELATED TO THE TASK 
# ---------------------------------------------------------
Bz_corr_dict    = {}; Bz_pvals_dict     = {}; Bz_fisher_dict = {}; 
maxcor_tfm      = {}; maxcor_absvalue   = {}; maxcor_value   = {}; maxcor_idx     = {}

for ses_idx, ses in enumerate(sessions):   
    Bz_corr_dict[ses], Bz_pvals_dict[ses] = trifle_stats.run_cors2d(task_dict[ses].T, Bz_dict[ses])
    
    # Fisher Z transform
    Bz_fisher   = np.zeros(Bz_corr_dict[ses].shape)
    for reg in range(Bz_corr_dict[ses].shape[0]):
        for net in range(Bz_corr_dict[ses].shape[1]):
            Bz_fisher[reg, net] = trifle_stats.r2fisherz(Bz_corr_dict[ses][reg,net])
    del reg, net
    Bz_fisher_dict[ses] = pd.DataFrame(Bz_fisher, index=names_task, columns= names_tfms)
    
    # Find max 
    maxcor_tfm[ses]= abs(Bz_fisher_dict[ses].iloc[0,:]).idxmax()
    maxcor_absvalue[ses]= abs(Bz_fisher_dict[ses].iloc[0,:]).max()
    maxcor_idx[ses]= np.int(maxcor_tfm[ses][3:])-1
    maxcor_value[ses]= Bz_fisher_dict[ses].loc['Visual', maxcor_tfm[ses]]
    del ses_idx, ses

#%% SELECT TASK-POSITIVE TFM; SUBTRACT STATIC M AND REVERSE TIME SERIES IF NEEDED*
# *)Note, direction is not meaningful (i.e., depending on the direction of M) Hence, reversed in case of a negative correlation with the task design.
# ---------------------------------------------------------
f_tpm_dict      = {}; M_rev = {}

for ses in sessions:
    tfm_txt     = maxcor_tfm[ses]
    tfm_num     = int(tfm_txt[3:])-1
    
    f           = f_dict[ses][:,tfm_num,:]
    N_t         = f.shape[1] #number of trials 
    M           = M_dict[ses]
    static_M    = M[:,tfm_num]
    
    ## REVERT TIMESERIES IF NEEDED 
    maxcor      = maxcor_value[ses]
    if maxcor < 0:
        f       = - f
        static_M= - static_M
    
    ## SUBTRACT M
    step1       = np.squeeze(np.dstack([static_M]*N_t))
    f_minM      = np.add(f,-step1)
    
    f_tpm_dict[ses] = f_minM
    M_rev[ses]  = static_M
    del tfm_txt, tfm_num, f, N_t, M, static_M, maxcor, step1, f_minM

#%% EPOCH 
# ---------------------------------------------------------
f_epochs_dict               = {}
regressors_epochs_dict      = {}
Ndel_dict                   = {}

for sub_ses in sessions:
    behavior                = np.loadtxt(op.join('/project/3013060.04/TK_data/behavioral/Behavioral_sub_'+str(sub_ses[:2])+'_ses_'+str(sub_ses[3:])+'.txt'), delimiter="\t", skiprows=1)
#TODO: remove local import
    onsets                  = behavior[:,:2].T;    
    regressors              = task_dict[sub_ses].T
    tfm_num                 = maxcor_idx[sub_ses];
    
    f_epochs_dict[sub_ses], regressors_epochs_dict[sub_ses], Ndel_dict[sub_ses] = trifle_stats.into_trials(onsets, regressors, TR, Nt_trial, Nt, f_tpm_dict[sub_ses])

#%%% TRIAL AVERAGE
# ---------------------------------------------------------------------------------------------------------------
f_average_dict  = {}; f_sem_dict = {}; regs_average_dict = {}

for sub_ses in sessions:
    f_average_dict[sub_ses] = np.mean(f_epochs_dict[sub_ses],axis=1)
    f_sem_dict[sub_ses]     = np.std(f_epochs_dict[sub_ses],axis=1)/np.sqrt(f_epochs_dict[sub_ses].shape[1])
    regs_average_dict[sub_ses]= np.mean(regressors_epochs_dict[sub_ses],axis=1) 

#%% Figure 3: RANKING PLOTS
# ---------------------------------------------------------
f_minM_dict                 = pickle.load(open('/project/3013060.04/TK_data/results_S20/f_minM_dict_whole.pickle',"rb")); 
#TODO: remove local import
sub_ses                     = '11_03'
behavior                    = np.loadtxt(op.join('/project/3013060.04/TK_data/behavioral/Behavioral_sub_'+str(sub_ses[:2])+'_ses_'+str(sub_ses[3:])+'.txt'), delimiter="\t", skiprows=1)
#TODO: remove local import
onsets                      = behavior[:,0].T
onset_frame                 = np.round((onsets[0]/TR),0) ; print(onset_frame)
task                        = task_dict[sub_ses]

### SINGLE TRIAL RANKS
#i.e., highest ranking network per epoch time point plotted on task regressor
f_tfm3                      = f_minM_dict[sub_ses]
f_tfm3_pd                   = pd.DataFrame(f_tfm3, index= names_S20)   ## CHOOSE: -M OR NOT 
f_tfm3_10                   = np.zeros([10,Nt]) #select only smith10
smith10                     = [1,2,5,6,7,8,9,11,12,15] #SM, AU, V1, DMN, ECN, CER, V3, FP left, FP right, Vis 2
for i in range(10):
    sel         = smith10[i]
    f_tfm3_10[i,:]= f_tfm3_pd.iloc[sel,:]
    del sel;
f_tfm3_10                   = pd.DataFrame(f_tfm3_10)
f_tfm3_ranks                = f_tfm3_10.rank(axis=0, ascending=True); 

I_highest_rank  = np.zeros(3000)
for t in range(3000):
    I_highest_rank[t]       = np.where(f_tfm3_ranks.iloc[:,t] == np.amax(f_tfm3_ranks.iloc[:,t]))[0]

# Fig. 3. Panel A: Single Trial Ranks
plt.figure(figsize=(5.8, 5.2))
plt.subplot(2,1,1)
plt.title('Single Trial: Highest Ranking Network per Timepoint', font = 'Futura Hv BT', fontweight="bold", fontsize=12)
for t_idx, t in enumerate(range(21,81)):
    if  np.isfinite(I_highest_rank[t]):
        if I_highest_rank[t] == 3: #DMN
            plt.scatter(timerange[t_idx],task[t,0], color= network_colors[I_highest_rank[t].astype(int)], s=15)
        elif I_highest_rank[t] == 0: # SM
            plt.scatter(timerange[t_idx],task[t,0], color= network_colors[I_highest_rank[t].astype(int)], s=15)
        elif I_highest_rank[t] == 6: # lateral vis 3
            plt.scatter(timerange[t_idx],task[t,0], color= network_colors[I_highest_rank[t].astype(int)], s=15)
        elif I_highest_rank[t] == 9: # primary vis 2
            plt.scatter(timerange[t_idx],task[t,0], color= network_colors[I_highest_rank[t].astype(int)], s=15)
        elif I_highest_rank[t] == 2: # medial vis 1
            plt.scatter(timerange[t_idx],task[t,0], color= network_colors[I_highest_rank[t].astype(int)], s=15)
        elif I_highest_rank[t] == 1: #auditory
            plt.scatter(timerange[t_idx],task[t,0], color= network_colors[I_highest_rank[t].astype(int)], s=15)
        elif I_highest_rank[t] == 7: #frontoparietal
            plt.scatter(timerange[t_idx],task[t,0], color= network_colors[I_highest_rank[t].astype(int)], s=15)
        elif I_highest_rank[t] == 8: #frontoparietal right
            plt.scatter(timerange[t_idx],task[t,0], color= network_colors[I_highest_rank[t].astype(int)], s=15)
plt.plot(timerange[:60],task[21:81,0], '--k', linewidth=0.5)
plt.yticks([])

N_t             = np.copy(Nt_trial) #set lines per network plot
network_lines   = ['  ', 'Frontoparietal','DMN','Auditory', 'Sensorimotor', 'Visual']
l               = [0,1,2,3,4,5];
y0              = l[1]*np.ones([N_t])
y1              = l[2]*np.ones([N_t])
y2              = l[3]*np.ones([N_t])
y3              = l[4]*np.ones([N_t])
y4              = l[5]*np.ones([N_t])

plt.subplot(2,1,2)
for t_idx, t in enumerate(range(21,81)):
    if  np.isfinite(I_highest_rank[t]):
        if I_highest_rank[t] == 3: #DMN
            plt.scatter(timerange[t_idx],y1[t_idx], color= network_colors[I_highest_rank[t].astype(int)], s=15)
        elif I_highest_rank[t] == 0: # SM
            plt.scatter(timerange[t_idx],y3[t_idx], color= network_colors[I_highest_rank[t].astype(int)], s=15) 
        elif I_highest_rank[t] == 6: # vis 3
             plt.scatter(timerange[t_idx],y4[t_idx], color= network_colors[I_highest_rank[t].astype(int)], s=15) 
        elif I_highest_rank[t] == 9: # vis 2
             plt.scatter(timerange[t_idx],y4[t_idx], color= network_colors[I_highest_rank[t].astype(int)], s=15) 
        elif I_highest_rank[t] == 2: # vis 1
             plt.scatter(timerange[t_idx],y4[t_idx], color= network_colors[I_highest_rank[t].astype(int)], s=15) 
        elif I_highest_rank[t] == 1: #auditory
            plt.scatter(timerange[t_idx],y2[t_idx], color= network_colors[I_highest_rank[t].astype(int)], s=15)
        elif I_highest_rank[t] == 7: # frontoparietal
             plt.scatter(timerange[t_idx],y0[t_idx], color= network_colors[I_highest_rank[t].astype(int)], s=15) 
        elif I_highest_rank[t] == 8: #frontoparietal
            plt.scatter(timerange[t_idx],y0[t_idx], color= network_colors[I_highest_rank[t].astype(int)], s=15) 

ax = plt.gca()
ax.yaxis.set_ticks(l)
ax.set_yticklabels(network_lines, fontsize=11)
plt.ylim([0, 6])
plt.xlabel('Time ($s$)', fontsize=12)
plt.show()

### TRIAL AVERAGED RANKS 
names_S20_s10               = ['Sensorimotor','Auditory','Visual 1','DMN','ECN', 'Cerebellum', 'Visual 3', 'FP Attention (Left)' , 'FP Attention (Right)', 'Visual 2']
ses                         = '11_03'
smith10                     = [1,2,5,6,7,8,9,11,12,15] #SM, AU, V1, DMN, ECN, CER, V3, FP left, FP right, Vis 2

regressors_epochs_dict      = pickle.load(open('/project/3013060.04/TK_data/results_S20/regressors_epochs_dict.pickle',"rb"))
#TODO: remove local import
regs_epochs_ss              = regressors_epochs_dict[ses]
regs_average_ss             = np.mean(regs_epochs_ss, axis=1)
f_ta_10                     = np.zeros([10,60])
f_average_ss                = f_average_dict[ses]
for i in range(10):
    sel = smith10[i]
    f_ta_10[i,:] = f_average_ss[sel,:]
    del sel;
f_ta_10                     = pd.DataFrame(f_ta_10, index=names_S20_s10)
f_ta_ranks                  = f_ta_10.rank(axis=0, ascending=True); 
I_highest_rank              = np.zeros([f_average_ss.shape[1]])
for t in range(f_average_ss.shape[1]):
    I_highest_rank[t] = np.where(f_ta_ranks.iloc[:,t] == np.amax(f_ta_ranks.iloc[:,t]))[0]
task                        = regs_average_ss.T; print(task.shape)

network_lines               = ['  ', 'Frontoparietal','DMN','Auditory', 'Sensorimotor', 'Visual']

# Fig. 3. Panel B: Trial-Averaged Ranks
plt.figure(figsize=(5.8, 5.2))
plt.subplot(2,1,1)
plt.title('Trial-Averaged: Highest Ranking Network per Timepoint', font = 'Futura Hv BT', fontweight="bold", fontsize=12)
for t in range(N_t):
    if I_highest_rank[t] == 3: #DMN
        plt.scatter(timerange_m[t],task[t,0], color= network_colors[I_highest_rank[t].astype(int)], s=15)
    elif I_highest_rank[t] == 0: # SM
        plt.scatter(timerange_m[t],task[t,0], color= network_colors[I_highest_rank[t].astype(int)], s=15)
    elif I_highest_rank[t] == 6: # vis 3
        plt.scatter(timerange_m[t],task[t,0], color= network_colors[I_highest_rank[t].astype(int)], s=15)
    elif I_highest_rank[t] == 9: # vis 2
        plt.scatter(timerange_m[t],task[t,0], color= network_colors[I_highest_rank[t].astype(int)], s=15)
    elif I_highest_rank[t] == 2: # vis 1
        plt.scatter(timerange_m[t],task[t,0], color= network_colors[I_highest_rank[t].astype(int)], s=15)
    elif I_highest_rank[t] == 1: #auditory
        plt.scatter(timerange_m[t],task[t,0], color= network_colors[I_highest_rank[t].astype(int)], s=15)
    elif I_highest_rank[t] == 7: #frontoparietal
        plt.scatter(timerange_m[t],task[t,0], color= network_colors[I_highest_rank[t].astype(int)], s=15)
    elif I_highest_rank[t] == 8:
        plt.scatter(timerange_m[t],task[t,0], color= network_colors[I_highest_rank[t].astype(int)], s=15)

plt.plot(timerange_m,task[:,0], '--k', linewidth=0.5)
plt.yticks([])
l           = [0,1,2,3,4,5]; #lines per network plot
y0          = l[1]*np.ones([N_t])
y1          = l[2]*np.ones([N_t])
y2          = l[3]*np.ones([N_t])
y3          = l[4]*np.ones([N_t])
y4          = l[5]*np.ones([N_t])

plt.subplot(2,1,2)
for t in range(N_t):
    if I_highest_rank[t] == 3: #DMN
        plt.scatter(timerange[t],y1[t], color= network_colors[I_highest_rank[t].astype(int)], s=15)
    elif I_highest_rank[t] == 0: # SM
        plt.scatter(timerange_m[t],y3[t], color= network_colors[I_highest_rank[t].astype(int)], s=15) 
    elif I_highest_rank[t] == 6: # vis 3
        plt.scatter(timerange_m[t],y4[t], color= network_colors[I_highest_rank[t].astype(int)], s=15) 
    elif I_highest_rank[t] == 9: # vis 2
        plt.scatter(timerange_m[t],y4[t], color= network_colors[I_highest_rank[t].astype(int)], s=15) 
    elif I_highest_rank[t] == 2: # vis 1
        plt.scatter(timerange_m[t],y4[t], color= network_colors[I_highest_rank[t].astype(int)], s=15) 
    elif I_highest_rank[t] == 1: #auditory
        plt.scatter(timerange_m[t],y2[t], color= network_colors[I_highest_rank[t].astype(int)], s=15)
    elif I_highest_rank[t] == 7: # frontoparietal
        plt.scatter(timerange_m[t],y0[t], color= network_colors[I_highest_rank[t].astype(int)], s=15) 
    elif I_highest_rank[t] == 8: #frontoparietal
        plt.scatter(timerange_m[t],y0[t], color= network_colors[I_highest_rank[t].astype(int)], s=15) 

ax = plt.gca()
ax.yaxis.set_ticks(l)
ax.set_yticklabels(network_lines, fontsize=11)
plt.ylim([0, 6])
plt.xlabel('Time ($s$)', fontsize=12)
plt.show()

### GROUP LEVEL RANKS
# most common highest ranking network across sessions 
f_ta_10_all = {}; f_ta_ranks = {}; I_highest_rank_all = {}; I_highest_rank_all = {}; task_all = {}

for ses_idx, ses in enumerate(sessions):
    f_average_ss            = f_average_dict[ses]
    regs_epochs_ss          = regressors_epochs_dict[ses]
    regs_average_ss         = np.mean(regs_epochs_ss, axis=1)

    f_ta_10 = np.zeros([10,60])
    for i in range(10):
        sel = smith10[i]
        f_ta_10[i,:] = f_average_ss[sel,:]
        del sel;
    
    f_ta_pd                 = pd.DataFrame(f_ta_10, index=names_S20_s10)
    f_ta_ranks[ses]         = f_ta_pd.rank(axis=0, ascending=True)
    I_highest_rank          = np.zeros([f_average_ss.shape[1]])
    for t in range(f_average_ss.shape[1]):
        I_highest_rank[t]   = np.where(f_ta_ranks[ses].iloc[:,t] == np.amax(f_ta_ranks[ses].iloc[:,t]))[0]
    I_highest_rank_all[ses] = I_highest_rank
    del f_ta_pd; f_ta_10; I_highest_rank
    
allranks                    =    np.vstack(I_highest_rank_all.values())
allranks_pd                 = pd.DataFrame(allranks)

ranks_pert                  = {}; I_highest_rank = np.zeros([60])
for t in range(60):
    ar                      = allranks_pd.iloc[:,t]
    df                      = pd.DataFrame({'Number': ar})
    df1                     = pd.DataFrame(df['Number'].value_counts())
    ranks_pert[t]           = df1
    I_highest_rank[t]       = df1.index[0]

network_lines = ['  ', 'Frontoparietal','DMN','Auditory', 'Sensorimotor', 'Visual']

# Fig. 3. Panel B: Group-Level Ranks
plt.figure(figsize=(5.8, 5.2))
plt.subplot(2,1,1)
plt.title('Group-level: Highest Ranking Network per Timepoint', fontweight='bold', fontsize=12)
for t in range(N_t):
    if I_highest_rank[t] == 3: #DMN
        plt.scatter(timerange_m[t],task[t,0], color= network_colors[I_highest_rank[t].astype(int)], s=15)
    elif I_highest_rank[t] == 0: # SM
        plt.scatter(timerange_m[t],task[t,0], color= network_colors[I_highest_rank[t].astype(int)], s=15)
    elif I_highest_rank[t] == 6: # vis 3
        plt.scatter(timerange_m[t],task[t,0], color= network_colors[I_highest_rank[t].astype(int)], s=15)
    elif I_highest_rank[t] == 9: # vis 2
        plt.scatter(timerange_m[t],task[t,0], color= network_colors[I_highest_rank[t].astype(int)], s=15)
    elif I_highest_rank[t] == 2: # vis 1
        plt.scatter(timerange_m[t],task[t,0], color= network_colors[I_highest_rank[t].astype(int)], s=15)
    elif I_highest_rank[t] == 1: #auditory
        plt.scatter(timerange_m[t],task[t,0], color= network_colors[I_highest_rank[t].astype(int)], s=15)
    elif I_highest_rank[t] == 7: #frontoparietal
        plt.scatter(timerange_m[t],task[t,0], color= network_colors[I_highest_rank[t].astype(int)], s=15)
    elif I_highest_rank[t] == 8:
        plt.scatter(timerange_m[t],task[t,0], color= network_colors[I_highest_rank[t].astype(int)], s=15)
plt.plot(timerange_m,task[:,0], '--k', linewidth=0.5)
plt.yticks([])

l       = [0,1,2,3,4,5]; #Lines per network plot
y0      = l[1]*np.ones([N_t])
y1      = l[2]*np.ones([N_t])
y2      = l[3]*np.ones([N_t])
y3      = l[4]*np.ones([N_t])
y4      = l[5]*np.ones([N_t])

plt.subplot(2,1,2)
for t in range(N_t):
    if I_highest_rank[t] == 3: #DMN
        plt.scatter(timerange[t],y1[t], color= network_colors[I_highest_rank[t].astype(int)], s=15)
    elif I_highest_rank[t] == 0: # SM
        plt.scatter(timerange_m[t],y3[t], color= network_colors[I_highest_rank[t].astype(int)], s=15) 
    elif I_highest_rank[t] == 6: # vis 3
        plt.scatter(timerange_m[t],y4[t], color= network_colors[I_highest_rank[t].astype(int)], s=15) 
    elif I_highest_rank[t] == 9: # vis 2
        plt.scatter(timerange_m[t],y4[t], color= network_colors[I_highest_rank[t].astype(int)], s=15) 
    elif I_highest_rank[t] == 2: # vis 1
        plt.scatter(timerange_m[t],y4[t], color= network_colors[I_highest_rank[t].astype(int)], s=15) 
    elif I_highest_rank[t] == 1: #auditory
        plt.scatter(timerange_m[t],y2[t], color= network_colors[I_highest_rank[t].astype(int)], s=15)
    elif I_highest_rank[t] == 7: # frontoparietal
        plt.scatter(timerange_m[t],y0[t], color= network_colors[I_highest_rank[t].astype(int)], s=15) 
    elif I_highest_rank[t] == 8: #frontoparietal
        plt.scatter(timerange_m[t],y0[t], color= network_colors[I_highest_rank[t].astype(int)], s=15) 

ax = plt.gca()
ax.yaxis.set_ticks(l)
ax.set_yticklabels(network_lines, fontsize=11)
plt.ylim([0, 6])
plt.xlabel('Time ($s$)', fontsize=12)

#%% CORRELATIONS ENTRIES OF F TO THE TASK
# ---------------------------------------------------------
f_taskcor_dict              = {}; f_taskcor_pvals_dict = {}
f_taskcorF_dict             = {}; 

for ses_idx, ses in enumerate(sessions):   
    f_taskcor_dict[ses], f_taskcor_pvals_dict[ses] = trifle_stats.run_cors2d(task_dict[ses].T, f_tpm_dict[ses])

    # Fisher Z transform
    f_taskcorF              = np.zeros(f_taskcor_dict[ses].shape)
    for reg in range(f_taskcor_dict[ses].shape[0]):
        for net in range(f_taskcor_dict[ses].shape[1]):
            f_taskcorF[reg, net] = trifle_stats.r2fisherz(f_taskcor_dict[ses][reg,net])
    del reg, net
    f_taskcorF_dict[ses]    = pd.DataFrame(f_taskcorF, index=names_task, columns= names_S20)
    del ses_idx, ses
    
f_taskcor_pNet = np.zeros([2,20,40])
for ses_idx, ses in enumerate(sessions):  
    for task_idx in range(2):
        for net_idx in range(20):
            f_taskcor_pNet[task_idx, net_idx, ses_idx] = f_taskcorF_dict[ses].iloc[task_idx,net_idx]

f_taskcor_mean              = pd.DataFrame(np.mean(f_taskcor_pNet, axis=2), index=names_task, columns=names_S20)
f_taskcor_mean_sort         = f_taskcor_mean.sort_values('Visual',axis=1, ascending=False)

#%% FIGURE 4a: SINGLE SUBJECT TRIAL AVERAGED SMITH20
# --------------------------------------------------------
sub_ses = '11_03'
plt.figure(figsize=(7.5, 4.8))
plt.title('Single Session: Trial-Averaged Network Weights (SMITH20)', font = 'Futura Hv BT', fontweight="bold", fontsize=12)
plt.errorbar(timerange_m, f_average_dict[sub_ses][15,:], f_sem_dict[sub_ses][15,:],color=colors_4[0], linewidth=1, label='Primary Visual') 
plt.errorbar(timerange_m, f_average_dict[sub_ses][1,:],f_sem_dict[sub_ses][1,:],color=colors_4[1]  , linewidth=1, label='Sensorimotor')  
plt.errorbar(timerange_m, f_average_dict[sub_ses][2,:],f_sem_dict[sub_ses][2,:],color=colors_4[2], linewidth=1, label='Auditory') 
plt.errorbar(timerange_m, f_average_dict[sub_ses][6,:],f_sem_dict[sub_ses][6,:],color=colors_4[3], linewidth=1, label='DMN')  
plt.xlabel('Time ($s$)', fontsize=11)
plt.ylabel('Amplitude', fontsize=11)
plt.legend(fontsize=11, frameon=False)

#%% TRIAL AVERAGED FOR ALL PPS
# -------------------------------------------------------------------------------------
fig, ax = plt.subplots(14,3, figsize=(9.2,22),sharex='col', sharey='row')
pltcount= 1
fig.add_axes=([0.1, 0.1, 0.6, 0.75])
for ses_idx, ses in enumerate(sessions):
    ## PLOT AVERAGE Mt VS. TASK    
    ax          = plt.subplot(14,3,pltcount)
    color_idx   = -1
    
    for net in [15,1, 2,6]: 
        color_idx=color_idx+1
        ax.errorbar(timerange_m, f_average_dict[ses][net,:], f_sem_dict[ses][net,:],color=mycolors[color_idx], linewidth=0.8, label=names_S20[net])
    ax.tick_params(axis='both', which='minor', labelsize=6)
    plt.grid(False)
    fig.tight_layout(pad=3.0)

    if ses == '10_02':  #'10_02':
        pltcount = pltcount +2
    elif ses == '15_02':  #'15_02':
        pltcount = pltcount +2
    else:
        pltcount = pltcount+1
plt.tight_layout(pad=3.0) #pad=3.0
fig.supxlabel('Time ($s$)', fontsize=16)
fig.supylabel('Amplitude', fontsize=16)
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.3,0.95), fontsize=16, frameon=False)
plt.tight_layout()

#%% GROUP-LEVEL Z-VALUE SMITH20
# ---------------------------------------------------------
# CONCATENATE ACROSS SESSIONS
import re
participants = ['03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '15', '16', '17']
f_concat = {}
for pp in participants:
    if re.search('10', pp) or re.search('15', pp):
        keys = [str(pp)+'_01', str(pp)+'_02']
        f_concat[pp] = np.concatenate((f_epochs_dict[keys[0]], f_epochs_dict[keys[1]]), axis=1)
    else:
        keys = [str(pp)+'_01', str(pp)+'_02', str(pp)+'_03']
        f_concat[pp] = np.concatenate((f_epochs_dict[keys[0]], f_epochs_dict[keys[1]], f_epochs_dict[keys[2]]), axis=1)

# GROUP-LEVEL STATISTICS FOR TRIAL-AVERAGED F
networks_S10                = [1,2,5,6,7,8,9,10,12,15,16]
networks_stats              = [1,2,5,6,9,15]
allpp_tvals                 = {}
for net_idx, net in enumerate(networks_stats):
    pp_tvals = np.zeros([14,60])
    for pp_idx, pp in enumerate(participants):
        pp_tvals[pp_idx,:]  = ss.ttest_1samp(f_concat[pp][net,:,:], popmean=0)[0]
    allpp_tvals[names_S20[net]]= pp_tvals
    
grouplevel_zvals = {}
for net_idx, net in enumerate(networks_stats):
    net_zvals               = np.zeros([60])
    for t in range(60):
        net_zvals[t]        = np.mean(allpp_tvals[names_S20[net]][:,t])/np.std(allpp_tvals[names_S20[net]][:,t])
    grouplevel_zvals[names_S20[net]] = net_zvals
    del net_zvals

#%% Figure 5b: GROUP LEVEL STATISTICS PLOT
# ---------------------------------------------------------
plt.figure(figsize=(7.5, 4.8))
plt.title('Group-level Statistics: Network Involvement (SMITH20)', font = 'Futura Hv BT', fontweight="bold", fontsize=12)
p1 =plt.plot(timerange_m, grouplevel_zvals[names_S20[15]], colors_4[0], linewidth=1)
p2 =plt.plot(timerange_m, grouplevel_zvals[names_S20[1]], colors_4[1] , linewidth=1)
p3= plt.plot(timerange_m, grouplevel_zvals[names_S20[2]], colors_4[2] , linewidth=1)
p4= plt.plot(timerange_m, grouplevel_zvals[names_S20[6]], colors_4[3] , linewidth=1)
plt.xlabel('Time ($s$)', size=11)
plt.ylabel('$Z$-value', size=11)
plt.ylim([-2,6.5])
plt.legend(['Primary Visual','Sensorimotor', 'Auditory','DMN'], frameon=False, fontsize=11) 

#%% CREATE DATAFRAME FOR FLOBS GLMS
# ---------------------------------------------------------
## PP ID [| SES ID] | TRIAL | Timepoint | BF1 | BF2 | BF3 | BF4 | BF5 | BF6 | f_nw1 .... f_nw20 | 
pps             = ['PP3', 'PP4', 'PP5', 'PP6', 'PP7', 'PP8','PP9', 'PP10', 'PP11','PP12', 'PP13', 'PP15','PP16', 'PP17']
pps_pub         = ['PP1', 'PP2', 'PP3', 'PP4', 'PP5', 'PP6','PP7', 'PP8', 'PP9','PP10', 'PP11', 'PP12','PP13', 'PP14']
N_pp            = 14
pp_id           = []; tp_id = []; ses_id = [];
start           = 0; ntp_ses = 3000
for I_pp, pp in enumerate(pps):
    if pp == 'PP10' or pp == 'PP15':
        n1      = 6000
        pp_id.extend(n1*[pp])
        tp_id.extend(np.arange(n1))
        ses_id.extend(ntp_ses*['ses_1'])
        ses_id.extend(ntp_ses*['ses_2'])
    else:
        n2      = 9000
        pp_id.extend(n2*[pp])
        tp_id.extend(np.arange(n2))
        ses_id.extend(ntp_ses*['ses_1'])
        ses_id.extend(ntp_ses*['ses_2'])
        ses_id.extend(ntp_ses*['ses_3'])
pp_id = np.array(pp_id); tp_id = np.array(tp_id); ses_id = np.array(ses_id)

# Load basis functions as created with Feat 
names_subses                = ['03_01', '03_02', '03_03', '04_01', '04_02', '04_03', '05_01', '05_02', '05_03', '06_01', '06_02', '06_03', '07_01', '07_02', '07_03', '08_01', '08_02', '08_03', '09_01', '09_02', '09_03', '10_01', '10_02', '11_01', '11_02', '11_03', '12_01', '12_02', '12_03', '13_01', '13_02', '13_03', '15_01', '15_02', '16_01', '16_02', '16_03', '17_01', '17_02', '17_03']
sessions                    = list(names_subses)
names_bfs                   = ['Visual_HRF', 'Visual_Shift', 'Visual_Disp','Visual_4', 'Visual_5', 'Visual_6']      
subses_bfs                  = np.zeros([120000, 6]); subses_fs = np.zeros([120000, 20]);
start                       = 0

for ses_idx, ses in enumerate(sessions):
    subses_task      = op.join('/project/3013060.04/TK_data/flobs/IV_basisfunctions/final_6bfs/design_flobs_'+str(ses)+'.txt')      #op.join('/project/3013060.04/TK_data/flobs/IV_basisfunctions/visual_v2_6bfs/design_flobs_'+str(ses)+'.txt')
    bfs              = np.loadtxt(subses_task)
    subses_bfs[start:start+ntp_ses,0:6] = bfs  
    subses_fs[start:start+ntp_ses,:]  = f_tpm_dict[ses].T
    start = start+ntp_ses

# Create success regressor
succes_reg = []   
succes = {}
for sub_ses in sessions:    
    behavior_table       = np.loadtxt(op.join('/project/3013060.04/TK_data/behavioral/Behavioral_sub_'+str(sub_ses[:2])+'_ses_'+str(sub_ses[3:])+'.txt'), delimiter="\t", skiprows=1)
#TODO: remove local import
    succes[sub_ses]      = behavior_table[:,3]; 
    onsets_visual        = behavior_table[:,0]; onsets_visual_f = onsets_visual/TR
    frame_dif            = 60; n_del = sum(onsets_visual_f+60 > 3000)
    
    epoch_start          = np.round(onsets_visual_f[:-n_del],0).astype(int)
    epoch_ends           = epoch_start + frame_dif; epoch_ends = epoch_ends.astype(int)
    
    s                    = succes[sub_ses][:epoch_start.shape[0]]
    hit_start            = epoch_start[s==1]; hit_end = epoch_ends[s==1]; 
    fail_start           = epoch_start[s==0]; fail_end = epoch_ends[s==0];
    
    reg                  = np.zeros([3000])
    for h in range(hit_start.shape[0]):
        reg[hit_start[h]:hit_end[h]] = 1
    for f in range(fail_start.shape[0]):
        reg[fail_start[f]:fail_end[f]] = 2
    succes_reg.extend(reg); 
    del reg, behavior_table, onsets_visual, onsets_visual_f, n_del, epoch_start, epoch_ends, s, hit_start, fail_start
    
succes_reg = pd.DataFrame(succes_reg, columns=["Succes"])
print(succes_reg)

from patsy import dmatrices
df = pd.DataFrame({'PP':pp_id, 'Session':ses_id, 'Timepoint':tp_id, 'Succes':succes_reg["Succes"], 'Visual:HRF':subses_bfs[:,0],'Visual:Shift':subses_bfs[:,1], 'Visual:Dispersion':subses_bfs[:,2],'Visual:4':subses_bfs[:,3],'Visual:5':subses_bfs[:,4], 'Visual:6':subses_bfs[:,5],'f_NWD1':subses_fs[:,0], 'f_Sensorimotor':subses_fs[:,1],  'f_Auditory':subses_fs[:,2], 'f_NWD2':subses_fs[:,3], 'f_NWD3':subses_fs[:,4], 'f_MedialVisual_V1':subses_fs[:,5], 'f_DMN':subses_fs[:,6], 'f_ECN':subses_fs[:,7], 'f_Cerebellum':subses_fs[:,8], 'f_LateralVisual_V3':subses_fs[:,9], 'f_NWD_IFGAmyg':subses_fs[:,10], 'f_FPleft':subses_fs[:,11], 'f_FPright':subses_fs[:,12], 'f_NWD4':subses_fs[:,13], 'f_NWD_Insula':subses_fs[:,14], 'f_PrimaryVisual_V2':subses_fs[:,15], 'f_NWD_Hipp':subses_fs[:,16], 'f_NWD5':subses_fs[:,17], 'f_NWD6':subses_fs[:,18], 'f_NWD7':subses_fs[:,19]})
from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
df['VisualHRF_z'] = scaler.fit_transform(df[["Visual:HRF"]])
df['VisualShift_z'] = scaler.fit_transform(df[["Visual:Shift"]])
df['VisualDisp_z'] = scaler.fit_transform(df[["Visual:Dispersion"]])
df['Visual4_z'] = scaler.fit_transform(df[["Visual:4"]])
df['Visual5_z'] = scaler.fit_transform(df[["Visual:5"]])
df['Visual6_z'] = scaler.fit_transform(df[["Visual:6"]])

df_mem = df[["PP", "Session", "Timepoint", "f_Sensorimotor", "f_Auditory", "f_MedialVisual_V1", "f_DMN", "f_ECN", "f_Cerebellum", "f_LateralVisual_V3", 'f_FPleft', 'f_FPright', "f_PrimaryVisual_V2", 'f_NWD_Hipp', "VisualHRF_z", "VisualShift_z", "VisualDisp_z","Visual4_z", "Visual5_z", "Visual6_z"]]
df_lm = pd.concat((df_mem, pd.get_dummies(succes_reg, drop_first=True)), axis=1)

#%% Epoch bfs for plotting
bfs_visual_dict         = {}
for sub_ses in sessions:
    bfs_vis_orig        = np.loadtxt('/project/3013060.04/TK_data/flobs/IV_basisfunctions/final_6bfs/design_flobs_'+str(sub_ses)+'.txt')
    scaler              = preprocessing.StandardScaler()
    bfs_vis             = scaler.fit_transform(bfs_vis_orig)
    behavior_table      = np.loadtxt(op.join('/project/3013060.04/TK_data/behavioral/Behavioral_sub_'+str(sub_ses[:2])+'_ses_'+str(sub_ses[3:])+'.txt'), delimiter="\t", skiprows=1)

    onsets_visual       = behavior_table[:,0]; onsets_visual_f = onsets_visual/TR
    frame_dif           = 60; 
    valid_indices       = onsets_visual_f + 60 <= 3000
    onsets_visual_f     = onsets_visual_f[valid_indices]
    
    epoch_start         = np.round(onsets_visual_f,0).astype(int)
    epoch_ends          = epoch_start + frame_dif; epoch_ends = epoch_ends.astype(int)
    Ntrials             = np.sum(valid_indices)
    bfs_cut             = np.zeros([Ntrials,60,6])
    
    for ti in range(Ntrials):
        bfs_cut[ti,:,:] = bfs_vis[epoch_start[ti]:epoch_ends[ti], :]
    bfs_visual_dict[sub_ses] = bfs_cut
    del bfs_cut

bfs_visual_all = np.zeros([40,60,6]); 
for ses_idx, sub_ses in enumerate(sessions):
    print(sub_ses+' check')
    for bf_idx in range(6):
        bfs_visual_all[ses_idx,:,bf_idx] = np.nanmean(bfs_visual_dict[sub_ses][:,:,bf_idx], axis=0)

bfs_vis_gm = np.zeros([60,6]);
for bf_idx in range(6):
    bfs_vis_gm[:,bf_idx] = np.nanmean(bfs_visual_all[:,:,bf_idx], axis=0)

#%% PLOT PACF FOR AR LAG  SELECTION
# ---------------------------------------------------------
from statsmodels.graphics.tsaplots import plot_pacf

names_df = ['f_PrimaryVisual_V2', 'f_Sensorimotor', 'f_Auditory', 'f_DMN']
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Partial Autocorrelation Function (PACF) for Different Networks', fontsize=16)
axes = axes.flatten()
for ax, network in zip(axes, names_df):
    plot_pacf(df_lm[network], ax=ax, lags=20)
    ax.set_title(f'PACF for {network}')
    ax.set_ylabel('Partial Autocorrelation')
    ax.set_xlabel('Lags')

plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for the title
plt.show()

#%% RUN GLSAR MODELS
# ---------------------------------------------------------
results_visual          = {}; contrasts_visual = {}; results_params = {}; results_se = {}; results_pvals = {}
names_df                = ['f_PrimaryVisual_V2', 'f_Sensorimotor', 'f_Auditory', 'f_DMN']
labels                  = ['Primary Visual', 'Sensorimotor', 'Auditory', 'DMN']

for net_idx, net in enumerate(names_df):
    print(net)
    y, X                = dmatrices(str(net)+'~ VisualHRF_z + VisualShift_z + VisualDisp_z + Visual4_z + Visual5_z +Visual6_z ' , data=df_lm, return_type='dataframe')    
    design              = X.iloc[:,1:]
    model               = sm.GLSAR(y, design, ar_lag)
    results             = model.iterative_fit()
    results_visual[net] = results
    results_params[net] = results.params
    results_se[net]     = results.bse
    results_pvals[net]  = results.pvalues
    #contrasts_visual[net] = results.t_test(["C(Succes, Treatment)[T.2.0] - C(Succes, Treatment)[T.1.0], VisualShift_z:C(Succes, Treatment)[T.2.0]-VisualShift_z:C(Succes, Treatment)[T.1.0],VisualDisp_z:C(Succes, Treatment)[T.2.0]-VisualDisp_z:C(Succes, Treatment)[T.1.0]"])
    print(results.summary())

# Fig. 4. Panel C model-based reconstruction of epoch response 
plt.figure(figsize=(7.5, 4.8))
plt.title('Group-level: Fitted Model of Network Involvement (SMITH20)', font='Futura Hv BT', fontweight="bold", fontsize=12)
for i, var in enumerate(names_df):
    fitted_values = np.dot(bfs_vis_gm, results_params[var])
    se = results_se[var]
    plt.plot(timerange_m, fitted_values, color=colors_4[i], linewidth=1, label=labels[i])
    plt.fill_between(
        timerange_m,
        np.dot(bfs_vis_gm, results_params[var] - se),
        np.dot(bfs_vis_gm, results_params[var] + se),
        color=colors_4[i],
        alpha=0.25
    )
plt.xlabel('Time ($s$)', size=11)
plt.ylabel('Amplitude', size=11)
plt.legend(loc='upper right')  
plt.show()

# MC correction 
from statsmodels.stats.multitest import fdrcorrection

allpvals                    = []
for reg in results_pvals.values():
    allpvals.extend(reg)

mc_pvals                    = fdrcorrection(allpvals)
regs                        = ['bf1','bf2','bf3','bf4','bf5','bf6']
names_mcpvals               = []
for net in names_df:
    for i in regs:
        names_mcpvals.extend([str(net)+'_'+str(i)])

mc_pvals_df = pd.concat((pd.DataFrame(names_mcpvals), pd.DataFrame(mc_pvals[0]), pd.DataFrame(mc_pvals[1])), axis=1)

#%% SUCCESFUL VS UNSUCCESSFUL TRIALS 
# ---------------------------------------------------------
succes_dict = {}
for sub_ses in sessions:
    behavior                = np.loadtxt(op.join('/project/3013060.04/TK_data/behavioral/Behavioral_sub_'+str(sub_ses[:2])+'_ses_'+str(sub_ses[3:])+'.txt'), delimiter="\t", skiprows=1)
#TODO: remove local import
    succes_dict[sub_ses]    = behavior[:,3]

succes_cut = {}
for sub_ses in sessions:
#TODO: remove local import
    onsets                  = behavior[:,1];    
    succes                  = behavior[:,3]
    Ntrials                 = onsets.shape[0]; 

    mask_nans               = np.isnan(onsets); sumnans = np.sum(mask_nans); 
    succes[mask_nans]       = np.nan; 
    succes_pd               = pd.DataFrame(succes)
    succes_pd               = succes_pd.dropna(axis=0); 
    if Ndel_t[sub_ses] > 0:
        succes_cut[sub_ses] = succes_pd[:-Ndel_t[sub_ses]]
    else:
        succes_cut[sub_ses] =  succes_pd

succes_concat               = {}
for pp in participants:
    if re.search('10', pp) or re.search('15', pp):
        keys = [str(pp)+'_01', str(pp)+'_02']
        succes_concat[pp]   = np.concatenate((succes_cut[keys[0]], succes_cut[keys[1]]))
    else:
        keys = [str(pp)+'_01', str(pp)+'_02', str(pp)+'_03']
        succes_concat[pp]   = np.concatenate((succes_cut[keys[0]], succes_cut[keys[1]], succes_cut[keys[2]]))

fepoch_hit                  = {}; fepoch_fail = {}
favg_trials_hit             = {}; favg_trials_fail = {}

tvals_hitfail               = np.zeros([N_pp,d,Nt_trials]); pvals_hitfail = np.zeros([N_pp,d,Nt_trials])
for pp_idx, pp in enumerate(participants):
    I_hit                   = np.squeeze(succes_concat[pp]) == 1
    I_fail                  = np.squeeze(succes_concat[pp]) == 0 
    fe_hit                  = f_concat[pp][:,np.squeeze(I_hit),:]
    fe_fail                 = f_concat[pp][:,np.squeeze(I_fail),:]

    for net_idx, net in enumerate(names_S20):
        tvals_hitfail[pp_idx, net_idx,:], pvals_hitfail[pp_idx, net_idx,:] = ss.ttest_ind(fe_fail[net_idx,:], fe_hit[net_idx,:])

    fepoch_hit[pp]          = fe_hit
    fepoch_fail[pp]         = fe_fail    
    favg_trials_hit[pp]     = np.mean(fe_hit, axis=1)
    favg_trials_fail[pp]    = np.mean(fe_fail, axis=1)
    del I_hit, I_fail, fe_hit, fe_fail

#%% Create dataframes for group-level
hits_allpp                  = np.zeros([14,20, 60])
fails_allpp                 = np.zeros([14,20,60])

for pp_idx, pp in enumerate(participants):
    hits_allpp[pp_idx, :,:] = favg_trials_hit[pp]
    fails_allpp[pp_idx, :,:]= favg_trials_fail[pp]   

grouplevel_zvals            = np.zeros([20,60])
for net_idx, net in enumerate(names_S20):
    for t in range(60):
        grouplevel_zvals[net_idx, t] = np.mean(tvals_hitfail[:,net_idx, t])/np.std(tvals_hitfail[:,net_idx, t])
        del t 

M_groupz                    = np.mean(abs(grouplevel_zvals), axis=1)
Max_groupz                  = np.max(abs(grouplevel_zvals), axis=1)
M_groupz_pd                 = pd.DataFrame(M_groupz, index=names_S20)
M_groupz_sort               = M_groupz_pd.sort_values(0, ascending=False)
Max_groupz_pd               = pd.DataFrame(Max_groupz, index=names_S20)
Max_groupz_sort             = Max_groupz_pd.sort_values(0, ascending=False)
I_maxDMN                    = np.where(abs(grouplevel_zvals[6,:]) == np.max(abs(grouplevel_zvals[6,:])))[0][0]; tmp_DMN = timerange_m[I_maxDMN]
I_maxAud                    = np.where(abs(grouplevel_zvals[2,:]) == np.max(abs(grouplevel_zvals[2,:])))[0][0]; tmp_Aud = timerange_m[I_maxAud]

# Fig. 6. Panel A Group level statistics for trial-average f SMITH20
plt.figure(figsize=(7.5, 4.8))
plt.title('Group-level Statistics: Fail > Hit (SMITH20)', font = 'Futura Hv BT', fontweight="bold", fontsize=12)
plt.plot(timerange_m, grouplevel_zvals[6,:], color= colors_4[3], label= "DMN", linewidth=1)
plt.fill_between(timerange_m, -2.0, 2.0, where = (timerange_m == tmp_DMN), color=colors_4[3], alpha=0.5)
plt.ylim([-2.0,2.0])
plt.xlabel('Time ($s$)', size=11)
plt.ylabel('$Z$-value', size=11)
plt.legend(frameon=False, fontsize=11, loc='upper right') 

#%% RUN GLSAR MODELS HIT VS FAIL
networks_glsar              = ['f_Sensorimotor', 'f_Auditory', 'f_MedialVisual_V1', 'f_DMN', 'f_ECN','f_LateralVisual_V3', 'f_FPleft', 'f_FPright','f_PrimaryVisual_V2']
results_visual_succes       = {}; contrasts_visual_succes = {}; results_params_succes = {}; results_se_succes = {}
results_tvals_succes        = {}; results_pvals_succes = {}
names_df                    = df_lm.columns[3:14]

for net_idx, net in enumerate(networks_glsar):
    print(net)
    y, X                    = dmatrices(str(net)+'~ VisualHRF_z + VisualShift_z + VisualDisp_z + Visual4_z + Visual5_z + Visual6_z + C(Succes, Treatment) + VisualHRF_z*C(Succes, Treatment) + VisualShift_z*C(Succes, Treatment) + VisualDisp_z*C(Succes, Treatment)+Visual4_z*C(Succes, Treatment)+ Visual5_z*C(Succes, Treatment)+Visual6_z*C(Succes, Treatment)' , data=df_lm, return_type='dataframe')    
    design                  = X.iloc[:,1:]
    model                   = sm.GLSAR(y, design, ar_lag)
    results                 = model.iterative_fit()
    results_visual_succes[net] = results
    results_params_succes[net] = results.params
    results_se_succes[net]  = results.bse
    results_tvals_succes[net] = results.tvalues
    results_pvals_succes[net] = results.pvalues
    contrasts_visual_succes[net] = results.t_test(["C(Succes, Treatment)[T.2.0] - C(Succes, Treatment)[T.1.0],VisualHRF_z:C(Succes, Treatment)[T.2.0] - VisualHRF_z:C(Succes, Treatment)[T.1.0], VisualShift_z:C(Succes, Treatment)[T.2.0]-VisualShift_z:C(Succes, Treatment)[T.1.0],VisualDisp_z:C(Succes, Treatment)[T.2.0]-VisualDisp_z:C(Succes, Treatment)[T.1.0], Visual4_z:C(Succes, Treatment)[T.2.0] - Visual4_z:C(Succes, Treatment)[T.1.0], Visual5_z:C(Succes, Treatment)[T.2.0] - Visual5_z:C(Succes, Treatment)[T.1.0], Visual6_z:C(Succes, Treatment)[T.2.0] - Visual6_z:C(Succes, Treatment)[T.1.0] "])
    print(results.summary())

# MC correction 
from statsmodels.stats.multitest import fdrcorrection
allpvals_succes             = []
for reg in results_pvals_succes.values():
    allpvals_succes.extend(reg)
mc_pvals_succes = fdrcorrection(allpvals_succes)

regs                        = ['Hit', 'Fail', 'bf1', 'bf1_hit', 'bf1_fail', 'bf2', 'bf2_hit', 'bf2_fail', 'bf3', 'bf3_hit', 'bf3_fail', 'bf4', 'bf4_hit', 'bf4_fail', 'bf5', 'bf5_hit', 'bf5_fail', 'bf6', 'bf6_hit', 'bf6_fail']
names_mcpvals               = []
for net in networks_glsar:
    for i in regs:
        names_mcpvals.extend([str(net)+'_'+str(i)])
mc_pvals_succes_df = pd.concat((pd.DataFrame(names_mcpvals), pd.DataFrame(mc_pvals_succes[0]), pd.DataFrame(mc_pvals_succes[1])), axis=1)

#%% GLSAR MODEL DMN ONLY
net                         = 'f_DMN'
y, X                        = dmatrices(f'{net} ~ VisualHRF_z + VisualShift_z + VisualDisp_z + Visual4_z + Visual5_z + Visual6_z + C(Succes, Treatment) + VisualHRF_z*C(Succes, Treatment) + VisualShift_z*C(Succes, Treatment) + VisualDisp_z*C(Succes, Treatment) + Visual4_z*C(Succes, Treatment) + Visual5_z*C(Succes, Treatment) + Visual6_z*C(Succes, Treatment)', data=df_lm, return_type='dataframe')
design                      = X.iloc[:, 1:] # Remove the intercept from the design matrix
model                       = sm.GLSAR(y, design, ar_lag)
results                     = model.iterative_fit()
print(results.summary())

contrast_test = results.t_test(["C(Succes, Treatment)[T.2.0] - C(Succes, Treatment)[T.1.0],"
                                "VisualHRF_z:C(Succes, Treatment)[T.2.0] - VisualHRF_z:C(Succes, Treatment)[T.1.0],"
                                "VisualShift_z:C(Succes, Treatment)[T.2.0] - VisualShift_z:C(Succes, Treatment)[T.1.0],"
                                "VisualDisp_z:C(Succes, Treatment)[T.2.0] - VisualDisp_z:C(Succes, Treatment)[T.1.0],"
                                "Visual4_z:C(Succes, Treatment)[T.2.0] - Visual4_z:C(Succes, Treatment)[T.1.0],"
                                "Visual5_z:C(Succes, Treatment)[T.2.0] - Visual5_z:C(Succes, Treatment)[T.1.0],"
                                "Visual6_z:C(Succes, Treatment)[T.2.0] - Visual6_z:C(Succes, Treatment)[T.1.0]"])
delta_b                     = contrast_test.effect   # Change in beta
t_values                    = contrast_test.tvalue  # t-values for the contrasts
p_values                    = contrast_test.pvalue  # p-values for the contrasts
print("\nContrast Results:") # Display the results
print(f"Change in beta (?b): {delta_b}")
print(f"t-values: {t_values}")
formatted_p_values          = [f"{pval:.6f}" for pval in p_values]
print(f"p-values: {formatted_p_values}")

# POST HOC CONTRASTS
contrast_labels             = [
    "C(Succes, Treatment)[T.2.0] - C(Succes, Treatment)[T.1.0]",
    "VisualHRF_z:C(Succes, Treatment)[T.2.0] - VisualHRF_z:C(Succes, Treatment)[T.1.0]",
    "VisualShift_z:C(Succes, Treatment)[T.2.0] - VisualShift_z:C(Succes, Treatment)[T.1.0]",
    "VisualDisp_z:C(Succes, Treatment)[T.2.0] - VisualDisp_z:C(Succes, Treatment)[T.1.0]",
    "Visual4_z:C(Succes, Treatment)[T.2.0] - Visual4_z:C(Succes, Treatment)[T.1.0]",
    "Visual5_z:C(Succes, Treatment)[T.2.0] - Visual5_z:C(Succes, Treatment)[T.1.0]",
    "Visual6_z:C(Succes, Treatment)[T.2.0] - Visual6_z:C(Succes, Treatment)[T.1.0]"
]
fdr_results_contrasts       = multipletests(p_values, alpha=0.05, method='fdr_bh')
is_significant_contrasts    = fdr_results_contrasts[0] 
corrected_pvals_contrasts   = fdr_results_contrasts[1]  
significant_contrast_indices= [i for i, significant in enumerate(is_significant_contrasts) if significant]
significant_contrast_labels = [contrast_labels[i] for i in significant_contrast_indices]
significant_corrected_pvals = corrected_pvals_contrasts[is_significant_contrasts]
print("Significant contrasts after FDR correction:")
for label, pval in zip(significant_contrast_labels, significant_corrected_pvals):
    print(f"Contrast: {label}, Corrected p-value: {pval:.6f}")

#%% Hit
dmn_hit = np.array([results_params_succes[net][2]+results_params_succes[net][3],
results_params_succes[net][5]+results_params_succes[net][6],
results_params_succes[net][8]+results_params_succes[net][9],
results_params_succes[net][11]+results_params_succes[net][12],
results_params_succes[net][14]+results_params_succes[net][15],
results_params_succes[net][17]+results_params_succes[net][18]])

#%% Fail
dmn_fail = np.array([results_params_succes[net][2]+results_params_succes[net][4],
results_params_succes[net][5]+results_params_succes[net][7],
results_params_succes[net][8]+results_params_succes[net][10],
results_params_succes[net][11]+results_params_succes[net][12],
results_params_succes[net][14]+results_params_succes[net][16],
results_params_succes[net][17]+results_params_succes[net][19]])

#%% Se Hit
dmn_hit_se = np.array([results_se_succes[net][2]+results_se_succes[net][3],
results_se_succes[net][5]+results_se_succes[net][6],
results_se_succes[net][8]+results_se_succes[net][9],
results_se_succes[net][11]+results_se_succes[net][12],
results_se_succes[net][14]+results_se_succes[net][15],
results_se_succes[net][17]+results_se_succes[net][18]])

#%% Se Fail
dmn_fail_se = np.array([results_se_succes[net][2]+results_se_succes[net][4],
results_se_succes[net][5]+results_se_succes[net][7],
results_se_succes[net][8]+results_se_succes[net][10],
results_se_succes[net][11]+results_se_succes[net][12],
results_se_succes[net][14]+results_se_succes[net][16],
results_se_succes[net][17]+results_se_succes[net][19]])

#%% Fig. 6. Panel B GROUP-LEVEL FITTED NETWORK INVOLVEMENT HIT/FAIL
plt.figure(figsize=(7.5, 4.8))
plt.title('Group-level: Fitted Model of Network Involvement: Hit vs. Fail (SMITH20)', font = 'Futura Hv BT', fontweight="bold", fontsize=12)
plt.plot(timerange_m, np.dot(bfs_vis_gm,dmn_hit), color='green', linewidth=1, label='Hits')
plt.fill_between(timerange_m,np.dot(bfs_vis_gm, dmn_hit-dmn_hit_se), y2= np.dot(bfs_vis_gm, dmn_hit + dmn_hit_se), color='green', alpha=0.25)
plt.plot(timerange_m, np.dot(bfs_vis_gm, dmn_fail), color='red', linewidth=1, label='Fails')
plt.fill_between(timerange_m,np.dot(bfs_vis_gm, dmn_fail-dmn_fail_se), y2= np.dot(bfs_vis_gm, dmn_fail + dmn_fail_se), color='red', alpha=0.25)
plt.xlabel('Time ($s$)', size=11)
plt.ylabel('Amplitude', size=11)
plt.ylim([-0.9, 0.9])
plt.legend()
#plt.savefig('/home/mrstats/tamdklo/figures/P1_trifle/S20/Rec_Grouplevel_SMITH20_hitfail.png', dpi=700)

#%% SUPPLEMENTARY MATERIAL
# ---------------------------------------------------------
#%% SECTION "CONFOUNDS"
confounds_path              = '/project/3013060.04/TK_data/results_S20/confounds_dict.pickle'
#TODO: remove local import
maxcors_TFM_visual          = pickle.load(open('/project/3013060.04/TK_data/results_S20/TFM_maxcors_visual.pickle',"rb"))
#TODO: remove local import
names_confounds             = [ 'CSF', 'WM', 'GS','X', 'Y', 'Z','RotX', 'RotY','RotZ' ]
confounds_dict              = pickle.load(open(confounds_path, "rb")); 
cors_confounds              = {}; pvals_confounds = {}
cors_confounds_z            = np.zeros([9,40])
confound_cor                = np.zeros([9,40])
for ses_idx, ses in enumerate(sessions):
    confounds               = confounds_dict[ses].T; confounds = confounds.to_numpy()
    Bz                      = Bz_dict[ses]
    cors_confounds[ses], pvals_confounds[ses] = trifle_stats.run_cors2d(confounds, Bz);
    num                     = int(maxcors_TFM_visual.loc[ses][0][3:])-1;
    confound_cor[:,ses_idx] = cors_confounds[ses][:,num]
    
    for conf in range(9):
        cors_confounds_z[conf,ses_idx] = trifle_stats.r2fisherz(confound_cor[conf,ses_idx])
        cors_confounds_z[conf,ses_idx] = abs(cors_confounds_z[conf,ses_idx])
    del confounds, Bz, num

confound_cor_pd             = pd.DataFrame(cors_confounds_z, index=names_confounds, columns= sessions).T; 
tfm_confounds_vis_plot      = confound_cor_pd.melt(var_name= 'Confound', value_name= 'Fisher-$Z$ value')
plt.figure(figsize=(9.2, 4.8))
sns.violinplot(x= 'Confound', y = 'Fisher-$Z$ value', data= tfm_confounds_vis_plot, color="0.8", inner='quartile') 
sns.stripplot(x= 'Confound', y = 'Fisher-$Z$ value', data= tfm_confounds_vis_plot ,jitter=True, palette=['steelblue', 'lightblue','cornflowerblue', "green",'red','saddlebrown', 'mediumseagreen', 'indianred', 'chocolate']) 
plt.title("Associations TFM Time Series with Confound Regressors", font = 'Futura Hv BT', fontweight="bold", fontsize=12)
plt.xlabel("Confound")
plt.ylabel("Fisher-$Z$ Tranformed Pearson correlation ($r$)")
plt.tight_layout()

#%% SECTION "IDENTIFIED TFMs"
##  Figure 3: Static M 
names_S20_new               = ['NWD (1)', 'Sensorimotor*','Auditory*','NWD (2)','NWD (3)','Medial visual*','DMN*','ECN*', 'Cerebellum*','Lateral visual*','NWD (4)', 'FP (Left)*' , 'FP (Right)*', 'NWD (5)', 'NWD (6)', 'Primary visual*','NWD (7)','NWD (8)', 'NWD (9)', 'NWD (10)']
ses                         = '11_03'
M_3_pd                      = pd.DataFrame(M_dict[ses][:,2], columns= ['Task-Positive Mode'], index=names_S20_new) 
M_7_pd                      = pd.DataFrame(M_dict[ses][:,6], columns= ['Default Temporal Mode'], index=names_S20_new) 
index_nets                  = ['Primary visual*','Lateral visual*','Medial visual*','Sensorimotor*','Cerebellum*','Auditory*', 'ECN*','FP (Left)*' , 'FP (Right)*','DMN*','NWD (1)','NWD (2)','NWD (3)','NWD (4)', 'NWD (5)','NWD (6)','NWD (7)', 'NWD (8)', 'NWD (9)', 'NWD (10)']
M_3_pd_re                   = M_3_pd.reindex(index_nets); M_psm = M_3_pd_re.iloc[:10,:]
M_7_pd_re                   = M_7_pd.reindex(index_nets); M_dtm = M_7_pd_re.iloc[:10,:]

## TPM
plt.figure(figsize=(5.2, 4.2))
sns.heatmap(M_psm, cmap = cmap_m , vmin=-2, vmax=2) 
plt.title("Mixing Matrix (M) Column", font = 'Futura Hv BT', fontweight="bold", fontsize=12)
plt.ylabel("Smith 20 Network")
plt.tight_layout()
plt.show()

## DTM
plt.figure(figsize=(5.2, 4.2))
sns.heatmap(M_dtm, cmap = cmap_m , vmin=-2, vmax=2)
plt.title("Mixing Matrix (M) Column", font = 'Futura Hv BT', fontweight="bold", fontsize=12)
plt.ylabel("Smith 20 Network")
plt.tight_layout()
plt.show()

## Smith 10 DTM
M_10_name = '/project/3013060.04/TK_data/dr/TFM_sub-11_03_dr_s10_c10.ica/melodic_unmix'
M_10 = np.loadtxt(M_10_name) 
M_10_pd  = pd.DataFrame(M_10[:,0], columns= ['Default Temporal Mode'], index=names_S10) 

plt.figure(figsize=(5.2, 4.2))
sns.heatmap(M_10_pd, cmap = cmap_m , vmin=-1.5, vmax=1.5) 
plt.title("Mixing Matrix (M) Column", font = 'Futura Hv BT', fontweight="bold", fontsize=12)
plt.ylabel("Smith 10 Network")
plt.tight_layout()
plt.show()

#%% SECTION "SPATIAL CORRESPONDENCE"
sessions_pub                = names_subses.copy()
nets_idx                    = [1,2,5,6,7,8,9,11,12,15] 
names_S10                   = ['Sensorimotor', 'Auditory','Primary Visual','DMN', 'Executive Control','Cerebellum', 'Lateral Visual', 'Frontoparietal Left', 'Frontoparietal Right', 'Medial Visual']
M_all = np.zeros([10,40])
for ses_idx, ses in enumerate(sessions):
    M_all[:,ses_idx] = M_rev[ses][nets_idx]

M_all_pd = pd.DataFrame(M_all, columns=sessions_pub, index= names_S10)
M_all_cor = M_all_pd.corr()
mask = np.triu(np.ones_like(M_all_cor, dtype=bool))

plt.figure(figsize=(7.5, 4.8))
sns.heatmap(M_all_cor, mask = mask, cmap=cmap, vmin=-1, vmax=1)
plt.title("Spatial Correlations Task-Relevant TFMs (SMITH20)",  font = 'Futura Hv BT', fontweight="bold", fontsize=12)
plt.xlabel("Task Run")
plt.ylabel("Task Run")
plt.tight_layout()
