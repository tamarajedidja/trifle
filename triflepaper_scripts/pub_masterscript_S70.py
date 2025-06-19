#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MASTERSCRIPT FOR THE SMITH 70 PARCELLATION FOR THE FOLLOWING MANUSCRIPT: 
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
from pathlib import Path

# Define path relative to current script location
trifle_module_path = os.path.join(os.path.dirname(__file__), "trifle_module")
sys.path.append(os.path.abspath(trifle_module_path))
import trifle_main
import trifle_stats

#%% STUDY SPECIFIC
# --------------------------------------------------------------

##  PARAMETERS
TR                  = .206;
Nt                  = 3000;         # total number of timepoints per task run 
Nt_trial            = 60;           # amount of time points per epoch 
timerange           = np.arange(0, TR*Nt, TR)
framerange          = np.arange(0, Nt, 1)
timerange_m         = timerange[:Nt_trial]
ar_lag              = 5

##  NAMES 
names_task          = ['Visual', 'Motor']
names_S70           = ['Intracalcerine (Medial Visual)', 'Lingual (Medial Visual)', 'NWD 1', 'Cerebellum','P. Cingulate', 
                   'Precuneus 1', 'Paracingulate 1', 'Thalamus & Putamen', 'Brainstem 1', 'Postcentral gyrus',
                   'L. Occipital (Lateral Visual)', 'NWD 2', 'Auditory', 'Caudate', 'Post. Cingulate & Precuneus', 
                   'A. Cingulate', 'M. Temporal', 'Occipital pole 1 (V2)', 'Paracingulate & FP', 'Insula',
                   'Sup. Lat. Occipital (V3)',  'Paracingulate 2',  'NWD 3', 'Ant. SMG', 'R. Lat. Occipital (V3)', 
                   'Brainstem 2','NWD 4', 'Mid. Temporal Gyrus', 'Precentral gyrus 1', 'IFG', 
                   'DMN (Precuneus)', 'Brainstem 3', 'NWD 5', 'NWD 6', 'NWD 7', 
                   'Sup. FG', 'Sup. Parietal 1', 'NWD 8', 'Angular', 'Hippocampus', 
                   'NWD 9', 'Occipital Pole 2 (V2)', 'Frontal pole 1', 'Precentral gyrus 2','Precuneus 2',
                   'L. Sensorimotor', 'NWD 10', 'NWD 11', 'Temporal pole', 'NWD 12',
                   'R. Sensorimotor', 'NWD 13', 'NWD 14','Primary visual 1 (V2)', 'NWD 15',
                   'R. Sup. Lat. Occipital', 'L. Sup. Lat. Occipital', 'Frontal pole 2','Sup. Parietal 2','Frontal medial & subcallosal', 
                   'NWD 16', 'Sup. Parietal & SMG', 'Precentral gyrus 3', 'Primary visual 2 (V2)', 'NWD 17', 
                   'NWD 18', 'R. Inf. Temporal gyrus' , 'R. Inf. Temporal gyrus', 'L. Inf. Temporal gyrus', 'R. Middle Temporal gyrus']
names_tfms          = ['TFM1', 'TFM2', 'TFM3', 'TFM4', 'TFM5', 'TFM6', 'TFM7', 'TFM8', 'TFM9', 'TFM10', 'TFM11', 'TFM12', 'TFM13', 'TFM14', 'TFM15', 'TFM16', 'TFM17', 'TFM18', 'TFM19', 'TFM20', 'TFM21']
names_subses        = ['03_01', '03_02', '03_03', '04_01', '04_02', '04_03', '05_01', '05_02', '05_03','06_01', '06_02', '06_03', '07_01', '07_02', '07_03', '08_01', '08_02', '08_03','09_01', '09_02', '09_03', '10_01', '10_02', '11_01', '11_02', '11_03','12_01', '12_02', '12_03', '13_01', '13_02', '13_03', '15_01', '15_02','16_01', '16_02', '16_03', '17_01', '17_02', '17_03']
sessions            = list(names_subses)
tfms                = list(names_tfms)

##  PLOTTING SPECS
cmap                = sns.diverging_palette(220, 10, as_cmap=True)
from matplotlib.colors import LinearSegmentedColormap
cmap_m              = LinearSegmentedColormap.from_list(name='test', colors=['blue','white','orange'])

import matplotlib as mpl
mpl.rc('font', family='Futura Md BT') 
mpl.rcParams.update({'font.size': 11})

mycolors            = ['#A6381A',"#2A9D8F",'#EFC560', '#EFC560','#3C6A89'] 
colors_4            = ['#A6381A',"#2A9D8F",'#EFC560','#3C6A89'] 

## DIMS
Nd = 70; Nk=21; Nses = 40; Npp = 14

# ROOT FOLDER PARSER
#TODO

# PATHS
#experiment_dir      = Path(args.data_root).resolve()
experiment_dir      = Path('/project/3013060.04/TK_data/')
fmridata_dir        = experiment_dir / 'derivative-menon' / 'melodic'
stage1_dir          = experiment_dir / 'dr'
designs_dir         = experiment_dir / 'glm' / 'designs'

# Filenames
Xfilename           = 'filtered_func_data_denoised_norm2_unitvariance.nii.gz'
Sfilename           = 'dr_stage2_subject00000.nii.gz'
maskfile_name       = 'mask.nii.gz'

filenames_X = {}; filenames_S = {}; filenames_mask = {}
filenames_T = {}; filenames_M = {}; filenames_B = {}
filenames_task = {} 

for sub_ses in sessions:
    subses_X         = Path(f"sub-{sub_ses}.ica")
    subses_stage1    = Path(f"DR_sub-{sub_ses}_s70.dr")
    stage2_dir       = stage1_dir / f"TFM_sub-{sub_ses}_dr_s70_c21.ica"
    subses_task      = Path(f"{sub_ses}.txt")
    
    filenames_X[sub_ses]     = fmridata_dir / subses_X / Xfilename
    filenames_S[sub_ses]     = stage1_dir / subses_stage1 / Sfilename
    filenames_mask[sub_ses]  = stage1_dir / subses_stage1 / maskfile_name
    filenames_M[sub_ses]     = stage2_dir / "melodic_unmix"
    filenames_B[sub_ses]     = stage2_dir / "melodic_mix"
    filenames_T[sub_ses]     = stage1_dir / subses_stage1 / "dr_stage1_subject00000.txt"
    filenames_task[sub_ses]  = designs_dir / subses_task

## IMPORT TASK DESIGNS
# ---------------------------------------------------------
X_orig = {}; X_orig_shape = {}
task_dict = {}; task_pd_dict = {}
for ses in sessions:
    task_dict[ses], task_pd_dict[ses] = load_taskdesign(filenames_task[ses], Nt, 'no')

#%% RUN MAIN (DATALOAD AND TRIFLE LAYER 3)
## CREATE DATAFRAMES 
# ---------------------------------------------------------
X_dict  = {};
S_dict  = {};
T_dict  = {}; Tz_dict  = {};
M_dict  = {};
B_dict  = {}; Bz_dict   = {};
Xr_dict = {};
Mt_dict = {}
f_dict  = {}

for ses in sessions:
    X_dict[ses], S_dict[ses], T_dict[ses], Tz_dict[ses], M_dict[ses], B_dict[ses], Bz_dict[ses], Xr_dict[ses], Mt_dict[ses], f_dict[ses] = trifle_main.main(filenames_X[ses], filenames_S[ses], filenames_mask[ses], filenames_T[ses], filenames_M[ses], filenames_B[ses])
    
#%%#%% IMPORT PREVIOUSLY CREATED DATAFRAMES
# ---------------------------------------------------------
#TODO: remove import
#X_dict       = pickle.load(open('/project/3013060.04/TK_data/results_S70/X_dict.pickle',"rb")); 
S_dict       = pickle.load(open('/project/3013060.04/TK_data/results_S70/S_dict_S70.pickle',"rb")); 
Tz_dict      = pickle.load(open('/project/3013060.04/TK_data/results_S70/DRz_dict_S70.pickle',"rb")); 
M_dict       = pickle.load(open('/project/3013060.04/TK_data/results_S70/M_dict_S70.pickle',"rb")); 
Bz_dict      = pickle.load(open('/project/3013060.04/TK_data/results_S70/Bz_dict_S70.pickle',"rb")); 
#Xr_dict      = pickle.load(open('/project/3013060.04/TK_data/results_S70/Xr_dict.pickle',"rb")); 
#Mt_dict      = pickle.load(open('/project/3013060.04/TK_data/results_S70/Mt_dict_S70.pickle',"rb")); 
f_dict       = pickle.load(open('/project/3013060.04/TK_data/results_S70/f_dict_S70.pickle',"rb")); 
task_dict    = pickle.load(open('/project/3013060.04/TK_data/results_S70/task_dict_S70.pickle',"rb")); 
task_pd_dict = pickle.load(open('/project/3013060.04/TK_data/results_S70/task_pd_dict_S70.pickle',"rb")); 


#%% FIND TFM TIMESERIES MOST STRONGLY RELATED TO THE TASK 
# ---------------------------------------------------------
# PER SESSION:
Bz_corr_dict      = {}; Bz_pvals_dict     = {}; Bz_fisher_dict = {}; 
maxcor_tfm        = {}; maxcor_absvalue   = {}; maxcor_value   = {}; maxcor_idx     = {}

for ses_idx, ses in enumerate(sessions):   
    Bz_corr_dict[ses], Bz_pvals_dict[ses] = trifle_stats.run_cors2d(task_dict[ses].T, Bz_dict[ses])
    
    # Fisher Z transform
    Bz_fisher = np.zeros(Bz_corr_dict[ses].shape)
    for reg in range(Bz_corr_dict[ses].shape[0]):
        for net in range(Bz_corr_dict[ses].shape[1]):
            Bz_fisher[reg, net] = trifle_stats.r2fisherz(Bz_corr_dict[ses][reg,net])
    del reg, net
    Bz_fisher_dict[ses] = pd.DataFrame(Bz_fisher, index=names_task, columns= names_tfms)
    
    # Find max 
    maxcor_tfm[ses] = abs(Bz_fisher_dict[ses].iloc[0,:]).idxmax()
    maxcor_absvalue[ses] = abs(Bz_fisher_dict[ses].iloc[0,:]).max()
    maxcor_idx[ses] = np.int(maxcor_tfm[ses][3:])-1
    maxcor_value[ses] = Bz_fisher_dict[ses].loc['Visual', maxcor_tfm[ses]]
    del ses_idx, ses

#%% SELECT TASK-POSITIVE TFM; SUBTRACT STATIC M AND REVERSE TIME SERIES IF NEEDED*
# Note, direction is not meaningful (i.e., depending on the direction of M)
# Hence, reversed in case of a negative correlation with the task design
# ---------------------------------------------------------
f_tpm_dict = {}; M_rev={}

for ses in sessions:
    tfm_txt = maxcor_tfm[ses]
    tfm_num = int(tfm_txt[3:])-1
    
    f       = f_dict[ses][:,tfm_num,:]
    N_t      = f.shape[1]
    M        = np.loadtxt(filenames_M[ses])
    static_M = M[:,tfm_num]
    
    ## REVERT TIMESERIES IF NEEDED 
    # --------------------------------------------------
    maxcor = maxcor_value[ses]
    if maxcor < 0:
        f = - f
        static_M = - static_M
    
    ## SUBTRACT M
    # --------------------------------------------------
    step1 = np.squeeze(np.dstack([static_M]*N_t))
    f_minM = np.add(f,-step1)
    
    f_tpm_dict[ses] = f_minM
    M_rev[ses] = static_M
    del tfm_txt, tfm_num, f, N_t, M, static_M, maxcor, step1, f_minM

#%% INTO EPOCHS 
f_epochs_dict              = {}
regressors_epochs_dict     = {}
Ndel_dict                  = {}

for sub_ses in sessions:
    behavior                = np.loadtxt(op.join('/project/3013060.04/TK_data/behavioral/Behavioral_sub_'+str(sub_ses[:2])+'_ses_'+str(sub_ses[3:])+'.txt'), delimiter="\t", skiprows=1)
#TODO: add behavioural to data folder and remove local reference
    onsets                  = behavior[:,:2].T;    
    regressors              = task_dict[sub_ses].T
    tfm_num                 = maxcor_idx[sub_ses];
    
    f_epochs_dict[sub_ses], regressors_epochs_dict[sub_ses], Ndel_dict[sub_ses] = trifle_stats.into_trials(onsets, regressors, TR, Nt_trial, Nt, f_tpm_dict[sub_ses])

#%%% TRIAL AVERAGE
# ---------------------------------------------------------------------------------------------------------------
f_average_dict = {}; f_sem_dict = {}; regs_average_dict = {}

for sub_ses in sessions:
    f_average_dict[sub_ses] = np.mean(f_epochs_dict[sub_ses],axis=1)
    f_sem_dict[sub_ses]     = np.std(f_epochs_dict[sub_ses],axis=1)/np.sqrt(f_epochs_dict[sub_ses].shape[1])
    regs_average_dict[sub_ses] = np.mean(regressors_epochs_dict[sub_ses],axis=1) 

#%% For all pp's
fig, ax = plt.subplots(14,3, figsize=(9.2,22),sharex='col', sharey='row')
pltcount= 1
fig.add_axes=([0.1, 0.1, 0.6, 0.75])
for ses_idx, ses in enumerate(sessions):
    ## PLOT AVERAGE Mt VS. TASK 
    ax          = plt.subplot(14,3,pltcount)
    color_idx   = -1
    
    for net in [17,45,50,12,30]: 
        color_idx=color_idx+1
        ax.errorbar(timerange_m, f_average_dict[ses][net,:], f_sem_dict[ses][net,:],color=mycolors[color_idx], linewidth=0.8, label=names_S70[net])
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

#%% GROUP-LEVEL Z-VALUE PLOTS S70
# ---------------------------------------------------------
# Concatenate across sessions 
import re
participants = ['03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '15', '16', '17']
f70_concat = {}
for pp in participants:
    if re.search('10', pp) or re.search('15', pp):
        keys = [str(pp)+'_01', str(pp)+'_02']
        f70_concat[pp] = np.concatenate((f_epochs_dict[keys[0]], f_epochs_dict[keys[1]]), axis=1)
    else:
        keys = [str(pp)+'_01', str(pp)+'_02', str(pp)+'_03']
        f70_concat[pp] = np.concatenate((f_epochs_dict[keys[0]], f_epochs_dict[keys[1]], f_epochs_dict[keys[2]]), axis=1)

networks_stats_70  = [17,45,50,12,30] #Primary visual, sensorimotor left, sensorimotor right, auditory, DMN (precuneus)
pv = 17
sml = 45
smr = 50
aud = 12
dmn = 30

allpp_tvals     = {}
for net_idx, net in enumerate(networks_stats_70):
    print(net_idx, net)
    pp_tvals = np.zeros([14,60])
    for pp_idx, pp in enumerate(participants):
        pp_tvals[pp_idx,:] = ss.ttest_1samp(f70_concat[pp][net,:,:], popmean=0)[0] #trials by time (60), t across trials 
    allpp_tvals[names_S70[net]]= pp_tvals
    
grouplevel_zvals = {}
for net_idx, net in enumerate(networks_stats_70):
    net_zvals = np.zeros([60])
    for t in range(60):
        net_zvals[t] = np.mean(allpp_tvals[names_S70[net]][:,t])/np.std(allpp_tvals[names_S70[net]][:,t])
    grouplevel_zvals[names_S70[net]] = net_zvals
    del net_zvals
    
# Figure 6a Group level statistics for trial-average f SMITH70
task_4grandaverage = np.zeros([2,40,60])
for ses_idx, sub_ses in enumerate(sessions):
    task_4grandaverage[0,:ses_idx,:] = regs_average_dict[sub_ses][0,:]
    task_4grandaverage[1,:ses_idx,:] = regs_average_dict[sub_ses][1,:]
task_grandaverage = np.mean(task_4grandaverage, axis=1)

plt.figure(figsize=(7.5, 4.8))
plt.title('Group-level Statistics: Subnetwork Involvement (SMITH70)',  font = 'Futura Hv BT', fontweight="bold", fontsize=12)
p1 =plt.plot(timerange_m, grouplevel_zvals[names_S70[pv]], colors_4[0], linewidth=1)
p2 =plt.plot(timerange_m, grouplevel_zvals[names_S70[sml]], colors_4[1] , linewidth=1)
p3= plt.plot(timerange_m, grouplevel_zvals[names_S70[aud]], colors_4[2] , linewidth=1)
p4= plt.plot(timerange_m, grouplevel_zvals[names_S70[dmn]], colors_4[3] , linewidth=1)
plt.xlabel('Time ($s$)', size=11)
plt.ylabel('$Z$-value', size=11)
plt.ylim([-2.5,4.5])

#%% SUCCESFUL VS UNSUCCESSFUL TRIALS 
# ---------------------------------------------------------
succes_dict = {}
for sub_ses in sessions:
    behavior    = np.loadtxt(op.join('/project/3013060.04/TK_data/behavioral/Behavioral_sub_'+str(sub_ses[:2])+'_ses_'+str(sub_ses[3:])+'.txt'), delimiter="\t", skiprows=1)
    succes_dict[sub_ses]      = behavior[:,3]

succes_cut = {}
for sub_ses in sessions:
    behavior                = np.loadtxt(op.join('/project/3013060.04/TK_data/behavioral/Behavioral_sub_'+str(sub_ses[:2])+'_ses_'+str(sub_ses[3:])+'.txt'), delimiter="\t", skiprows=1)
    onsets                  = behavior[:,1];    
    succes                  = behavior[:,3]
    Ntrials                 = onsets.shape[0]; 

    mask_nans               = np.isnan(onsets); sumnans = np.sum(mask_nans); 
    succes[mask_nans]       = np.nan; 
    succes_pd               = pd.DataFrame(succes)
    succes_pd               = succes_pd.dropna(axis=0); 
    if Ndel_t[sub_ses] > 0:
        succes_cut[sub_ses]  = succes_pd[:-Ndel_t[sub_ses]]
    else:
        succes_cut[sub_ses]  =  succes_pd

succes_concat = {}
for pp in participants:
    if re.search('10', pp) or re.search('15', pp):
        keys = [str(pp)+'_01', str(pp)+'_02']
        succes_concat[pp] = np.concatenate((succes_cut[keys[0]], succes_cut[keys[1]]))
    else:
        keys = [str(pp)+'_01', str(pp)+'_02', str(pp)+'_03']
        succes_concat[pp] = np.concatenate((succes_cut[keys[0]], succes_cut[keys[1]], succes_cut[keys[2]]))

f70epoch_hit  = {}; f70epoch_fail = {}
f70avg_trials_hit = {}; f70avg_trials_fail = {}

tvals_hitfail70 = np.zeros([14,70,60]); pvals_hitfail70 = np.zeros([14,70,60])
for pp_idx, pp in enumerate(participants):
    I_hit = np.squeeze(succes_concat[pp]) == 1
    I_fail = np.squeeze(succes_concat[pp]) == 0 
    fe_hit = f70_concat[pp][:,np.squeeze(I_hit),:]
    fe_fail = f70_concat[pp][:,np.squeeze(I_fail),:]
    for net_idx, net in enumerate(names_S70):
        tvals_hitfail70[pp_idx, net_idx,:], pvals_hitfail70[pp_idx, net_idx,:] = ss.ttest_ind(fe_fail[net_idx,:], fe_hit[net_idx,:])

    f70epoch_hit[pp] = fe_hit
    f70epoch_fail[pp] = fe_fail    

    f70avg_trials_hit[pp] = np.mean(fe_hit, axis=1)
    f70avg_trials_fail[pp] = np.mean(fe_fail, axis=1)
    del I_hit, I_fail, fe_hit, fe_fail

# Create dataframes for all pp's averaged over trials
hits70_allpp = np.zeros([14,70, 60])
fails70_allpp = np.zeros([14,70,60])

for pp_idx, pp in enumerate(participants):
    hits70_allpp[pp_idx, :,:] = f70avg_trials_hit[pp]
    fails70_allpp[pp_idx, :,:] = f70avg_trials_fail[pp]   

grouplevel_zvals70 = np.zeros([70,60])
for net_idx, net in enumerate(names_S70):
    for t in range(60):
        grouplevel_zvals70[net_idx, t] = np.mean(tvals_hitfail70[:,net_idx, t])/np.std(tvals_hitfail70[:,net_idx, t])
        del t 

M_groupz_70 = np.mean(abs(grouplevel_zvals70), axis=1)
Max_groupz_70 = np.max(abs(grouplevel_zvals70), axis=1)
M_groupz_pd_70 = pd.DataFrame(M_groupz_70, index=names_S70)
Max_groupz_pd_70 = pd.DataFrame(Max_groupz_70, index=names_S70)
Max_groupz_sort_70 = Max_groupz_pd_70.sort_values(0, ascending=False)
M_groupz_sort_70 = M_groupz_pd_70.sort_values(0, ascending=False)

#%%
sp = 58
ifg = 29
pc = 21
dmn = 30

#%%
ymax= 2.7; ymin = -2.7
# Superior parietal 2 and DMN 
I_maxDMN70= np.where(abs(grouplevel_zvals70[30,:]) == np.max(abs(grouplevel_zvals70[30,:])))[0][0]; tmp_DMN = timerange_m[I_maxDMN70]
I_maxSP70= np.where(abs(grouplevel_zvals70[58,:]) == np.max(abs(grouplevel_zvals70[58,:])))[0][0]; tmp_SP = timerange_m[I_maxSP70]
I_maxIFG70= np.where(abs(grouplevel_zvals70[29,:]) == np.max(abs(grouplevel_zvals70[29,:])))[0][0]; tmp_IFG = timerange_m[I_maxIFG70]
I_maxPC70= np.where(abs(grouplevel_zvals70[21,:]) == np.max(abs(grouplevel_zvals70[21,:])))[0][0]; tmp_PC = timerange_m[I_maxPC70]

# DMN and auditory
# Figure 7a Group level statistics for trial-average f SMITH70
plt.figure(figsize=(7.5, 4.8))
plt.title('Group-level Statistics: Fail – Hit (SMITH70)',  font = 'Futura Hv BT', fontweight="bold", fontsize=12)
plt.plot(timerange_m, grouplevel_zvals70[sp,:], 'k', linestyle='-.', label= "Superior Parietal", linewidth=1) ##008000
plt.fill_between(timerange_m, ymin, ymax, where = (timerange_m == tmp_SP), color='k', alpha=0.5)
plt.plot(timerange_m, grouplevel_zvals70[ifg,:], color= 'black', linestyle='--', label= "IFG", linewidth=1) ##D2691E
plt.fill_between(timerange_m, ymin, ymax, where = timerange_m == tmp_IFG, color='k', linestyle='--', alpha=0.5)
plt.plot(timerange_m, grouplevel_zvals70[pc,:], color= 'black', linestyle='dotted', label= "PC", linewidth=1) ##A52A2A
plt.fill_between(timerange_m, ymin, ymax, where = timerange_m == tmp_PC, color='black',linestyle='dotted', alpha=0.5)
plt.plot(timerange_m, grouplevel_zvals70[dmn,:], color= colors_4[3], label= "DMN (Precuneus)", linewidth=1)
plt.fill_between(timerange_m, ymin, ymax, where = timerange_m == tmp_DMN, color=colors_4[3], alpha=0.5)
plt.xlabel('Time ($s$)', size=11)
plt.ylabel('$Z$-value', size=11)
plt.legend(frameon=False, fontsize=11, loc='upper right') 
plt.ylim([ymin,ymax])

#%% IDENTIFY PARAMETERS FOR FLOBS BASISFUNCTIONS 
#Create dataframe of: 
## PP ID [| SES ID] | TRIAL | Timepoint | BF1 | BF2 | BF3 | BF4 | BF5 | BF6 | f_nw1 .... f_nw20 | 
pps       = ['PP3', 'PP4', 'PP5', 'PP6', 'PP7', 'PP8','PP9', 'PP10', 'PP11','PP12', 'PP13', 'PP15','PP16', 'PP17']
pps_pub = ['PP1', 'PP2', 'PP3', 'PP4', 'PP5', 'PP6','PP7', 'PP8', 'PP9','PP10', 'PP11', 'PP12','PP13', 'PP14']
N_pp      = 14

#Create pp_id, tp_id and session var
pp_id = []; tp_id = []; ses_id = [];
start = 0; ntp_ses = 3000
for I_pp, pp in enumerate(pps):
    if pp == 'PP10' or pp == 'PP15':
        n1 = 6000
        pp_id.extend(n1*[pp])
        tp_id.extend(np.arange(n1))
        ses_id.extend(ntp_ses*['ses_1'])
        ses_id.extend(ntp_ses*['ses_2'])
    else:
        n2 = 9000
        pp_id.extend(n2*[pp])
        tp_id.extend(np.arange(n2))
        ses_id.extend(ntp_ses*['ses_1'])
        ses_id.extend(ntp_ses*['ses_2'])
        ses_id.extend(ntp_ses*['ses_3'])
pp_id = np.array(pp_id); tp_id = np.array(tp_id); ses_id = np.array(ses_id)

# Load basis functions as created with Feat 
names_subses    = ['03_01', '03_02', '03_03', '04_01', '04_02', '04_03', '05_01', '05_02', '05_03', '06_01', '06_02', '06_03', '07_01', '07_02', '07_03', '08_01', '08_02', '08_03', '09_01', '09_02', '09_03', '10_01', '10_02', '11_01', '11_02', '11_03', '12_01', '12_02', '12_03', '13_01', '13_02', '13_03', '15_01', '15_02', '16_01', '16_02', '16_03', '17_01', '17_02', '17_03']
sessions        = list(names_subses)
names_bfs       = ['Visual_HRF', 'Visual_Shift', 'Visual_Disp','Visual_4', 'Visual_5', 'Visual_6','Motor_HRF', 'Motor_Shift', 'Motor_Disp', 'Motor_4', 'Motor_5', 'Motor_6']      
subses_bfs      = np.zeros([120000, 12]); subses_fs = np.zeros([120000, 70]);
start           = 0;

for ses_idx, ses in enumerate(sessions):
    print(ses)
    subses_task      = op.join('/project/3013060.04/TK_data/flobs/IV_basisfunctions/visual_v2_6bfs/design_flobs_'+str(ses)+'.txt')
    bfs              = np.loadtxt(subses_task)
    subses_bfs[start:start+ntp_ses,0:6] = bfs  
    subses_task_m    = op.join('/project/3013060.04/TK_data/flobs/IV_basisfunctions/motor_v2_6bfs/design_flobs_'+str(ses)+'.txt')
    bfs_m            = np.loadtxt(subses_task_m)
    subses_bfs[start:start+ntp_ses, 6:12] = bfs_m
    subses_fs[start:start+ntp_ses,:]  = f_tpm_dict[ses].T
    start = start+ntp_ses
    
succes_reg = []   
succes = {}
for sub_ses in sessions:
    behavior    = np.loadtxt(op.join('/project/3013060.04/TK_data/behavioral/Behavioral_sub_'+str(sub_ses[:2])+'_ses_'+str(sub_ses[3:])+'.txt'), delimiter="\t", skiprows=1)
    succes[sub_ses]      = behavior[:,3]
    reg    = np.zeros([3000])
    behavior_table = np.loadtxt(op.join('/project/3013060.04/TK_data/behavioral/Behavioral_sub_'+str(sub_ses[:2])+'_ses_'+str(sub_ses[3:])+'.txt'), delimiter="\t", skiprows=1)
    onsets_visual= behavior_table[:,0]; onsets_visual_f = onsets_visual/TR
    frame_dif = 60; n_del = sum(onsets_visual_f+60 > 3000)
    epoch_start = np.round(onsets_visual_f[:-n_del],0).astype(int)
    epoch_ends  = epoch_start + frame_dif; epoch_ends = epoch_ends.astype(int)
    s = succes[sub_ses][:epoch_start.shape[0]]
    hit_start = epoch_start[s==1]; hit_end = epoch_ends[s==1]; 
    fail_start= epoch_start[s==0]; fail_end = epoch_ends[s==0];
    for h in range(hit_start.shape[0]):
        reg[hit_start[h]:hit_end[h]] = 1
    for f in range(fail_start.shape[0]):
        reg[fail_start[f]:fail_end[f]] = 2
    succes_reg.extend(reg); 
    del behavior, reg, behavior_table, onsets_visual, onsets_visual_f, n_del, epoch_start, epoch_ends, s, hit_start, fail_start
    
succes_reg = pd.DataFrame(succes_reg, columns=["Succes"])
print(succes_reg)

from patsy import dmatrices 
df = pd.DataFrame({'PP':pp_id, 'Session':ses_id, 'Timepoint':tp_id, 'Visual:HRF':subses_bfs[:,0],'Visual:Shift':subses_bfs[:,1], 'Visual:Dispersion':subses_bfs[:,2],'Visual:4':subses_bfs[:,3],'Visual:5':subses_bfs[:,4], 'Visual:6':subses_bfs[:,5],'Motor:HRF':subses_bfs[:,6],'Motor:Shift':subses_bfs[:,7], 'Motor:Dispersion':subses_bfs[:,8], 'Motor:4':subses_bfs[:,9], 'Motor:5':subses_bfs[:,10], 'Motor:6':subses_bfs[:,11], 'f_primVis_1':subses_fs[:,17], 'f_primVis_2':subses_fs[:,41],  'f_primVis_3':subses_fs[:,53], 'f_primVis_4':subses_fs[:,63], 'f_lateralVis_1':subses_fs[:,10], 'f_lateralVis_2':subses_fs[:,24], 'f_sm_r':subses_fs[:,50], 'f_sm_l':subses_fs[:,45], 'f_aud':subses_fs[:,12], 'f_hipp':subses_fs[:,39], 'f_dmn':subses_fs[:,30]})
from sklearn import preprocessing
scaler_visualHRF = preprocessing.StandardScaler().fit(df[["Visual:HRF"]]); df['VisualHRF_z'] = scaler_visualHRF.transform(df[["Visual:HRF"]])
scaler_visualShift = preprocessing.StandardScaler().fit(df[["Visual:Shift"]]); df['VisualShift_z'] = scaler_visualShift.transform(df[["Visual:Shift"]])
scaler_visualDisp = preprocessing.StandardScaler().fit(df[["Visual:Dispersion"]]); df['VisualDisp_z'] = scaler_visualDisp.transform(df[["Visual:Dispersion"]])
scaler_visual4 = preprocessing.StandardScaler().fit(df[["Visual:4"]]); df['Visual4_z'] = scaler_visual4.transform(df[["Visual:4"]])
scaler_visual5 = preprocessing.StandardScaler().fit(df[["Visual:5"]]); df['Visual5_z'] = scaler_visual5.transform(df[["Visual:5"]])
scaler_visual6 = preprocessing.StandardScaler().fit(df[["Visual:6"]]); df['Visual6_z'] = scaler_visual6.transform(df[["Visual:6"]])
scaler_motorHRF = preprocessing.StandardScaler().fit(df[["Motor:HRF"]]) ; df['MotorHRF_z'] = scaler_motorHRF.transform(df[["Motor:HRF"]])
scaler_motorShift = preprocessing.StandardScaler().fit(df[["Motor:Shift"]]) ; df['MotorShift_z'] = scaler_motorShift.transform(df[["Motor:Shift"]])
scaler_motorDisp = preprocessing.StandardScaler().fit(df[["Motor:Dispersion"]]); df['MotorDisp_z'] = scaler_motorDisp.transform(df[["Motor:Dispersion"]])
scaler_motor4 = preprocessing.StandardScaler().fit(df[["Motor:4"]]); df['Motor4_z'] = scaler_motor4.transform(df[["Motor:4"]])
scaler_motor5 = preprocessing.StandardScaler().fit(df[["Motor:5"]]); df['Motor5_z'] = scaler_motor5.transform(df[["Motor:5"]])
scaler_motor6 = preprocessing.StandardScaler().fit(df[["Motor:6"]]); df['Motor6_z'] = scaler_motor6.transform(df[["Motor:6"]])

df_1 = df[["PP", "Session", "Timepoint", 'f_primVis_1', 'f_primVis_2',  'f_primVis_3', 'f_primVis_4',  'f_lateralVis_1', 'f_lateralVis_2', 'f_sm_r', 'f_sm_l', 'f_aud', 'f_hipp', 'f_dmn', "VisualHRF_z", "VisualShift_z", "VisualDisp_z","Visual4_z", "Visual5_z", "Visual6_z", "MotorHRF_z", "MotorShift_z", "MotorDisp_z","Motor4_z", "Motor5_z", "Motor6_z"]]
df_lm = pd.concat((df_1, pd.get_dummies(succes_reg, drop_first=True)), axis=1)

names_df = df_lm.columns[3:14]
networks_visual = names_df.delete([1,2,3,4,5, 6, 7, 8])
networks_motor = names_df.delete([0,1,2,3,4,5, 9,10])

df2 = pd.DataFrame({
    'PP': pp_id,
    'Session': ses_id,
    'Timepoint': tp_id,
    'Visual:HRF': subses_bfs[:, 0],
    'Visual:Shift': subses_bfs[:, 1],
    'Visual:Dispersion': subses_bfs[:, 2],
    'Visual:4': subses_bfs[:, 3],
    'Visual:5': subses_bfs[:, 4],
    'Visual:6': subses_bfs[:, 5],
    'Motor:HRF': subses_bfs[:, 6],
    'Motor:Shift': subses_bfs[:, 7],
    'Motor:Dispersion': subses_bfs[:, 8],
    'Motor:4': subses_bfs[:, 9],
    'Motor:5': subses_bfs[:, 10],
    'Motor:6': subses_bfs[:, 11],
    'f_primVis_1': subses_fs[:, 17],
    'f_primVis_2': subses_fs[:, 41],
    'f_primVis_3': subses_fs[:, 53],
    'f_primVis_4': subses_fs[:, 63],
    'f_lateralVis_1': subses_fs[:, 10],
    'f_lateralVis_2': subses_fs[:, 24],
    'f_sm_r': subses_fs[:, 50],
    'f_sm_l': subses_fs[:, 45],
    'f_aud': subses_fs[:, 12],
    'f_hipp': subses_fs[:, 39],
    'f_dmn': subses_fs[:, 30],
    'f_sp': subses_fs[:, 58],   # Superior Parietal
    'f_ifg': subses_fs[:, 29],  # Inferior Frontal
    'f_pc': subses_fs[:, 21]    # Paracingulate
})
# === Z-scoring using consistent naming convention ===
scaler_visualHRF = preprocessing.StandardScaler().fit(df2[["Visual:HRF"]])
df2['VisualHRF_z'] = scaler_visualHRF.transform(df2[["Visual:HRF"]])

scaler_visualShift = preprocessing.StandardScaler().fit(df2[["Visual:Shift"]])
df2['VisualShift_z'] = scaler_visualShift.transform(df2[["Visual:Shift"]])

scaler_visualDisp = preprocessing.StandardScaler().fit(df2[["Visual:Dispersion"]])
df2['VisualDisp_z'] = scaler_visualDisp.transform(df2[["Visual:Dispersion"]])

scaler_visual4 = preprocessing.StandardScaler().fit(df2[["Visual:4"]])
df2['Visual4_z'] = scaler_visual4.transform(df2[["Visual:4"]])

scaler_visual5 = preprocessing.StandardScaler().fit(df2[["Visual:5"]])
df2['Visual5_z'] = scaler_visual5.transform(df2[["Visual:5"]])

scaler_visual6 = preprocessing.StandardScaler().fit(df2[["Visual:6"]])
df2['Visual6_z'] = scaler_visual6.transform(df2[["Visual:6"]])

scaler_motorHRF = preprocessing.StandardScaler().fit(df2[["Motor:HRF"]])
df2['MotorHRF_z'] = scaler_motorHRF.transform(df2[["Motor:HRF"]])

scaler_motorShift = preprocessing.StandardScaler().fit(df2[["Motor:Shift"]])
df2['MotorShift_z'] = scaler_motorShift.transform(df2[["Motor:Shift"]])

scaler_motorDisp = preprocessing.StandardScaler().fit(df2[["Motor:Dispersion"]])
df2['MotorDisp_z'] = scaler_motorDisp.transform(df2[["Motor:Dispersion"]])

scaler_motor4 = preprocessing.StandardScaler().fit(df2[["Motor:4"]])
df2['Motor4_z'] = scaler_motor4.transform(df2[["Motor:4"]])

scaler_motor5 = preprocessing.StandardScaler().fit(df2[["Motor:5"]])
df2['Motor5_z'] = scaler_motor5.transform(df2[["Motor:5"]])

scaler_motor6 = preprocessing.StandardScaler().fit(df2[["Motor:6"]])
df2['Motor6_z'] = scaler_motor6.transform(df2[["Motor:6"]])
    
    #%%
df2_1 = df2[[
    "PP", "Session", "Timepoint",
    'f_primVis_1', 'f_primVis_2', 'f_primVis_3', 'f_primVis_4',
    'f_lateralVis_1', 'f_lateralVis_2', 'f_sm_r', 'f_sm_l',
    'f_aud', 'f_hipp', 'f_dmn', 'f_sp', 'f_ifg', 'f_pc',
    "VisualHRF_z", "VisualShift_z", "VisualDisp_z",
    "Visual4_z", "Visual5_z", "Visual6_z",
    "MotorHRF_z", "MotorShift_z", "MotorDisp_z",
    "Motor4_z", "Motor5_z", "Motor6_z"
]]    

df2_lm = pd.concat((df2_1, pd.get_dummies(succes_reg, drop_first=True)), axis=1)
names_df2 = df2_lm.columns[3:19]  # includes all 13 networks now


#%%

bfs_visual_dict  = {}
for sub_ses in sessions:
    bfs_vis_orig = np.loadtxt('/project/3013060.04/TK_data/flobs/IV_basisfunctions/visual_v2_6bfs/design_flobs_'+str(sub_ses)+'.txt')
    scaler = preprocessing.StandardScaler()
    bfs_vis = scaler.fit_transform(bfs_vis_orig)
    behavior_table = np.loadtxt(op.join('/project/3013060.04/TK_data/behavioral/Behavioral_sub_'+str(sub_ses[:2])+'_ses_'+str(sub_ses[3:])+'.txt'), delimiter="\t", skiprows=1)

    ## INTO EPOCHS VISUAL
    onsets_visual    = behavior_table[:,0]; onsets_visual_f = onsets_visual/TR
    frame_dif = 60; Ndel = sum(onsets_visual_f+60 > 3000)
    
    if Ndel > 0:
        epoch_start = np.round(onsets_visual_f[:-Ndel]).astype(int)
    else:
        epoch_start = onsets_visual_f[:].astype(int)
    epoch_ends  = epoch_start + frame_dif; epoch_ends = epoch_ends.astype(int)
    Ntrials = epoch_start.shape[0]-Ndel
    bfs_cut = np.zeros([Ntrials,60,6])
    
    for ti in range(Ntrials):
        bfs_cut[ti,:,:] = bfs_vis[epoch_start[ti]:epoch_ends[ti], :]
    bfs_visual_dict[sub_ses] = bfs_cut
    del bfs_cut
 
#%%
bfs_visual_all = np.zeros([40,60,6]); 
for ses_idx, sub_ses in enumerate(sessions):
    for bf_idx in range(6):
        bfs_visual_all[ses_idx,:,bf_idx] = np.mean(bfs_visual_dict[sub_ses][:,:,bf_idx], axis=0)

#%%
bfs_vis_gm = np.zeros([60,6]);
for bf_idx in range(6):
    bfs_vis_gm[:,bf_idx] = np.nanmean(bfs_visual_all[:,:,bf_idx], axis=0)
#%%
# Predictor names (adjust according to your formula structure)
predictors = ['VisualHRF_z', 'VisualShift_z', 'VisualDisp_z', 'Visual4_z', 'Visual5_z', 'Visual6_z']

results_visual = {}; contrasts_visual = {}; pvals = []; results_params = {}; results_se = {}
pval_labels = []  # To track predictors and networks for each p-value

# Iterate over each network
for net_idx, net in enumerate(names_df):
    print(net)
    
    # Set up the design matrix and model
    y, X = dmatrices(str(net) + '~ VisualHRF_z + VisualShift_z + VisualDisp_z + Visual4_z + Visual5_z + Visual6_z', data=df_lm, return_type='dataframe')    
    design = X.iloc[:,1:]  # Drop the intercept column
    model = sm.GLSAR(y, design, ar_lag)
    
    # Fit the model
    results = model.iterative_fit()
    
    # Store results
    results_visual[net] = results
    results_params[net] = results.params
    results_se[net] = results.bse
    
    # Extend p-values list with the current network's p-values
    pvals.extend(results.pvalues)
    
    # Track which network and predictor each p-value belongs to
    for predictor in predictors:
        pval_labels.append((net, predictor))
    
    print(results.summary())

# Multiple comparison correction (FDR)
from statsmodels.stats.multitest import fdrcorrection
mc_pvals = fdrcorrection(pvals)

# Significance threshold
alpha = 0.05

# Associate corrected p-values with their corresponding predictors and networks
corrected_pval_dict = {
    label: {
        'corrected_pval': mc_pval,
        'significant': mc_pval < alpha
    } for label, mc_pval in zip(pval_labels, mc_pvals[1])
}

# Print out the corrected p-values with the network, predictor, and significance
for label, stats in corrected_pval_dict.items():
    net, predictor = label
    corrected_pval = stats['corrected_pval']
    is_significant = stats['significant']
    significance_str = "Significant" if is_significant else "Not significant"
    print(f"Network: {net}, Predictor: {predictor}, Corrected p-value: {corrected_pval}, {significance_str}")


#%%
plt.figure(figsize=(7.5, 4.8))
plt.title('Group-level: Fitted Model of Network Involvement (SMITH70)',  font = 'Futura Hv BT', fontweight="bold", fontsize=12)
#plt.step(timerange_m, bc, where='post', label='Visual Stimulus', color='black', linewidth=0.8)
plt.plot(timerange_m, np.dot(bfs_vis_gm,results_params['f_primVis_1']),color=colors_4[0], linewidth=1, label='Occipital Pole')
plt.fill_between(timerange_m,np.dot(bfs_vis_gm, results_params['f_primVis_1']-results_se['f_primVis_1']), y2= np.dot(bfs_vis_gm, results_params['f_primVis_1']+results_se['f_primVis_1']), color=colors_4[0], alpha=0.25)
plt.plot(timerange_m, np.dot(bfs_vis_gm,results_params['f_sm_l']),color=colors_4[1], linewidth=1, label='Sensorimotor (LH)')
plt.fill_between(timerange_m,np.dot(bfs_vis_gm, results_params['f_sm_l']-results_se['f_sm_l']), y2= np.dot(bfs_vis_gm, results_params['f_sm_l']+results_se['f_sm_l']), color=colors_4[1], alpha=0.25)
plt.plot(timerange_m, np.dot(bfs_vis_gm,results_params['f_sm_r']),color=colors_4[1], linewidth=1, linestyle='--', label='Sensorimotor (RH)')
plt.fill_between(timerange_m,np.dot(bfs_vis_gm, results_params['f_sm_r']-results_se['f_sm_r']), y2= np.dot(bfs_vis_gm, results_params['f_sm_r']+results_se['f_sm_r']), color=colors_4[1], linestyle= '--', alpha=0.25)
plt.plot(timerange_m, np.dot(bfs_vis_gm,results_params['f_aud']),color=colors_4[2], linewidth=1, label='Auditory')
plt.fill_between(timerange_m,np.dot(bfs_vis_gm, results_params['f_aud']-results_se['f_aud']), y2= np.dot(bfs_vis_gm, results_params['f_aud']+results_se['f_aud']), color=colors_4[2], alpha=0.25)
plt.plot(timerange_m, np.dot(bfs_vis_gm,results_params['f_dmn']),color=colors_4[3], linewidth=1, label='DMN (Precuneus)')
plt.fill_between(timerange_m,np.dot(bfs_vis_gm, results_params['f_dmn']-results_se['f_dmn']), y2= np.dot(bfs_vis_gm, results_params['f_dmn']+results_se['f_dmn']), color=colors_4[3], alpha=0.25)
plt.xlabel('Time ($s$)', size=11)
plt.ylabel('Amplitude', size=11)
plt.legend(frameon=False)
#plt.savefig('/home/mrstats/tamdklo/figures/P1_trifle/S70/Rec_Grouplevel_SMITH70.png', dpi=700)

#%%

results_visual_succes = {}; contrasts_visual_succes = {}; results_params_succes = {}; results_se_succes = {}
results_tvals_succes = {}; results_pvals_succes = {}
names_df = df2_lm.columns[14:17]

for net_idx, net in enumerate(names_df):
    print(net)
    y, X = dmatrices(str(net)+'~ VisualHRF_z + VisualShift_z + VisualDisp_z + Visual4_z + Visual5_z + Visual6_z + C(Succes, Treatment) + VisualHRF_z*C(Succes, Treatment) + VisualShift_z*C(Succes, Treatment) + VisualDisp_z*C(Succes, Treatment)+Visual4_z*C(Succes, Treatment)+ Visual5_z*C(Succes, Treatment)+Visual6_z*C(Succes, Treatment)' , data=df2_lm, return_type='dataframe')    
    design = X.iloc[:,1:]
    model = sm.GLSAR(y, design, ar_lag)
    results = model.iterative_fit()
    results_visual_succes[net] = results
    results_params_succes[net] = results.params
    results_se_succes[net] = results.bse
    results_tvals_succes[net] = results.tvalues
    results_pvals_succes[net] = results.pvalues
    contrasts_visual_succes[net] = results.t_test(["C(Succes, Treatment)[T.2.0] - C(Succes, Treatment)[T.1.0],VisualHRF_z:C(Succes, Treatment)[T.2.0] - VisualHRF_z:C(Succes, Treatment)[T.1.0], VisualShift_z:C(Succes, Treatment)[T.2.0]-VisualShift_z:C(Succes, Treatment)[T.1.0],VisualDisp_z:C(Succes, Treatment)[T.2.0]-VisualDisp_z:C(Succes, Treatment)[T.1.0], Visual4_z:C(Succes, Treatment)[T.2.0] - Visual4_z:C(Succes, Treatment)[T.1.0], Visual5_z:C(Succes, Treatment)[T.2.0] - Visual5_z:C(Succes, Treatment)[T.1.0], Visual6_z:C(Succes, Treatment)[T.2.0] - Visual6_z:C(Succes, Treatment)[T.1.0] "])
    print(results.summary())


#%% 
allpvals_succes = []
for reg in results_pvals_succes.values():
    allpvals_succes.extend(reg)

#%% MC correction 
# existing tools to check our implementation against
from statsmodels.stats.multitest import fdrcorrection
mc_pvals_succes = fdrcorrection(allpvals_succes)

#%%
regs = ['Hit', 'Fail', 'bf1', 'bf1_hit', 'bf1_fail', 'bf2', 'bf2_hit', 'bf2_fail', 'bf3', 'bf3_hit', 'bf3_fail', 'bf4', 'bf4_hit', 'bf4_fail', 'bf5', 'bf5_hit', 'bf5_fail', 'bf6', 'bf6_hit', 'bf6_fail']
names_mcpvals = []
for net in names_df:
    for i in regs:
        names_mcpvals.extend([str(net)+'_'+str(i)])

#%%
mc_pvals_df = pd.concat((pd.DataFrame(names_mcpvals), pd.DataFrame(mc_pvals_succes[0]), pd.DataFrame(mc_pvals_succes[1])), axis=1)

#%%
from statsmodels.stats.multitest import fdrcorrection

pvals_succes_dmn = []
net = 'f_dmn'

# Set up design matrix and model
y, X = dmatrices(str(net)+'~ VisualHRF_z + VisualShift_z + VisualDisp_z + Visual4_z + Visual5_z + Visual6_z + C(Succes, Treatment) + VisualHRF_z*C(Succes, Treatment) + VisualShift_z*C(Succes, Treatment) + VisualDisp_z*C(Succes, Treatment)+Visual4_z*C(Succes, Treatment)+ Visual5_z*C(Succes, Treatment)+Visual6_z*C(Succes, Treatment)' , data=df_lm, return_type='dataframe')    
design = X.iloc[:,1:]
model = sm.GLSAR(y, design, ar_lag)
results = model.iterative_fit()

# Store the results
results_visual_succes[net] = results
results_params_succes[net] = results.params
results_se_succes[net] = results.bse
pvals_succes_dmn.extend(results.pvalues)
results_tvals_succes[net] = results.tvalues
results_pvals_succes[net] = results.pvalues

# Define contrasts
contrasts_visual_succes[net] = results.t_test([
    "C(Succes, Treatment)[T.2.0] - C(Succes, Treatment)[T.1.0]",
    "VisualHRF_z:C(Succes, Treatment)[T.2.0] - VisualHRF_z:C(Succes, Treatment)[T.1.0]",
    "VisualShift_z:C(Succes, Treatment)[T.2.0] - VisualShift_z:C(Succes, Treatment)[T.1.0]",
    "VisualDisp_z:C(Succes, Treatment)[T.2.0] - VisualDisp_z:C(Succes, Treatment)[T.1.0]",
    "Visual4_z:C(Succes, Treatment)[T.2.0] - Visual4_z:C(Succes, Treatment)[T.1.0]",
    "Visual5_z:C(Succes, Treatment)[T.2.0] - Visual5_z:C(Succes, Treatment)[T.1.0]",
    "Visual6_z:C(Succes, Treatment)[T.2.0] - Visual6_z:C(Succes, Treatment)[T.1.0]"
])

# Extract contrast results (?b, t, p-values)
delta_b = contrasts_visual_succes[net].effect       # Change in beta (?b)
t_values = contrasts_visual_succes[net].tvalue      # t-values for the contrasts
p_values = contrasts_visual_succes[net].pvalue      # p-values for the contrasts

# Apply FDR correction to the contrast p-values
alpha = 0.05  # Significance level
fdr_reject, fdr_corrected_pvals = fdrcorrection(p_values, alpha=alpha)

# Display the original and FDR-corrected results
print("\nContrast Results:")
print(f"Change in beta (?b): {delta_b}")
print(f"t-values: {t_values}")

# Format p-values and FDR-corrected p-values for display
formatted_p_values = [f"{pval:.6f}" for pval in p_values]
formatted_fdr_p_values = [f"{pval:.6f}" for pval in fdr_corrected_pvals]

print(f"Original p-values: {formatted_p_values}")
print(f"FDR-corrected p-values: {formatted_fdr_p_values}")

# Also display whether each contrast was significant after FDR correction
for i, reject in enumerate(fdr_reject):
    print(f"Contrast {i+1} significant after FDR correction: {reject}")



#%% Hit
dmn_hit = np.array([results_params_succes['f_dmn'][2]+results_params_succes['f_dmn'][3],
results_params_succes['f_dmn'][5]+results_params_succes['f_dmn'][6],
results_params_succes['f_dmn'][8]+results_params_succes['f_dmn'][9],
results_params_succes['f_dmn'][11]+results_params_succes['f_dmn'][12],
results_params_succes['f_dmn'][14]+results_params_succes['f_dmn'][15],
results_params_succes['f_dmn'][17]+results_params_succes['f_dmn'][18]])

#%% Fail
dmn_fail = np.array([results_params_succes['f_dmn'][2]+results_params_succes['f_dmn'][4],
results_params_succes['f_dmn'][5]+results_params_succes['f_dmn'][7],
results_params_succes['f_dmn'][8]+results_params_succes['f_dmn'][10],
results_params_succes['f_dmn'][11]+results_params_succes['f_dmn'][12],
results_params_succes['f_dmn'][14]+results_params_succes['f_dmn'][16],
results_params_succes['f_dmn'][17]+results_params_succes['f_dmn'][19]])

#%% Se Hit
dmn_hit_se = np.array([results_se_succes['f_dmn'][2]+results_se_succes['f_dmn'][3],
results_se_succes['f_dmn'][5]+results_se_succes['f_dmn'][6],
results_se_succes['f_dmn'][8]+results_se_succes['f_dmn'][9],
results_se_succes['f_dmn'][11]+results_se_succes['f_dmn'][12],
results_se_succes['f_dmn'][14]+results_se_succes['f_dmn'][15],
results_se_succes['f_dmn'][17]+results_se_succes['f_dmn'][18]])

#%% Se Fail
dmn_fail_se = np.array([results_se_succes['f_dmn'][2]+results_se_succes['f_dmn'][4],
results_se_succes['f_dmn'][5]+results_se_succes['f_dmn'][7],
results_se_succes['f_dmn'][8]+results_se_succes['f_dmn'][10],
results_se_succes['f_dmn'][11]+results_se_succes['f_dmn'][12],
results_se_succes['f_dmn'][14]+results_se_succes['f_dmn'][16],
results_se_succes['f_dmn'][17]+results_se_succes['f_dmn'][19]])


#%%
plt.figure(figsize=(7.5, 4.8))
plt.title('Group-level: Fitted Model of Network Involvement: Hit vs. Fail (SMITH70)',  font = 'Futura Hv BT', fontweight="bold", fontsize=12)
#plt.step(timerange_m, bc, where='post', label='Visual Stimulus', color='black', linewidth=0.8)
plt.plot(timerange_m, np.dot(bfs_vis_gm,dmn_hit), color='green', linewidth=1, label='Hits')
plt.fill_between(timerange_m,np.dot(bfs_vis_gm, dmn_hit-dmn_hit_se), y2= np.dot(bfs_vis_gm, dmn_hit + dmn_hit_se), color='green', alpha=0.25)
plt.plot(timerange_m, np.dot(bfs_vis_gm, dmn_fail), color='red', linewidth=1, label='Fails')
plt.fill_between(timerange_m,np.dot(bfs_vis_gm, dmn_fail-dmn_fail_se), y2= np.dot(bfs_vis_gm, dmn_fail + dmn_fail_se), color='red', alpha=0.25)
plt.xlabel('Time ($s$)', size=11)
plt.ylabel('Amplitude', size=11)
plt.ylim([-0.9, 0.85])
plt.legend(frameon=False)
plt.savefig('/home/mrstats/tamdklo/figures/P1_trifle/S70/Rec_Grouplevel_SMITH70_hitfail.png', dpi=700)

#%% SM IX Spatial cors
sessions_pub = ['01_01', '01_02', '01_03','02_01', '02_02', '02_03', '03_01', '03_02', '03_03', '04_01', '04_02', '04_03', '05_01', '05_02', '05_03', '06_01', '06_02', '06_03', '07_01', '07_02', '07_03', '08_01', '08_02', '09_01', '09_02', '09_03', '10_01', '10_02', '10_03', '11_01', '11_02', '11_03', '12_01', '12_02', '13_01', '13_02', '13_03', '14_01', '14_02', '14_03' ]
names_S10_orig       = ['Occipital Pole','Sensorimotor (LH)','Sensorimotor (RH)','Auditory', 'DMN(Precuneus', 'Superior Parietal', 'IFG', 'Posterior Cingulate']
M_all = np.zeros([8,40])
nets_idx = [17,45,50,12,30, 58,29,21]
for ses_idx, ses in enumerate(sessions):
    M_all[:,ses_idx] = M_rev[ses][nets_idx]
    #M_all[:,ses_idx] = M_dict[ses][[1,2,5,6,7,8,9,11,12,15],maxcor_idx[ses]]

M_all_pd = pd.DataFrame(M_all, columns=sessions_pub, index= names_S10_orig)
M_all_cor = M_all_pd.corr()
mask = np.triu(np.ones_like(M_all_cor, dtype=bool))

plt.figure(figsize=(7.5, 4.8))
sns.heatmap(M_all_cor, mask = mask, cmap=cmap, vmin=-1, vmax=1)
plt.title("Spatial Correlations Task-Relevant TFMs (SMITH70)",  font = 'Futura Hv BT', fontweight="bold", fontsize=12)
plt.xlabel("Task Run")
plt.ylabel("Task Run")
plt.tight_layout()
plt.savefig('/home/mrstats/tamdklo/figures/P1_trifle/S70/SMIX_spatialcors_S70.png', dpi=700)
