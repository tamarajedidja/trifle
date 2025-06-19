# TRIFLE: Time-Resolved Instantaneous Functional Loci Estimation
*Estimating Time-Varying Allocation of Spatially Overlapping Sources in Functional Magnetic Resonance Imaging*

**Developed at**  
Donders Centre for Cognitive Neuroimaging (DCCN),  
Radboud University, Nijmegen, The Netherlands

**Author:** TJ de Kloe

---
## 🪜 Overview

This repository contains:

1. **The TRIFLE module** for running the full pipeline on fMRI data.  
   _Note: This module is under construction. A streamlined CLI version will be released soon._

2. **Scripts used in the TRIFLE paper**  
   These scripts are located in the `triflepaper_scripts/` folder and reproduce all core figures and analyses from the publication.

   > **Citation**  
   > de Kloe, T. J., Fazal, Z., Kohn, N., Norris, D. G., Menon, R. S., Llera, A., & Beckmann, C. F. (2025).  
   > *Time-Resolved Instantaneous Functional Loci Estimation (TRIFLE): Estimating Time-Varying Allocation of Spatially Overlapping Sources in Functional Magnetic Resonance Imaging*.  
   > *Imaging Neuroscience.*  
   > DOI: [https://doi.org/10.1162/IMAG.a.58](https://doi.org/10.1162/IMAG.a.58)
 
---
## 🧠 About TRIFLE

**TRIFLE** is a multivariate time-varying functional connectivity method that explicitly accounts for spatial overlap. It extends **Temporal Functional Mode (TFM)** analysis (Smith et al., 2012) to estimate time-varying changes in the linear mixing of spatially and temporally independent sources in fMRI data.

In case you use TRIFLE, please cite the following paper:

   > **Citation**  
   > de Kloe, T. J., Fazal, Z., Kohn, N., Norris, D. G., Menon, R. S., Llera, A., & Beckmann, C. F. (2025).  
   > *Time-Resolved Instantaneous Functional Loci Estimation (TRIFLE): Estimating Time-Varying Allocation of Spatially Overlapping Sources in Functional Magnetic Resonance Imaging*.  
   > *Imaging Neuroscience.*  
   > DOI: [https://doi.org/10.1162/IMAG.a.58](https://doi.org/10.1162/IMAG.a.58)

---
## 📦 Package Contents

```
trifle/
├── trifle_module/           # In-development module for running TRIFLE
│   ├── trifle_main.py
│   ├── trifle_spatial.py
│   ├── trifle_temporal.py
│   ├── trifle_timevaryingmixing.py
│   ├── utils.py
│   ├── trifle_dataload.py
│   ├── trifle_analysis.py
│   ├── trifle_stats.py
├── triflepaper_scripts/     # Scripts used to produce results in the TRIFLE paper
│   ├── README.md
│   ├── pub_masterscript_s20.py
│   ├── pub_masterscript_s70.py
│   └── trifle_module/
│       .... 
├── setup.py           
└── README.md
```

