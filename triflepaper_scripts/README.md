## Paper Reproducibility Scripts

This folder contains the version of the code used in our publication.
> **Citation**\
> Tamara Jedidja de Kloe, Zahra Fazal, Nils Kohn, David Gordon Norris, Ravi Shankar Menon, Alberto Llera, Christian Friedrich Beckmann\
> *Time-Resolved Instantaneous Functional Loci Estimation (TRIFLE): Estimating Time-Varying Allocation of Spatially Overlapping Sources in Functional Magnetic Resonance Imaging*.\
> **Imaging Neuroscience (2025)**\
> DOI: [https://doi.org/10.1162/IMAG.a.58](https://doi.org/10.1162/IMAG.a.58)
---
### What’s here

- `pub_masterscript_s20.py`: Runs TRIFLE using SMITH20 template.
- `pub_masterscript_S70.py`: Runs TRIFLE using SMITH70 template.
- `trifle_module/`: A frozen copy of the module used during the original analysis. 

> ⚠️ **Note:** This module is distinct from the installable trifle package in the main project root. It reflects the version used during peer-reviewed analysis. 

### How to Run

No installation of the trifle package is required. Simply download or clone the triflepaper_scripts/ folder and run the scripts directly. The scripts automatically use the local trifle_module/ in this folder. 