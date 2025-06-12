# TRIFLE: Time-Resolved Instantaneous Functional Loci Estimation
*Estimating Time-Varying Allocation of Spatially Overlapping Sources in Functional Magnetic Resonance Imaging*

**Developed at**\
Donders Centre for Cognitive Neuroimaging (DCCN),\
Donders Institute for Brain, Cognition and Behaviour,\
Radboud University, Nijmegen, The Netherlands

**Author:** TJ de Kloe

---

## 🧠 About TRIFLE

**TRIFLE** is a multivariate time-varying functional connectivity method that explicitly accounts for spatial overlap. It extends **Temporal Functional Mode (TFM)** analysis (Smith et al., 2012) to estimate time-varying changes in the linear mixing of spatially and temporally independent sources in fMRI data.

In case you use TRIFLE, please cite the following paper:

> **Citation**\
> Tamara Jedidja de Kloe, Zahra Fazal, Nils Kohn, David Gordon Norris, Ravi Shankar Menon, Alberto Llera, Christian Friedrich Beckmann\
> *Time-Resolved Instantaneous Functional Loci Estimation (TRIFLE): Estimating Time-Varying Allocation of Spatially Overlapping Sources in Functional Magnetic Resonance Imaging*.\
> **Imaging Neuroscience (2025)**\
> DOI: [https://doi.org/10.1162/IMAG.a.58](https://doi.org/10.1162/IMAG.a.58)

---

## 📦 Package Contents

```
trifle/
├── trifle_main.py          # Full TRIFLE pipeline 
├── trifle_spatial.py       # CLI for group-level spatial ICA & individual-level Dual Regression
├── trifle_temporal.py      # CLI for temporal ICA 
├── trifle_dataload.py      # Data loading and prep utilities
├── trifle_analysis.py      # Core TRIFLE algorithm
├── trifle_stats.py         # Optional: Post-hoc stats & visualisation
├── templates/
│   ├── Smith20.nii.gz      # Included templates
│   └── Smith70.nii.gz
├── setup.py                # Installation script
├── README.md               # This file
└── LICENSE
```

---

## ⚙️ Requirements

### 🐍 Python

TRIFLE requires Python 3.6+ and is best installed via the provided environment.yaml file:
```bash
conda env create -f environment.yaml
conda activate trifle
```

The environment installs all necessary dependencies, including:
	•	numpy, scipy, pandas
	•	matplotlib, seaborn
	•	nibabel, nilearn
	•	scikit-learn, statsmodels
	•	nipype, tqdm

### 🧰 System Dependencies

- **FSL** (version ≥ 6.0)
  - Required for spatial ICA (`melodic`) and dual regression.
  - Must be sourced before use:

```bash
source $FSLDIR/etc/fslconf/fsl.sh
```

📌 [FSL Installation Guide](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation)

---

## 🔧 Installation

### 🐍 Create Conda Environment

```bash
git clone https://github.com/tamarajedidja/trifle.git
cd trifle
conda env create -f environment.yaml
conda activate trifle
pip install -e .
```

This installs TRIFLE as an editable Python package with command-line tools enabled.

---

## 🚀 Running TRIFLE

You can run TRIFLE in two ways:

1. 🔁 Full Pipeline
Use `trifle_run` to automatically execute the full TRIFLE workflow, either for:
	•	Group-level analysis (group ICA), or:
	•	Individual-level analysis (dual regression)

**Group ICA (Full Pipeline)**
```bash
trifle_run \
  --level group \
  --input list_of_niftis.txt \
  --tr 1.5 \
  --output path/to/output_dir \
  --spatial_modelorder 20 \
  --temporal_modelorder 10
```

**Dual Regression (Individual-level)**
```bash
trifle_run \
  --level individual \
  --input subj1.nii.gz \
  --tr 1.5 \
  --output path/to/output_dir \
  --template smith20 \
  --temporal_modelorder 10
```

2. 🧩 Individual modules: 

**1. Spatial decomposition**
_Group-level:_
```bash
trifle_spatial \
  --level group \
  --input list_of_niftis.txt \
  --output path/to/spatial_output \
  --tr 1.5 \
  --n_components 20
```
_or for individual-level:_
```bash
trifle_spatial \
  --level individual \
  --input subj1.nii.gz \
  --output path/to/spatial_output \
  --template smith20
```

**2. Temporal Decomposition**
Run temporal ICA on extracted time series (melodic_mix or dr_stage1 file):

```bash
trifle_temporal \
  --input path/to/melodic_mix.txt \
  --output path/to/temporal_output \
  --model_order 10
```

**3. Time-Varying Mixing Estimation**
Run TRIFLE analysis on decomposed outputs to compute the time-varying mixing matrix:

```bash
trifle_tvmixing \
  --input subj1.nii.gz \
  --S path/to/spatial_output/melodic_IC.nii.gz \
  --T path/to/spatial_output/melodic_mix \
  --M path/to/temporal_output/tica_k10/M.pkl \
  --B path/to/temporal_output/tica_k10/B.pkl \
  --output path/to/final_output_dir \
  --mask path/to/mask.nii.gz
```

---

## 📂 Output

A ZIP file of all outputs is automatically created.

---

## 🤝 Support

Please open an issue on [GitHub](https://github.com/tamarajedidja/trifle/issues) for questions or bugs.
