## 📄 Paper Reproducibility Scripts

This folder contains the version of the code used in our publication:

> **Citation**  
> Tamara Jedidja de Kloe, Zahra Fazal, Nils Kohn, David Gordon Norris, Ravi Shankar Menon, Alberto Llera, Christian Friedrich Beckmann  
> *Time-Resolved Instantaneous Functional Loci Estimation (TRIFLE): Estimating Time-Varying Allocation of Spatially Overlapping Sources in Functional Magnetic Resonance Imaging*.  
> **Imaging Neuroscience (2025)**  
> DOI: [https://doi.org/10.1162/IMAG.a.58](https://doi.org/10.1162/IMAG.a.58)

---

### 📦 What’s Included

- `pub_masterscript_s20.py`: Runs TRIFLE using the **SMITH20** template.
- `pub_masterscript_s70.py`: Runs TRIFLE using the **SMITH70** template.
- `trifle_module/`: A frozen version of the module used during the original analysis.  

> ⚠️ **Note:** This version of the module is distinct from the installable `trifle` package in the project root. It reflects the code as used during peer-reviewed analysis.

---

### ▶️ How to Run

No installation of the `trifle` package is required. Simply clone or download the `triflepaper_scripts/` folder and execute the scripts directly. The scripts automatically use the local `trifle_module/` in this folder.

---

### 📂 Data Availability

The data used in the TRIFLE paper will be stored at the **Donders Repository**:  
🔗 https://data.donders.ru.nl/

Access is available **upon request**, in accordance with institutional guidelines.

The installable Python package for TRIFLE is available at:  
🔗 https://github.com/tamarajedidja/trifle  
_Requires Python 3.6 or later._
