# 📦 Physics BE Clustering Pipeline

A reproducible Python pipeline for:

- converting Igor ITX datasets to structured CSV  
- preprocessing Brillouin Energy (BE) time-series signals  
- extracting statistical & PCA-based features  
- performing clustering with spatial weighting and smoothing  
- generating visual cluster maps  
- computing evaluation metrics

The pipeline is fully automated — users only provide the raw ITX files.  
All intermediate datasets and results are generated programmatically.

---

## ✨ Features

✔ ITX → CSV reshape into `(kx, ky, BE, time-series)` format  
✔ Dataset split into BE-indexed subsets  
✔ Robust time-series preprocessing:

- baseline correction  
- Savitzky–Golay smoothing  
- moving-average denoising  
- percentile-based normalization  
- valid-signal masking  

✔ Feature extraction per pixel:

- mean / std / max intensity  
- skewness  
- time-series PCA (PC1)

✔ Clustering pipelines

- **Set 1 — K-Means + spatial weighting + smoothing (default)**
- **Set 2 — Hyperparameter / GMM variant (optional — disabled by default due to runtime)**

✔ Cluster map visualization  
✔ Evaluation metrics:

- Silhouette  
- Davies–Bouldin  
- Calinski–Harabasz  

✔ End-to-end execution via:

```bash
python run_pipeline.py
```

---

## 🧑‍💻 Author

**Mohammadsadra Amini**

---

## 📂 Project Structure

```
physics_data_analysis/
│
├─ src/
│   ├─ io/
│   ├─ preprocessing.py
│   ├─ feature_extraction.py
│   ├─ clustering.py                     # Clustering Set 1 (default)
│   ├─ cluster_hyperparam_search_v2.py   # Clustering Set 2 (optional / heavy)
│   ├─ visualization.py
│   ├─ evaluation.py
│   └─ config.py
│
├─ data/        # user-provided + generated (ignored in git)
│   └─ itx/     # place ITX files here
│
├─ results/     # generated plots & evaluation (ignored in git)
│
├─ run_pipeline.py
├─ requirements.txt
├─ .env.example
├─ .gitignore
└─ README.md
```

> The contents of `data/` and `results/` are generated automatically  
> and intentionally excluded from version control.

---

## 🧩 Input Data

Place the three ITX files in:

```
data/itx/
  4D Datastack.itx
  BE.itx
  Time Steps.itx
```

They are reshaped into:

```
(kx, ky, BE, time-series)
```

grid-indexed form.

---

## ⚙️ Installation & Setup

## ✅ Requirements

- **Python 3.9 or higher**  
  *(Recommended: Python 3.10 or 3.11)*  
- **pip** (Python package manager)

If Python is not installed on your system, download it from:  
https://www.python.org/downloads/

### 1️⃣ Clone repository

```bash
git clone https://github.com/sadramini/physics_data_analysis.git
cd physics_data_analysis
```

### 2️⃣ Create virtual environment

macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Configure environment paths

```bash
cp .env.example .env
```

Defaults:

```
DATA_DIR=./data
RESULTS_DIR=./results
```

No changes are normally required.

### 5️⃣ Add ITX files

Place them in:

```
data/itx/
```

---

# ▶️ Run the Full Pipeline

From project root:

```bash
python run_pipeline.py
```

The pipeline executes:

1) ITX → CSV reshape  
2) Split dataset by BE  
3) Preprocessing  
4) Feature extraction  
5) **Clustering Set 1 (default)**  
6) Visualization  
7) Evaluation  

---

## ⚠️ Note about Clustering Set 2 (Hyperparameter / GMM)

The script:

```
src/cluster_hyperparam_search_v2.py
```

performs an **extended hyperparameter search** and supports:

- alternative feature sets  
- spatial coherence scoring  
- Gaussian Mixture clustering  

However:

> ⏳ It is computationally expensive and takes a long time to run.

For this reason:

- its execution is **currently commented out in `run_pipeline.py`**
- Clustering Set 1 is used as the default pipeline

Users who wish to run Set 2 may:

- manually enable the step in `run_pipeline.py`, or  
- run it separately via:

```bash
python -m src.cluster_hyperparam_search_v2
```

---

# 🗂️ Output Locations

### Intermediate generated datasets

```
data/
  raw/
  BE_datasets/
  Preprocessed_BE_datasets/
  Extracted_Features/
  Clustered_BE_datasets/
```

### Results & Plots

```
results/
  Cluster_Plots/<version>/
  evaluation/<version>/
```

Outputs include:

- BE cluster maps (PNG)  
- final clustered datasets  
- evaluation metric tables  
- hyperparameter summaries (when enabled)

---

# 📊 Evaluation Metrics

For each BE dataset the pipeline reports:

| Metric | Meaning |
|--------|--------|
| Silhouette | separation quality |
| Davies–Bouldin | cluster compactness |
| Calinski–Harabasz | variance ratio |

Saved to:

```
results/evaluation/<version>/clustering_metrics_summary.csv
```

---

# 🧱 Reproducibility Design

This project is structured as a research-grade pipeline:

✔ deterministic preprocessing & feature extraction  
✔ centralized path config via `.env` + `config.py`  
✔ outputs generated — not stored in repo  
✔ rerunnable on any machine  
✔ suitable for experiment comparison

---

# 📜 License — MIT

This project is released under the **MIT License**.

Copyright (c) 2025 Mohammadsadra Amini

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

---
## 📧 Contact
 
University webpage: [https://your-university-page-link](https://wiwi.tu-dortmund.de/fakultaet/fakultaetsangehoerige/mohammadsadra-amini/)
