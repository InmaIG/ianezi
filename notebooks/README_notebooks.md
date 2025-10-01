
# Demo notebooks

This folder contains a minimal **demo** for **m3DinAI** that can be run **without real images**.

## Files
- **m3DinAI_demo.ipynb** — Self‑contained notebook that generates a small synthetic feature table, runs 2D UMAP, and performs a simple Welch’s t‑test vs DMSO.

## Prerequisites
- Python 3.10
- Packages from the project root: `pip install -r ../requirements.txt`
- Jupyter Notebook: `python -m pip install jupyter`

## How to run
From the repository root:
```bash
# (optional) create & activate the env
python -m venv .venv
# Linux/Mac:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

pip install -r requirements.txt
python -m pip install jupyter

# open the demo
jupyter notebook notebooks/m3DinAI_demo.ipynb
```

## What the demo does
- Creates `results/demo/demo_spheroid_features_trat.xlsx` with synthetic features and `Treatment` labels.
- Standardizes features and computes a **UMAP** embedding.
- Plots a scatter colored by treatment.
- Runs a **Welch’s t‑test** vs **DMSO** and saves `results/demo/welch_summary_demo.csv`.

## Expected outputs
- One UMAP plot displayed in the notebook.
- Files written under `results/demo/`.

## Troubleshooting
- **`ModuleNotFoundError`** → make sure the virtual environment is active and `pip install -r requirements.txt` ran successfully.
- **`umap-learn` errors** → try `pip install umap-learn==0.5.3`.
- **Notebook doesn’t open** → install Jupyter with `python -m pip install jupyter` and relaunch.
- **No outputs in `results/demo/`** → check you have write permissions in the repo folder.

> For real experiments, use the scripts under `src/` and adjust paths (`root_dir`, `excel_folder`, `results_folder`) as described in the main README.
