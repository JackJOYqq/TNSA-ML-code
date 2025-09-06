# EPOCH 1D Postprocessing and Machine Learning Tools

This repository contains Python scripts developed for analyzing and visualizing results from **EPOCH 1D** particle-in-cell (PIC) simulations. The focus is on **Target Normal Sheath Acceleration (TNSA)** and related ion acceleration studies, with tools for feature extraction, machine learning analysis, and visualization.

---

## Repository Structure

- **cross_run_ml_plus.py**  
  A machine learning workflow for analyzing multiple EPOCH 1D simulation runs.  
  - Performs feature extraction across different runs.  
  - Applies ML models (e.g., XGBoost, clustering) to classify and predict sheath/ion acceleration properties.  
  - Generates performance metrics and plots.

- **merge_features.py**  
  A utility script to merge extracted features from multiple runs into a single dataset.  
  - Combines CSV/feature files into one unified table.  
  - Prepares input for machine learning models.  
  - Useful for large-scale parameter sweeps.

- **runmax/single_run_viz.py**  
  Visualization tool for analyzing a single EPOCH 1D run.  
  - Plots sheath field, density evolution, and ion acceleration dynamics.  
  - Helps verify physical consistency of simulations before ML analysis.
