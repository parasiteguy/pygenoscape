# Simulation validation (Figure 2)

This directory contains the script used to generate the simulation-based validation shown in Figure 2 of:

Davinack & Seaberg (2026)  
*pygenoscape: a Python framework for spatial interpolation of genetic distance landscapes*  
Bioinformatics Advances

## Overview

This validation demonstrates that pygenoscape accurately reconstructs known spatial genetic patterns without introducing interpolation artifacts.

Two scenarios are simulated:

1. **Isolation-by-distance (IBD)**  
   Pairwise genetic distances increase as a function of Euclidean geographic distance.

2. **Barrier to gene flow**  
   Genetic distances are increased between samples located on opposite sides of a spatial boundary.

The workflow mirrors the pygenoscape pipeline:
genetic distance -> PCoA -> spatial interpolation (RBF) -> surface visualization


## Files

- `simulation_validation.py`  
  Python script that generates both simulation scenarios and produces the figure.

- `pygenoscape_simulation_validation.png`  
  Output figure corresponding to Figure 2 in the manuscript.

## Requirements

The script requires the following Python packages:

- numpy
- matplotlib
- scipy

Install via:

```bash
pip install numpy matplotlib scipy

#run the script 
python simulation_validation.py

This will generate: 
pygenoscape_simulation_validation.png
