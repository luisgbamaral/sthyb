"""Residual spatial-dependence diagnostics (Pesaran CD, hop correlogram with
permutation null band, Marchenko-Pastur eigenvalue check, error-correlation maps).

Produces results/probabilistic/spatial_diag.csv and figs/{correlogram,ecm}_*.png.
Run from the repo root, after the checkpoints are in place.
"""
from sthyb.eval.spatial_diag import main

if __name__ == "__main__":
    main()
