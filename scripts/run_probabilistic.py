"""Probabilistic evaluation over all cities x backbones (sthyb/config.py).

Produces results/probabilistic/{als_master, diag_residuals, gw_dm_tests,
calibration, per_node}.csv — the source of the paper's main tables.
Run from the repo root. Fast check: SMOKE=1 python scripts/run_probabilistic.py
"""
from sthyb.eval.probabilistic import main

if __name__ == "__main__":
    main()
