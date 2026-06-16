"""
prepare_crime_data.py
---------------------
Converts the LibCity-format SP_CRIME dataset into the grid CSV format used by
the backbones (V = counts, W/W2 = gaussian-kernel adjacencies, masks for SAEA).

Outputs written to ./data/ (never touches the original LibCity files):
  <DATASET>_V.csv   — shape (T, N), rows=timesteps, cols=nodes, no header
  <DATASET>_W.csv   — shape (N, N) weighted adjacency matrix, no header
  <DATASET>_mask.npy  — binary (N,N), 1 where no edge (for saea=structural)
  <DATASET>_mask2.npy — binary (N,N), 1 where no edge (denser graph version)

Usage (POA_CRIME / BA_LESIONES have their own prepare_* scripts):
  python prepare_crime_data.py --dataset SP_CRIME
  python prepare_crime_data.py --dataset SP_CRIME --sigma2 5.0 --epsilon 0.5 --sigma2_dense 10.0 --epsilon_dense 0.1

Gaussian kernel applied to distances (metres):
  W[i,j] = exp(-d[i,j]^2 / (scale^2 * sigma2)) * (result >= epsilon) * no_self_loop_mask
where scale = 10000.

Default sigma2 values:
  SP_CRIME : sigma2=5.0  (connects pairs roughly within ~18 km)
"""

import argparse
import os
import numpy as np
import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────────
RAW_BASE = "C:/Users/luisg/Bigscity-LibCity-master/raw_data"   # raw LibCity source (edit for your machine)
OUT_DIR   = "./data"

# ── default sigma per dataset (tuned to distance distributions) ────────────────
SIGMA2_DEFAULTS = {
    "SP_CRIME": (5.0, 0.5, 10.0, 0.1),   # sigma2, eps, sigma2_dense, eps_dense
}


def weight_matrix_from_distances(D: np.ndarray, sigma2: float, epsilon: float) -> np.ndarray:
    """
    Apply a Gaussian kernel to a pairwise distance matrix D (in metres).
    Identical normalisation to utils/math_graph.py::weight_matrix():
      1. scale distances by /10000
      2. W = exp(-d^2 / sigma2) * (W >= epsilon) * no_self_loop
    Returns an (N, N) weighted adjacency matrix.
    """
    n = D.shape[0]
    D_scaled = D / 10_000.0
    W = np.exp(-(D_scaled ** 2) / sigma2)
    # zero out self-loops and sub-threshold entries
    mask_no_self = np.ones((n, n)) - np.eye(n)
    W = W * (W >= epsilon) * mask_no_self
    return W


def build_distance_matrix(rel: pd.DataFrame, n_nodes: int) -> np.ndarray:
    """
    Build a dense (N, N) distance matrix from a LibCity .rel file.
    Missing pairs (if any) are filled with the maximum observed distance.
    """
    D = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for _, row in rel.iterrows():
        i, j, w = int(row["origin_id"]), int(row["destination_id"]), float(row["weight"])
        D[i, j] = w
        D[j, i] = w   # symmetrise
    # fill missing off-diagonal zeros with max distance (so they get weight ~0)
    max_d = D.max()
    off_diag_zero = (D == 0) & (np.eye(n_nodes) == 0)
    D[off_diag_zero] = max_d
    return D


def build_value_matrix(dyna: pd.DataFrame, n_nodes: int) -> np.ndarray:
    """
    Pivot the LibCity .dyna file into a (T, N) matrix.
    Rows = chronological timesteps, columns = node ids.
    """
    # sort by time then entity_id to guarantee consistent ordering
    dyna = dyna.sort_values(["time", "entity_id"]).reset_index(drop=True)
    times = dyna["time"].unique()
    T = len(times)
    V = dyna["crime_count"].values.reshape(T, n_nodes)
    return V.astype(np.float64)


def make_mask(W: np.ndarray) -> np.ndarray:
    """
    Structural mask: 1 where there is NO edge (i.e. W[i,j]==0), 0 elsewhere.
    Diagonal stays 0 (no self-penalty).
    This is used in saea=structural regularisation:
      reg = sum(|phi| * mask)  →  penalises phi entries for non-adjacent pairs.
    """
    mask = (W == 0).astype(np.float64)
    np.fill_diagonal(mask, 0.0)
    return mask


def prepare(dataset: str, sigma2: float, epsilon: float,
            sigma2_dense: float, epsilon_dense: float) -> None:
    raw_dir = os.path.join(RAW_BASE, dataset)
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Processing: {dataset}")
    print(f"  W params  : sigma2={sigma2}, epsilon={epsilon}")
    print(f"  W2 params : sigma2={sigma2_dense}, epsilon={epsilon_dense}")

    # ── load source files ──────────────────────────────────────────────────────
    geo  = pd.read_csv(os.path.join(raw_dir, f"{dataset}.geo"))
    dyna = pd.read_csv(os.path.join(raw_dir, f"{dataset}.dyna"))
    rel  = pd.read_csv(os.path.join(raw_dir, f"{dataset}.rel"))

    n_nodes = len(geo)
    print(f"  Nodes     : {n_nodes}")
    print(f"  Timesteps : {dyna['time'].nunique()}")

    # ── build value matrix V (T × N) ──────────────────────────────────────────
    V = build_value_matrix(dyna, n_nodes)
    print(f"  V shape   : {V.shape}  (min={V.min():.2f}, max={V.max():.2f}, "
          f"mean={V.mean():.4f})")

    v_path = os.path.join(OUT_DIR, f"{dataset}_V.csv")
    pd.DataFrame(V).to_csv(v_path, header=False, index=False)
    print(f"  Saved V → {v_path}")

    # ── build distance matrix & adjacency matrices ────────────────────────────
    D = build_distance_matrix(rel, n_nodes)

    W  = weight_matrix_from_distances(D, sigma2,       epsilon)
    W2 = weight_matrix_from_distances(D, sigma2_dense, epsilon_dense)

    n_edges  = (W  > 0).sum()
    n_edges2 = (W2 > 0).sum()
    print(f"  W  edges  : {n_edges}  ({n_edges / (n_nodes*(n_nodes-1))*100:.1f}% of off-diagonal)")
    print(f"  W2 edges  : {n_edges2} ({n_edges2 / (n_nodes*(n_nodes-1))*100:.1f}% of off-diagonal)")

    w_path  = os.path.join(OUT_DIR, f"{dataset}_W.csv")
    w2_path = os.path.join(OUT_DIR, f"{dataset}_W2.csv")
    pd.DataFrame(W).to_csv(w_path,  header=False, index=False)
    pd.DataFrame(W2).to_csv(w2_path, header=False, index=False)
    print(f"  Saved W  → {w_path}")
    print(f"  Saved W2 → {w2_path}")

    # ── build masks ────────────────────────────────────────────────────────────
    mask  = make_mask(W)
    mask2 = make_mask(W2)
    np.save(os.path.join(OUT_DIR, f"{dataset}_mask.npy"),  mask)
    np.save(os.path.join(OUT_DIR, f"{dataset}_mask2.npy"), mask2)
    print(f"  Saved mask  → {OUT_DIR}/{dataset}_mask.npy  "
          f"(nonzero fraction: {mask.mean():.3f})")
    print(f"  Saved mask2 → {OUT_DIR}/{dataset}_mask2.npy "
          f"(nonzero fraction: {mask2.mean():.3f})")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert LibCity crime data to grid CSV format")
    parser.add_argument("--dataset", type=str, default="SP_CRIME",
                        choices=["SP_CRIME"],
                        help="Dataset name (default: SP_CRIME)")
    parser.add_argument("--sigma2",        type=float, default=None,
                        help="Gaussian kernel sigma² for W (default: dataset-specific)")
    parser.add_argument("--epsilon",       type=float, default=None,
                        help="Edge threshold for W (default: dataset-specific)")
    parser.add_argument("--sigma2_dense",  type=float, default=None,
                        help="Gaussian kernel sigma² for denser W2 (default: dataset-specific)")
    parser.add_argument("--epsilon_dense", type=float, default=None,
                        help="Edge threshold for denser W2 (default: dataset-specific)")
    args = parser.parse_args()

    defaults = SIGMA2_DEFAULTS[args.dataset]
    sigma2       = args.sigma2        if args.sigma2        is not None else defaults[0]
    epsilon      = args.epsilon       if args.epsilon       is not None else defaults[1]
    sigma2_dense = args.sigma2_dense  if args.sigma2_dense  is not None else defaults[2]
    epsilon_dense = args.epsilon_dense if args.epsilon_dense is not None else defaults[3]

    prepare(args.dataset, sigma2, epsilon, sigma2_dense, epsilon_dense)
