"""
prepare_ba_lesiones.py
----------------------
Concatena delitos_2023.xlsx + delitos_2024.xlsx, filtra pelo tipo "Lesiones",
cria grade de 3km×3km e salva no formato SAEA pronto para os modelos.

Projeção: EPSG:32720 (UTM zona 20S — Buenos Aires)

Saídas:
  raw_data/BA_LESIONES/BA_LESIONES.geo
  raw_data/BA_LESIONES/BA_LESIONES.rel
  raw_data/BA_LESIONES/BA_LESIONES.dyna
  raw_data/BA_LESIONES/config.json
  dataset/BA_LESIONES_V.csv
  dataset/BA_LESIONES_W.csv
  dataset/BA_LESIONES_W2.csv
  dataset/BA_LESIONES_mask.npy
  dataset/BA_LESIONES_mask2.npy
  dataset/BA_LESIONES.geo

Uso:
  python prepare_ba_lesiones.py
  python prepare_ba_lesiones.py --cell_km 3.0 --sigma2 5.0 --epsilon 0.5
"""

import argparse
import os
import sys
import json
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

# ── Caminhos ───────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
INPUT_2023   = "C:/Users/luisg/Downloads/delitos_2023.xlsx"
INPUT_2024   = "C:/Users/luisg/Downloads/delitos_2024.xlsx"
RAW_OUT_DIR  = os.path.join("C:/Users/luisg/Bigscity-LibCity-master/raw_data", "BA_LESIONES")
DATASET_DIR  = os.path.join(SCRIPT_DIR, "dataset")
DATASET_NAME = "BA_LESIONES"
TIPO_FILTER  = "Lesiones"

# Buenos Aires bounding box para validação de coordenadas
BA_LAT = (-35.0, -34.4)
BA_LON = (-58.7, -58.2)

# ── Parâmetros padrão ──────────────────────────────────────────────────────────
DEFAULT_CELL_KM      = 3.0
DEFAULT_SIGMA2       = 5.0
DEFAULT_EPSILON      = 0.5
DEFAULT_SIGMA2_DENSE = 10.0
DEFAULT_EPSILON_DENSE = 0.1


# ── Helpers ────────────────────────────────────────────────────────────────────

def normalize_coord(val, lo, hi):
    """Divide pelo expoente de 10 correto até o valor cair em [lo, hi]."""
    if val == 0:
        return np.nan
    for exp in range(0, 12):
        v = val / (10 ** exp)
        if lo <= v <= hi:
            return v
    return np.nan


def haversine_matrix(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    R = 6_371_000.0
    lats_r = np.radians(lats)
    lons_r = np.radians(lons)
    n = len(lats)
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        dlat = lats_r - lats_r[i]
        dlon = lons_r - lons_r[i]
        a = (np.sin(dlat / 2) ** 2
             + np.cos(lats_r[i]) * np.cos(lats_r) * np.sin(dlon / 2) ** 2)
        dist[i] = R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return dist


def build_weight_matrix(D: np.ndarray, sigma2: float, epsilon: float) -> np.ndarray:
    n = D.shape[0]
    D_s = D / 10_000.0
    W = np.exp(-(D_s ** 2) / sigma2)
    mask_no_self = 1.0 - np.eye(n)
    W = W * (W >= epsilon) * mask_no_self
    return W


def make_mask(W: np.ndarray) -> np.ndarray:
    mask = (W == 0).astype(np.float64)
    np.fill_diagonal(mask, 0.0)
    return mask


# ── Pipeline ───────────────────────────────────────────────────────────────────

def prepare(cell_km, sigma2, epsilon, sigma2_dense, epsilon_dense):

    os.makedirs(RAW_OUT_DIR, exist_ok=True)
    os.makedirs(DATASET_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"BA_LESIONES — grade {cell_km:.1f}km x {cell_km:.1f}km")
    print(f"  W  : sigma2={sigma2}, epsilon={epsilon}")
    print(f"  W2 : sigma2={sigma2_dense}, epsilon={epsilon_dense}")

    # ── 1. Leitura e concatenação ───────────────────────────────────────────────
    print("\n[1] Lendo xlsx 2023 + 2024...")
    frames = []
    for path in [INPUT_2023, INPUT_2024]:
        try:
            tmp = pd.read_excel(path)
            frames.append(tmp)
            print(f"  {os.path.basename(path)}: {len(tmp):,} linhas")
        except Exception as e:
            sys.exit(f"ERRO ao ler {path}: {e}")

    df = pd.concat(frames, ignore_index=True)
    print(f"  Total bruto: {len(df):,} linhas")

    # ── 2. Filtro por tipo ──────────────────────────────────────────────────────
    df = df[df['tipo'] == TIPO_FILTER].copy()
    print(f"  Após filtro '{TIPO_FILTER}': {len(df):,} linhas")

    # ── 3. Normalização de coordenadas ─────────────────────────────────────────
    print("\n[2] Normalizando coordenadas...")
    lat_raw = df['latitud'].astype(float)
    lon_raw = df['longitud'].astype(float)

    df['lat'] = lat_raw.apply(lambda v: normalize_coord(v, *BA_LAT))
    df['lon'] = lon_raw.apply(lambda v: normalize_coord(v, *BA_LON))

    n_invalid = df['lat'].isna().sum()
    df = df.dropna(subset=['lat', 'lon'])
    print(f"  Coordenadas inválidas/zero descartadas: {n_invalid:,}")
    print(f"  Registros com coordenada válida: {len(df):,}")

    df['date'] = pd.to_datetime(df['fecha']).dt.normalize()
    print(f"  Período: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"  Dias: {df['date'].nunique()}")

    # ── 4. Projeção UTM e grid ─────────────────────────────────────────────────
    print("\n[3] Projetando para UTM 20S e criando grid...")
    try:
        from pyproj import Transformer
        tr = Transformer.from_crs('EPSG:4326', 'EPSG:32720', always_xy=True)
    except ImportError:
        sys.exit("ERRO: instale pyproj  →  pip install pyproj")

    x_m, y_m = tr.transform(df['lon'].values, df['lat'].values)
    df['x_m'] = x_m
    df['y_m'] = y_m

    cell_m = cell_km * 1000.0
    x_min, y_min = x_m.min(), y_m.min()
    df['grid_x'] = ((df['x_m'] - x_min) / cell_m).astype(int)
    df['grid_y'] = ((df['y_m'] - y_min) / cell_m).astype(int)

    # ── 5. Centroides das células ───────────────────────────────────────────────
    cell_centers = (
        df.groupby(['grid_x', 'grid_y'])
        .agg(lat=('lat', 'mean'), lon=('lon', 'mean'))
        .reset_index()
        .sort_values(['grid_x', 'grid_y'])
        .reset_index(drop=True)
    )
    cell_centers['geo_id'] = cell_centers.index
    n_nodes = len(cell_centers)

    n_days = df['date'].nunique()
    daily_cells = df.groupby(['date', 'grid_x', 'grid_y']).size()
    sp = 1.0 - len(daily_cells) / (n_days * n_nodes)
    print(f"  Células: {n_nodes}  |  Dias: {n_days}  |  Esparsidade: {sp*100:.1f}%")
    print(f"  Extensão: {(x_m.max()-x_min)/1000:.1f} km x {(y_m.max()-y_min)/1000:.1f} km")

    # ── 6. .geo ─────────────────────────────────────────────────────────────────
    print("\n[4] Gerando .geo...")
    geo_df = pd.DataFrame({
        'geo_id':      cell_centers['geo_id'].astype(int),
        'type':        'Point',
        'coordinates': ('[' + cell_centers['lon'].round(6).astype(str)
                        + ',' + cell_centers['lat'].round(6).astype(str) + ']'),
    })
    for dst in [os.path.join(RAW_OUT_DIR, f"{DATASET_NAME}.geo"),
                os.path.join(DATASET_DIR,  f"{DATASET_NAME}.geo")]:
        geo_df.to_csv(dst, index=False)
        print(f"  → {dst}")

    # ── 7. .rel ─────────────────────────────────────────────────────────────────
    print("\n[5] Calculando distâncias (Haversine) e .rel...")
    lats = cell_centers['lat'].values
    lons = cell_centers['lon'].values
    D = haversine_matrix(lats, lons)

    i_idx, j_idx = np.where(~np.eye(n_nodes, dtype=bool))
    rel_df = pd.DataFrame({
        'rel_id':         np.arange(len(i_idx)),
        'type':           'geo',
        'origin_id':      cell_centers['geo_id'].values[i_idx],
        'destination_id': cell_centers['geo_id'].values[j_idx],
        'weight':         D[i_idx, j_idx],
    })
    rel_path = os.path.join(RAW_OUT_DIR, f"{DATASET_NAME}.rel")
    rel_df.to_csv(rel_path, index=False)
    print(f"  → {rel_path}  ({len(rel_df):,} pares)")

    # ── 8. .dyna ─────────────────────────────────────────────────────────────────
    print("\n[6] Gerando série temporal diária (.dyna)...")
    cell_map = {(r.grid_x, r.grid_y): int(r.geo_id)
                for r in cell_centers.itertuples()}

    daily_index = {
        (r.grid_x, r.grid_y, r.date): r.crime_count
        for r in df.groupby(['grid_x', 'grid_y', 'date'])
                   .size().reset_index(name='crime_count').itertuples()
    }

    dates = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='D')
    print(f"  Dias no período: {len(dates)}")

    records = []
    dyna_id = 0
    for date in dates:
        time_str = date.strftime('%Y-%m-%dT00:00:00Z')
        d_key = pd.Timestamp(date).normalize()
        for _, cell_row in cell_centers.iterrows():
            gx, gy = int(cell_row['grid_x']), int(cell_row['grid_y'])
            geo_id = int(cell_row['geo_id'])
            count  = float(daily_index.get((gx, gy, d_key), 0))
            records.append((dyna_id, 'state', time_str, geo_id, count))
            dyna_id += 1

    dyna_df = pd.DataFrame(records,
                           columns=['dyna_id', 'type', 'time', 'entity_id', 'crime_count'])
    dyna_path = os.path.join(RAW_OUT_DIR, f"{DATASET_NAME}.dyna")
    dyna_df.to_csv(dyna_path, index=False)
    total_dyna = int(dyna_df['crime_count'].sum())
    n_raw = len(df)
    print(f"  → {dyna_path}  ({len(dyna_df):,} linhas)")
    print(f"  Total crimes .dyna: {total_dyna:,}  (original: {n_raw:,}  diff: {n_raw - total_dyna:,})")

    # ── 9. config.json ──────────────────────────────────────────────────────────
    config = {
        'geo':  {'including_types': ['Point'], 'Point': {}},
        'rel':  {'including_types': ['geo'], 'geo': {'weight': 'num'}},
        'dyna': {'including_types': ['state'],
                 'state': {'entity_id': 'geo_id', 'crime_count': 'num'}},
        'info': {
            'data_col':                ['crime_count'],
            'weight_col':              'weight',
            'data_files':              [DATASET_NAME],
            'geo_file':                DATASET_NAME,
            'rel_file':                DATASET_NAME,
            'output_dim':              1,
            'time_intervals':          86400,
            'init_weight_inf_or_zero': 'inf',
            'set_weight_link_or_dist': 'dist',
            'calculate_weight_adj':    True,
            'weight_adj_epsilon':      0.1,
            'cache_dataset':           True,
            'scaler':                  'log',
            'num_workers':             0,
        },
    }
    cfg_path = os.path.join(RAW_OUT_DIR, 'config.json')
    with open(cfg_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    print(f"  → {cfg_path}")

    # ── 10. V.csv, W.csv, W2.csv, mask.npy ─────────────────────────────────────
    print("\n[7] Gerando dataset/ (V, W, W2, mask)...")

    V = dyna_df.pivot(index='time', columns='entity_id', values='crime_count').values
    v_path = os.path.join(DATASET_DIR, f"{DATASET_NAME}_V.csv")
    pd.DataFrame(V).to_csv(v_path, header=False, index=False)
    print(f"  V shape: {V.shape}  → {v_path}")

    W  = build_weight_matrix(D, sigma2,       epsilon)
    W2 = build_weight_matrix(D, sigma2_dense, epsilon_dense)
    n_e  = int((W  > 0).sum())
    n_e2 = int((W2 > 0).sum())
    print(f"  W  arestas: {n_e:,}  ({n_e/(n_nodes*(n_nodes-1))*100:.1f}% off-diagonal)")
    print(f"  W2 arestas: {n_e2:,}  ({n_e2/(n_nodes*(n_nodes-1))*100:.1f}% off-diagonal)")

    pd.DataFrame(W).to_csv(os.path.join(DATASET_DIR, f"{DATASET_NAME}_W.csv"),
                           header=False, index=False)
    pd.DataFrame(W2).to_csv(os.path.join(DATASET_DIR, f"{DATASET_NAME}_W2.csv"),
                            header=False, index=False)
    np.save(os.path.join(DATASET_DIR, f"{DATASET_NAME}_mask.npy"),  make_mask(W))
    np.save(os.path.join(DATASET_DIR, f"{DATASET_NAME}_mask2.npy"), make_mask(W2))
    print(f"  → W.csv, W2.csv, mask.npy, mask2.npy")

    # ── 11. Resumo ──────────────────────────────────────────────────────────────
    T_total = V.shape[0]
    n_val = n_test = 110
    n_train = T_total - n_val - n_test
    crimes_flat = V.flatten()
    print(f"\n{'='*60}")
    print(f"BA_LESIONES — resumo final")
    print(f"  Nós (células {cell_km:.0f}km²): {n_nodes}")
    print(f"  Dias:                      {T_total}")
    print(f"  Zeros:                     {(crimes_flat == 0).mean()*100:.2f}%")
    print(f"  Máx crimes/dia/célula:     {int(crimes_flat.max())}")
    print(f"  Média:                     {crimes_flat.mean():.5f}")
    print(f"  Percentil 95:              {np.percentile(crimes_flat, 95):.1f}")
    print(f"  Partição sugerida (n_val=110, n_test=110):")
    print(f"    treino={n_train}  val={n_val}  teste={n_test}")
    print(f"\n  Treino do backbone:")
    print(f"    python scripts/train_stgcn.py --dataset {DATASET_NAME} --n_route {n_nodes}")
    print('='*60)
    print('Concluído.')


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cell_km',       type=float, default=DEFAULT_CELL_KM)
    parser.add_argument('--sigma2',        type=float, default=DEFAULT_SIGMA2)
    parser.add_argument('--epsilon',       type=float, default=DEFAULT_EPSILON)
    parser.add_argument('--sigma2_dense',  type=float, default=DEFAULT_SIGMA2_DENSE)
    parser.add_argument('--epsilon_dense', type=float, default=DEFAULT_EPSILON_DENSE)
    args = parser.parse_args()
    prepare(args.cell_km, args.sigma2, args.epsilon,
            args.sigma2_dense, args.epsilon_dense)
