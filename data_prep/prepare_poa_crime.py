"""
prepare_poa_crime.py
--------------------
Converte dados de ocorrências RS (SPJ) para o formato SAEA,
usando bairros de Porto Alegre como unidade espacial.

Unidade espacial: 94 bairros oficiais (CD2022)
Adjacência W  : contiguidade Queen ponderada por fronteira compartilhada
Adjacência W2 : kernel Gaussiano sobre distâncias entre centroides

Saídas:
  dataset/POA_CRIME_V.csv
  dataset/POA_CRIME_W.csv    ← contiguidade (fronteira compartilhada)
  dataset/POA_CRIME_W2.csv   ← Gaussiana sobre centroides
  dataset/POA_CRIME_mask.npy
  dataset/POA_CRIME_mask2.npy
  dataset/POA_CRIME.geo

Uso:
  python prepare_poa_crime.py
  python prepare_poa_crime.py --tipo FURTO   (filtrar por tipo de crime)
"""

import argparse, os, sys, json
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
CSV_2024     = "C:/Users/luisg/Downloads/SPJ_DADOS_ABERTOS_OCORRENCIAS_JAN_DEZ_2024 - Em 05.01.2026.csv"
CSV_2025     = "C:/Users/luisg/Downloads/SPJ_DADOS_ABERTOS_OCORRÊNCIAS_JAN_DEZ_2025 - Em 05.01.2026.csv"
SHP_PATH     = "C:/Users/luisg/Downloads/RS_bairros_CD2022/RS_bairros_CD2022.shp"
DATASET_DIR  = os.path.join(SCRIPT_DIR, "dataset")
DATASET_NAME = "POA_CRIME"

# Aliases de grafia no dado de crime → nome oficial do shapefile
BAIRRO_ALIAS = {
    'MENINO DEUS':              'MENINO-DEUS',
    'MENINO  DEUS':             'MENINO-DEUS',
    'CORONEL APARICIO BORGES':  'CORONEL APARÍCIO BORGES',
    'CORONEL APARICO BORGES':   'CORONEL APARÍCIO BORGES',
    'CORONEL APARICIO BOR':     'CORONEL APARÍCIO BORGES',
    'MONTSERRAT':               'MONTSERRAT',
    'MONT\ufffdsERRAT':         'MONTSERRAT',
}

DEFAULT_SIGMA2_DENSE  = 5.0
DEFAULT_EPSILON_DENSE = 0.5


# ── Helpers ────────────────────────────────────────────────────────────────────

def haversine_matrix(lats, lons):
    R = 6_371_000.0
    lr, lo = np.radians(lats), np.radians(lons)
    n = len(lats)
    D = np.zeros((n, n))
    for i in range(n):
        dlat = lr - lr[i]; dlon = lo - lo[i]
        a = np.sin(dlat/2)**2 + np.cos(lr[i])*np.cos(lr)*np.sin(dlon/2)**2
        D[i] = R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return D


def gaussian_weight(D, sigma2, epsilon):
    n = D.shape[0]
    W = np.exp(-(D/10_000.0)**2 / sigma2)
    W = W * (W >= epsilon) * (1.0 - np.eye(n))
    return W


def make_mask(W):
    M = (W == 0).astype(np.float64)
    np.fill_diagonal(M, 0.0)
    return M


def shared_border_matrix(gdf):
    """
    W_ij = comprimento de fronteira compartilhada entre polígonos i e j,
    normalizado pela raiz do produto dos perímetros (simetria garantida).
    Diagonal = 0.
    """
    n = len(gdf)
    W = np.zeros((n, n))
    geoms = gdf.geometry.values

    for i in range(n):
        for j in range(i+1, n):
            inter = geoms[i].intersection(geoms[j])
            if inter.is_empty:
                continue
            # só conta fronteira linear (não pontos isolados)
            if inter.geom_type in ('LineString', 'MultiLineString',
                                   'GeometryCollection'):
                try:
                    length = inter.length
                except Exception:
                    length = 0.0
            else:
                length = 0.0

            if length > 0:
                # normalização pelo comprimento médio dos perímetros
                norm = (geoms[i].length * geoms[j].length) ** 0.5
                w = length / norm if norm > 0 else 0.0
                W[i, j] = w
                W[j, i] = w

    return W


# ── Pipeline ───────────────────────────────────────────────────────────────────

def prepare(tipo_filter, sigma2_dense, epsilon_dense):
    import geopandas as gpd

    os.makedirs(DATASET_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"POA_CRIME  —  bairros como unidade espacial")
    if tipo_filter:
        print(f"  Filtro tipo: '{tipo_filter}'")
    print(f"  W  : contiguidade Queen (fronteira compartilhada)")
    print(f"  W2 : kernel Gaussiano centroides (sigma2={sigma2_dense}, eps={epsilon_dense})")

    # ── 1. Shapefile ────────────────────────────────────────────────────────────
    print("\n[1] Carregando shapefile...")
    shp = gpd.read_file(SHP_PATH)
    poa_shp = shp[shp['NM_MUN'].str.upper().str.contains('PORTO ALEGRE', na=False)].copy()
    poa_shp = poa_shp.reset_index(drop=True)
    poa_shp['geo_id'] = poa_shp.index
    poa_shp['NM_UPPER'] = poa_shp['NM_BAIRRO'].str.upper().str.strip()
    n_nodes = len(poa_shp)
    print(f"  Bairros POA: {n_nodes}")

    # ── 2. Dados de crime ───────────────────────────────────────────────────────
    print("\n[2] Lendo dados de crime 2024+2025...")
    frames = []
    for path in [CSV_2024, CSV_2025]:
        tmp = pd.read_csv(path, sep=None, engine='python',
                          encoding='latin1', usecols=range(13))
        frames.append(tmp)
    df = pd.concat(frames, ignore_index=True)
    df = df[df['Municipio Fato'] == 'PORTO ALEGRE'].copy()
    print(f"  Registros POA: {len(df):,}")

    if tipo_filter:
        mask = df['Tipo Fato'].str.upper().str.contains(tipo_filter.upper(), na=False)
        df   = df[mask].copy()
        print(f"  Após filtro '{tipo_filter}': {len(df):,}")

    # ── 3. Normalizar bairro ────────────────────────────────────────────────────
    df['bairro_up'] = (df['Bairro'].str.upper().str.strip()
                                   .replace(BAIRRO_ALIAS))

    # manter só bairros que existem no shapefile
    df = df[df['bairro_up'].isin(poa_shp['NM_UPPER'])].copy()
    print(f"  Registros com bairro válido: {len(df):,}")

    # ── 4. Data ─────────────────────────────────────────────────────────────────
    df['date'] = pd.to_datetime(df['Data Fato'], dayfirst=True,
                                errors='coerce').dt.normalize()
    df = df.dropna(subset=['date'])
    print(f"  Período: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"  Dias: {df['date'].nunique()}")

    # ── 5. Série temporal (T × N) ───────────────────────────────────────────────
    print("\n[3] Construindo série temporal...")
    bairro_id = dict(zip(poa_shp['NM_UPPER'], poa_shp['geo_id']))
    df['node_id'] = df['bairro_up'].map(bairro_id)

    dates    = pd.date_range(df['date'].min(), df['date'].max(), freq='D')
    T        = len(dates)
    date_idx = {d: i for i, d in enumerate(dates)}
    df['t']  = df['date'].map(date_idx)

    V = np.zeros((T, n_nodes), dtype=np.float32)
    counts = df.groupby(['t','node_id']).size()
    for (t, nid), cnt in counts.items():
        V[t, nid] = cnt

    sp = (V == 0).mean()
    print(f"  V shape: {V.shape}")
    print(f"  Esparsidade: {sp*100:.1f}%")
    print(f"  Crimes/dia: {df.groupby('date').size().mean():.0f}")
    print(f"  Máx crimes/dia/bairro: {int(V.max())}")

    pd.DataFrame(V).to_csv(
        os.path.join(DATASET_DIR, f"{DATASET_NAME}_V.csv"),
        header=False, index=False)

    # ── 6. W — contiguidade por fronteira compartilhada ─────────────────────────
    print("\n[4] Calculando W (contiguidade Queen — pode demorar ~1 min)...")
    poa_proj = poa_shp.to_crs(epsg=32722)   # UTM 22S para cálculo em metros
    W_cont = shared_border_matrix(poa_proj)
    n_edges = int((W_cont > 0).sum())
    print(f"  Pares contíguos: {n_edges//2}  "
          f"({n_edges/(n_nodes*(n_nodes-1))*100:.1f}% off-diag)")

    pd.DataFrame(W_cont).to_csv(
        os.path.join(DATASET_DIR, f"{DATASET_NAME}_W.csv"),
        header=False, index=False)
    np.save(os.path.join(DATASET_DIR, f"{DATASET_NAME}_mask.npy"),
            make_mask(W_cont))

    # ── 7. W2 — kernel Gaussiano sobre centroides ───────────────────────────────
    print("\n[5] Calculando W2 (Gaussiana sobre centroides)...")
    # compute centroids on projected CRS, then reproject to WGS84 for lat/lon
    centroids_proj = poa_shp.to_crs(epsg=32722).geometry.centroid
    import geopandas as gpd
    centroids_wgs  = gpd.GeoSeries(centroids_proj, crs='epsg:32722').to_crs(epsg=4326)
    lats = np.array([c.y for c in centroids_wgs])
    lons = np.array([c.x for c in centroids_wgs])
    D    = haversine_matrix(lats, lons)
    W2   = gaussian_weight(D, sigma2_dense, epsilon_dense)
    n_e2 = int((W2 > 0).sum())
    print(f"  W2 arestas: {n_e2}  ({n_e2/(n_nodes*(n_nodes-1))*100:.1f}% off-diag)")

    pd.DataFrame(W2).to_csv(
        os.path.join(DATASET_DIR, f"{DATASET_NAME}_W2.csv"),
        header=False, index=False)
    np.save(os.path.join(DATASET_DIR, f"{DATASET_NAME}_mask2.npy"),
            make_mask(W2))

    # ── 8. .geo ─────────────────────────────────────────────────────────────────
    print("\n[6] Gerando .geo...")
    geo_df = pd.DataFrame({
        'geo_id':      poa_shp['geo_id'].astype(int),
        'type':        'Point',
        'coordinates': [f'[{lo},{la}]' for lo, la in zip(lons.round(6), lats.round(6))],
    })
    geo_df.to_csv(os.path.join(DATASET_DIR, f"{DATASET_NAME}.geo"), index=False)
    print(f"  → dataset/{DATASET_NAME}.geo")

    # ── 9. Resumo ────────────────────────────────────────────────────────────────
    n_val = n_test = 110
    n_train = T - n_val - n_test
    print(f"\n{'='*60}")
    print(f"POA_CRIME — resumo")
    print(f"  Nós (bairros):  {n_nodes}")
    print(f"  Dias:           {T}")
    print(f"  Esparsidade:    {sp*100:.1f}%")
    print(f"  Partição: treino={n_train}  val={n_val}  teste={n_test}")
    print(f"\n  Treino do backbone:")
    print(f"    python scripts/train_stgcn.py --dataset {DATASET_NAME} --n_route {n_nodes}")
    print('='*60)
    print('Concluído.')


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--tipo',          type=str,   default=None,
                   help='Filtrar por substring no campo Tipo Fato (ex: FURTO)')
    p.add_argument('--sigma2_dense',  type=float, default=DEFAULT_SIGMA2_DENSE)
    p.add_argument('--epsilon_dense', type=float, default=DEFAULT_EPSILON_DENSE)
    args = p.parse_args()
    prepare(args.tipo, args.sigma2_dense, args.epsilon_dense)
