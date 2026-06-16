"""Train the STGCN backbone (single model) or its SAEA variants on a crime dataset.

  python scripts/train_stgcn.py --dataset SP_CRIME --n_route 1445            # plain STGCN
  python scripts/train_stgcn.py --dataset SP_CRIME --n_route 1445 --saea sparse
  python scripts/train_stgcn.py --dataset SP_CRIME --n_route 1445 --saea structural

Checkpoints → ./checkpoints/<DATASET>/STGCN-<saea>-<step_p>-*; run from the repo root.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from os.path import join as pjoin
import argparse
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.Session(config=config)

from sthyb.utils.math_graph import scaled_laplacian, cheb_poly_approx
from sthyb.data.data_utils import data_gen_crime
from sthyb.models.trainer import model_train
from sthyb.models.tester import model_test

# ── arguments ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--dataset',   type=str, required=True,
                    help='Crime dataset name (SP_CRIME / POA_CRIME / BA_LESIONES)')
parser.add_argument('--n_route',    type=int, required=True, help='Number of grid cells (N)')
# network
parser.add_argument('--n_his',      type=int, default=7, help='Historical window length (days)')
parser.add_argument('--n_pred',     type=int, default=1, help='Prediction horizon (days)')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--epoch',      type=int, default=300)
parser.add_argument('--save',       type=int, default=100)
parser.add_argument('--ks',         type=int, default=3, help='Spatial kernel size (Chebyshev order)')
parser.add_argument('--kt',         type=int, default=2, help='Temporal kernel size')
parser.add_argument('--lr',         type=float, default=5e-4)
parser.add_argument('--opt',        type=str, default='RMSProp', choices=['RMSProp', 'ADAM'])
parser.add_argument('--inf_mode',   type=str, default='merge', choices=['merge', 'sep'])
parser.add_argument('--step_p',     type=int, default=1, help='Prediction step to evaluate (1..n_pred)')
parser.add_argument('--saea',       type=str, default='none',
                    choices=['none', 'sparse', 'structural', 'structural2'],
                    help="'none' = plain STGCN; sparse/structural(2) = SAEA adjustment")
parser.add_argument('--small_model', action='store_true',
                    help='Smaller channels [[1,16,32],[32,16,64]] to save GPU memory (large graphs)')
parser.add_argument('--model_dir',  type=str, default=None,
                    help='Checkpoint dir (default: ./checkpoints/<DATASET>)')
parser.add_argument('--test_only',  action='store_true', help='Skip training, only evaluate')
parser.add_argument('--n_train_days', type=int, default=None, help='Training days (default: rest)')
parser.add_argument('--n_val_days',   type=int, default=110, help='Validation days')
parser.add_argument('--n_test_days',  type=int, default=110, help='Test days')
args = parser.parse_args()
print(f'Training configs: {args}')

n, n_his, n_pred = args.n_route, args.n_his, args.n_pred
Ks, Kt = args.ks, args.kt
blocks = [[1, 16, 32], [32, 16, 64]] if args.small_model else [[1, 32, 64], [64, 32, 128]]

# ── graph kernel ──────────────────────────────────────────────────────────────
# structural2 uses the denser W2 graph (→ mask2); all others use W (→ mask).
ds = args.dataset
w_file = pjoin('./data', f'{ds}_W2.csv' if args.saea == 'structural2' else f'{ds}_W.csv')
# The crime CSVs already hold the gaussian-transformed weights — load directly,
# do NOT re-apply a kernel.
W = pd.read_csv(w_file, header=None).values
Lk = cheb_poly_approx(scaled_laplacian(W), Ks, n)
tf.add_to_collection(name='graph_kernel', value=tf.cast(tf.constant(Lk), tf.float32))

# ── data ──────────────────────────────────────────────────────────────────────
PeMS = data_gen_crime(pjoin('./data', f'{ds}_V.csv'),
                      n_train_days=args.n_train_days, n_val_days=args.n_val_days,
                      n_test_days=args.n_test_days, n_route=n, n_frame=n_his + n_pred)
print(f'>> Loading dataset with Mean: {PeMS.mean:.2f}, STD: {PeMS.std:.2f}')

# ── structural mask ───────────────────────────────────────────────────────────
mask_path = pjoin('./data', f'{ds}_mask.npy' if args.saea == 'structural' else f'{ds}_mask2.npy')
mask = np.load(mask_path)
tf.compat.v1.add_to_collection(name='masked', value=tf.cast(tf.constant(mask), tf.float32))

if __name__ == '__main__':
    model_dir = args.model_dir if args.model_dir else f'./checkpoints/{args.dataset}/'
    os.makedirs(model_dir, exist_ok=True)
    if not args.test_only:
        model_train(PeMS, blocks, args, model_dir=model_dir)
    model_test(PeMS, PeMS.get_len('test'), n_his, n_pred, args.inf_mode, args.step_p, args.saea,
               load_path=model_dir, dataset=args.dataset)
