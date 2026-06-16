"""
Graph WaveNet — TF1 (Wu et al. 2019). Faithful to nnzhan/Graph-WaveNet.

Architecture:
  blocks=4, layers=2 → 8 blocks, dilations [1,2,1,2,1,2,1,2], kernel_size=2
  GCN: order=2, support_len=3 (fwd+bwd+adp), concat [x,A¹x,A²x]×3 → MLP → dropout
  Skip: from gated output (before GCN), ALL timesteps, accumulated across blocks
  Residual: GCN_out + block_input, then BatchNorm
  Output: ReLU(skip) → FC(skip_ch→end_ch,ReLU) → FC(end_ch→1)

Loss: MAE (tf.reduce_mean(tf.abs(pred - true))) on normalised scale.
Training protocol (SAEA paper Section 4.5, unified across all models):
  RMSProp, lr=5e-4, LR decay 0.7/5 epochs, batch=50, epochs=300.

Paper defaults: res_ch=32, skip_ch=256, end_ch=512, dropout=0.3
SP_CRIME memory hint: --res_ch 16 --skip_ch 64 --end_ch 128 (order-2 GCN is memory-heavy)

Saves → ./predictions/y_{dataset}_gwavenet_{step_p}.npy
"""
import os, argparse, time
import numpy as np
import pandas as pd
from os.path import join as pjoin

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from sthyb.data.data_utils import data_gen_crime, gen_batch
from sthyb.utils.math_utils import evaluation

# ── args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--dataset',      type=str,   default='SP_CRIME')
parser.add_argument('--n_route',      type=int,   default=1445)
parser.add_argument('--n_his',        type=int,   default=30)
parser.add_argument('--n_pred',       type=int,   default=1)
parser.add_argument('--step_p',       type=int,   default=1)
parser.add_argument('--batch_size',   type=int,   default=50)
parser.add_argument('--epoch',        type=int,   default=300)
parser.add_argument('--save',         type=int,   default=100)
parser.add_argument('--lr',           type=float, default=5e-4)
parser.add_argument('--opt',          type=str,   default='RMSProp',
                    choices=['RMSProp', 'ADAM'])
parser.add_argument('--res_ch',       type=int,   default=32,
                    help='Paper default=32. Use 16 for SP_CRIME to avoid OOM.')
parser.add_argument('--skip_ch',      type=int,   default=256,
                    help='Paper default=256. Use 64 for SP_CRIME.')
parser.add_argument('--end_ch',       type=int,   default=512,
                    help='Paper default=512. Use 128 for SP_CRIME.')
parser.add_argument('--dropout',      type=float, default=0.3,
                    help='Dropout rate inside GCN (paper default=0.3).')
parser.add_argument('--emb_dim',      type=int,   default=10)
parser.add_argument('--n_train_days', type=int,   default=None)
parser.add_argument('--n_val_days',   type=int,   default=110)
parser.add_argument('--n_test_days',  type=int,   default=110)
parser.add_argument('--test_only',    action='store_true')
parser.add_argument('--loss',         type=str,   default='mae',
                    choices=['mae', 'mse'],
                    help='Prediction loss: mae (SAEA protocol, default) or mse')
args = parser.parse_args()

ds     = args.dataset
N      = args.n_route
n_his  = args.n_his
step_p = args.step_p
_ltag     = '' if args.loss == 'mae' else f'_{args.loss}'   # MAE keeps original dir
model_dir = f'./checkpoints/{ds}_gwavenet{_ltag}/'
pred_path = f'./predictions/y_{ds}_gwavenet{_ltag}_{step_p}.npy'
os.makedirs(model_dir, exist_ok=True)
os.makedirs('./predictions', exist_ok=True)

# ── adjacency ─────────────────────────────────────────────────────────────────
W = pd.read_csv(pjoin('./data', f'{ds}_W.csv'), header=None).values.astype(np.float32)
A_fwd = (W   / (W.sum(axis=1,   keepdims=True) + 1e-8)).astype(np.float32)
A_bwd = (W.T / (W.T.sum(axis=1, keepdims=True) + 1e-8)).astype(np.float32)

# ── data ──────────────────────────────────────────────────────────────────────
PeMS = data_gen_crime(
    pjoin('./data', f'{ds}_V.csv'),
    n_train_days=args.n_train_days, n_val_days=args.n_val_days,
    n_test_days=args.n_test_days,   n_route=N,
    n_frame=n_his + args.n_pred,
)
print(f'>> Mean={PeMS.mean:.4f}  Std={PeMS.std:.4f}')

# Mirror STGCN trainer: swap label slot to target step_p
x_train_raw = PeMS.get_data('train')
x_train_ = x_train_raw[:, :n_his+1, :, :].copy()
x_train_[:, n_his, :, :] = x_train_raw[:, n_his + step_p - 1, :, :]

x_test = PeMS.get_data('test')

# ── model ─────────────────────────────────────────────────────────────────────
def _nconv(x, adj):
    """Single-hop graph diffusion. x: (BT,N,C), adj: (N,N) → (BT,N,C)."""
    return tf.transpose(tf.tensordot(adj, x, [[1],[1]]), [1,0,2])


def _gcn(h, adj_fwd, adj_bwd, adj_adp, res_ch, dropout_rate, is_training):
    """
    Diffusion GCN, order=2, support_len=3 — identical to gcn.forward() in paper.
    Concatenates [h, A_fwd h, A²_fwd h, A_bwd h, A²_bwd h, A_adp h, A²_adp h]
    → (7*C_in channels) → linear MLP → dropout.  NO activation on output.
    h: (BT, N, C_in) → (BT, N, res_ch)
    """
    out = [h]
    for adj in [adj_fwd, adj_bwd, adj_adp]:
        x1 = _nconv(h, adj)    # A¹ h
        out.append(x1)
        x2 = _nconv(x1, adj)   # A² h
        out.append(x2)
    h_cat = tf.concat(out, axis=-1)                          # (BT, N, 7*C_in)
    h_out = tf.layers.dense(h_cat, res_ch, name='gcn_mlp')  # (BT, N, res_ch)
    h_out = tf.layers.dropout(h_out, rate=dropout_rate, training=is_training)
    return h_out


def wavenet_block(x, adj_fwd, adj_bwd, adj_adp,
                  dilation, res_ch, skip_ch, dropout_rate, bid, is_training):
    """
    One WaveNet block — exact order from gwnet.forward():
      1. residual = x
      2. x = tanh(filter_conv(x)) * sigmoid(gate_conv(x))   [gated]
      3. skip += skip_conv(x)                                [before GCN]
      4. x = gcn(x, supports)                                [order-2 diffusion]
      5. x = x + residual                                    [residual]
      6. x = bn(x)                                           [BatchNorm]
    x: (B,T,N,C_in) → x_out: (B,T,N,res_ch),  skip: (B,T,N,skip_ch)
    """
    C_in  = x.shape[-1].value
    pad_t = dilation

    with tf.variable_scope(f'block_{bid}'):
        residual = x
        B_dyn = tf.shape(x)[0]
        T_in  = tf.shape(x)[1]

        # ── causal dilated gated conv ────────────────────────────────────────
        x_pad = tf.pad(x, [[0,0],[pad_t,0],[0,0],[0,0]])
        T_pad = tf.shape(x_pad)[1]
        xr    = tf.reshape(tf.transpose(x_pad, [0,2,1,3]), [-1, T_pad, 1, C_in])

        Wf = tf.get_variable('Wf', [2,1,C_in,res_ch], initializer=tf.glorot_uniform_initializer())
        Wg = tf.get_variable('Wg', [2,1,C_in,res_ch], initializer=tf.glorot_uniform_initializer())
        bf = tf.get_variable('bf', [res_ch], initializer=tf.zeros_initializer())
        bg = tf.get_variable('bg', [res_ch], initializer=tf.zeros_initializer())

        f     = tf.nn.convolution(xr, Wf, padding='VALID', dilations=[dilation,1]) + bf
        g     = tf.nn.convolution(xr, Wg, padding='VALID', dilations=[dilation,1]) + bg
        T_out = tf.shape(f)[1]

        # (B*N, T, 1, res_ch) → (B, T, N, res_ch)
        x = tf.transpose(
            tf.reshape(tf.tanh(f) * tf.sigmoid(g), [B_dyn, N, T_out, res_ch]),
            [0,2,1,3])

        # ── skip: from gated output, ALL timesteps (before GCN) ─────────────
        Ws   = tf.get_variable('Ws', [res_ch, skip_ch], initializer=tf.glorot_uniform_initializer())
        skip = tf.tensordot(x, Ws, [[3],[0]])   # (B, T, N, skip_ch)

        # ── GCN order-2: all timesteps ───────────────────────────────────────
        x_bt = tf.reshape(x, [-1, N, res_ch])              # (B*T, N, res_ch)
        x_bt = _gcn(x_bt, adj_fwd, adj_bwd, adj_adp,
                    res_ch, dropout_rate, is_training)      # (B*T, N, res_ch)
        x    = tf.reshape(x_bt, [B_dyn, T_in, N, res_ch]) # (B, T, N, res_ch)

        # ── residual + BatchNorm ─────────────────────────────────────────────
        if C_in != res_ch:
            Wr       = tf.get_variable('Wr', [C_in,res_ch], initializer=tf.glorot_uniform_initializer())
            residual = tf.tensordot(residual, Wr, [[3],[0]])
        x = tf.layers.batch_normalization(x + residual, training=is_training, name='bn')

    return x, skip   # skip: (B, T, N, skip_ch)


def build_gwavenet(x_ph, adj_fwd_tf, adj_bwd_tf, is_training):
    # Adaptive adjacency — computed once per forward pass (as in paper)
    E1      = tf.get_variable('E1', [N, args.emb_dim], initializer=tf.glorot_uniform_initializer())
    E2      = tf.get_variable('E2', [args.emb_dim, N], initializer=tf.glorot_uniform_initializer())
    adj_adp = tf.nn.softmax(tf.nn.relu(tf.matmul(E1, E2)), axis=-1)

    y_true = x_ph[:, n_his, :, 0]   # (B, N)
    x_in   = x_ph[:, :n_his, :, :]  # (B, n_his, N, 1)

    # start_conv: in_dim=1 → res_ch  (paper: start_conv is a 1×1 conv)
    with tf.variable_scope('start_conv'):
        W0 = tf.get_variable('W0', [1, args.res_ch], initializer=tf.glorot_uniform_initializer())
        x  = tf.tensordot(x_in, W0, [[3],[0]])   # (B, n_his, N, res_ch)

    skip_total = 0   # paper initialises skip=0 then accumulates
    for bid, d in enumerate([1,2,1,2,1,2,1,2]):
        x, skip = wavenet_block(x, adj_fwd_tf, adj_bwd_tf, adj_adp,
                                d, args.res_ch, args.skip_ch,
                                args.dropout, bid, is_training)
        skip_total = skip_total + skip

    # Output head — mirrors end_conv_1 / end_conv_2 in paper
    with tf.variable_scope('output'):
        h      = tf.nn.relu(skip_total[:, -1, :, :])                           # (B, N, skip_ch)
        h      = tf.layers.dense(h, args.end_ch, activation=tf.nn.relu, name='fc1')
        y_pred = tf.squeeze(tf.layers.dense(h, 1, name='fc2'), axis=-1)        # (B, N)

    train_loss = (tf.reduce_mean(tf.square(y_pred - y_true)) if args.loss == 'mse'
                  else tf.reduce_mean(tf.abs(y_pred - y_true)))
    copy_loss  = tf.reduce_mean(tf.abs(x_ph[:, n_his-1, :, 0] - y_true))
    test_loss  = train_loss
    return y_pred, train_loss, copy_loss, test_loss


# ── build TF graph ────────────────────────────────────────────────────────────
tf.reset_default_graph()
x_ph        = tf.placeholder(tf.float32, [None, n_his+1, N, 1], name='data_input')
is_training = tf.placeholder(tf.bool, name='is_training')
adj_fwd_t   = tf.constant(A_fwd)
adj_bwd_t   = tf.constant(A_bwd)

y_pred, train_loss, copy_loss, test_loss = build_gwavenet(
    x_ph, adj_fwd_t, adj_bwd_t, is_training)

# ── LR decay — identical to STGCN trainer (SAEA paper Section 4.5) ────────────
len_train  = x_train_.shape[0]
epoch_step = int(np.ceil(len_train / args.batch_size))
global_steps = tf.Variable(0, trainable=False)
lr = tf.train.exponential_decay(args.lr, global_steps,
                                 decay_steps=5 * epoch_step,
                                 decay_rate=0.7, staircase=True)
step_op    = tf.assign_add(global_steps, 1)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)   # BN moving stats
with tf.control_dependencies(update_ops + [step_op]):
    if args.opt == 'RMSProp':
        train_op = tf.train.RMSPropOptimizer(lr).minimize(train_loss)
    else:
        train_op = tf.train.AdamOptimizer(lr).minimize(train_loss)

saver = tf.train.Saver(max_to_keep=3)

# ── train ─────────────────────────────────────────────────────────────────────
if not args.test_only:
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(args.epoch):
            start_time = time.time()
            for j, x_batch in enumerate(
                    gen_batch(x_train_, args.batch_size, dynamic_batch=True, shuffle=True)):
                sess.run(train_op,
                         feed_dict={'data_input:0': x_batch, 'is_training:0': True})
                if j % 50 == 0:
                    lv = sess.run([train_loss, copy_loss, test_loss],
                                  feed_dict={'data_input:0': x_batch, 'is_training:0': False})
                    print(f'Epoch {i:2d}, Step {j:3d}: '
                          f'[{lv[0]:.3f}, {lv[1]:.3f}, {lv[2]:.3f}]')
            print(f'Epoch {i:2d} Training Time {time.time()-start_time:.3f}s')
            if (i + 1) % args.save == 0:
                saver.save(sess, pjoin(model_dir, f'GWaveNet-{step_p}'),
                           global_step=global_steps)
    print('Training model finished!')

# ── test ──────────────────────────────────────────────────────────────────────
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint(model_dir))
    preds = []
    for batch in gen_batch(x_test, args.batch_size, dynamic_batch=True):
        p = sess.run(y_pred,
                     feed_dict={'data_input:0': batch, 'is_training:0': False})
        preds.append(p)

pred_arr = np.concatenate(preds, axis=0)[:, :, np.newaxis]   # (W, N, 1)
np.save(pred_path, pred_arr)

gt  = x_test[:len(pred_arr), n_his + step_p - 1, :, :]
evl = evaluation(gt, pred_arr, PeMS.get_stats())
print(f'GWaveNet  MAPE={evl[0]:.3%}  MAE={evl[1]:.4f}  RMSE={evl[2]:.4f}')
print(f'Saved → {pred_path}')
