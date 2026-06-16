"""
STHSL — Spatial-Temporal Hypergraph Self-Supervised Learning
Lou et al., ICDE 2022. Adapted for SP_CRIME (N=1445 nodes, C=1, no spatial grid).

Architecture (adapted from original):
  Input: (B, N, T, 1)  where T=n_his=30
  Lift:  dense 1→latdim  →  (B, N, T, latdim)   [shared pos/neg weights]

  Local encoder:
    2× tem_cnn_local: conv2d(1, ks) 'SAME' + LeakyReLU + dropout + residual
      Note: original uses 4 parallel convs each collapsing cateNum→1, then
      concat+residual. With C=1 the concat (4×) vs residual (1×) dims mismatch,
      so adaptation is 1 conv with residual — semantically equivalent for C=1.
    eb_local: lc2 → transform_3d → (B, N, T, L)  [for InfoNCE]
    mean over T → dense(L→1) → out_local (B, N)

  Global encoder:
    Hypergraph_Infomax:
      Learnable adj (T, H, N) — equivalent to paper's (T, H, N*C) with C=1
      h_pos = hypergraph(x_pos, adj)   [pure einsum, no extra conv — paper's
      h_neg = hypergraph(x_neg, adj)    self.Conv is defined but never called]
      Readout: mean over N → sigmoid → score (B, 1, T, L)
      Bilinear discriminator: mean over T → logits (B, 1, 2N) → BCE loss
    4× tem_cnn_global (conv2d(1, ki), NO padding): e.g. T 30→22→15→8→1
    eb_global: g4 → transform_3d → (B, N, 1, L)  [for InfoNCE]
    squeeze T → dense(L→1) → out_global (B, N)   ← prediction output

  Losses (matching paper):
    infomax_loss  = BCE(disc logits, [1…1, 0…0])          × ir=1.0
    infoNCE_loss  = contrastive(eb_local, eb_global)        × cr=0.8
      [subsampled to nce_samples nodes because N=1445 makes (B,T,N,N) infeasible]
    mae_local     = mean|out_local  - y_true|
    mae_global    = mean|out_global - y_true|

Key adaptations vs original:
  • C=1  (single crime category, no row×col grid → spa_cnn_local dropped entirely)
  • InfoNCE node subsampling (--nce_samples, default 256) to avoid OOM with N=1445
  • Training: SAEA unified protocol (RMSProp, lr=5e-4, LR ×0.7/5 ep, batch=50, 300 ep)
  • SSL loss weights unchanged from paper (ir=1.0, cr=0.8, temp=0.05)

Saves:
  ./predictions/y_{dataset}_sthsl_{step_p}.npy   (W, N, 1)  z-scored
"""
import os, argparse, time
import numpy as np
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
parser.add_argument('--step_p',       type=int,   default=1)
parser.add_argument('--batch_size',   type=int,   default=50)
parser.add_argument('--epoch',        type=int,   default=300)
parser.add_argument('--save',         type=int,   default=100)
parser.add_argument('--lr',           type=float, default=5e-4)
parser.add_argument('--opt',          type=str,   default='RMSProp',
                    choices=['RMSProp', 'ADAM'])
# STHSL architecture (paper defaults)
parser.add_argument('--latdim',       type=int,   default=16,
                    help='Latent embedding dimension (paper default=16).')
parser.add_argument('--hyper_num',    type=int,   default=128,
                    help='Number of hyperedges (paper default=128).')
parser.add_argument('--kernel_size',  type=int,   default=3,
                    help='Local temporal CNN kernel size (paper default=3).')
parser.add_argument('--drop_rate_l',  type=float, default=0.2,
                    help='Dropout rate for local encoder (paper default=0.2).')
parser.add_argument('--drop_rate_g',  type=float, default=0.1,
                    help='Dropout rate for global encoder (paper default=0.1).')
parser.add_argument('--cr',           type=float, default=0.8,
                    help='InfoNCE contrastive loss weight (paper default=0.8).')
parser.add_argument('--ir',           type=float, default=1.0,
                    help='Infomax loss weight (paper default=1.0).')
parser.add_argument('--temp',         type=float, default=0.05,
                    help='InfoNCE temperature (paper default=0.05).')
parser.add_argument('--nce_samples',  type=int,   default=256,
                    help='Nodes subsampled per batch for InfoNCE (avoids OOM with N=1445).')
parser.add_argument('--n_train_days', type=int,   default=None)
parser.add_argument('--n_val_days',   type=int,   default=110)
parser.add_argument('--n_test_days',  type=int,   default=110)
parser.add_argument('--test_only',    action='store_true')
parser.add_argument('--loss',         type=str,   default='mae',
                    choices=['mae', 'mse'],
                    help='Prediction loss: mae (SAEA protocol, default) or mse (original STHSL)')
args = parser.parse_args()

ds     = args.dataset
N      = args.n_route
n_his  = args.n_his
step_p = args.step_p
L      = args.latdim
H      = args.hyper_num
Ns     = min(args.nce_samples, N)   # nodes subsampled for InfoNCE


def _auto_global_kernels(n_his):
    """
    Compute 'valid'-padded conv kernel sizes that reduce T from n_his to 1.
    Uses n_layers=4 for n_his>=20, 3 for n_his>=7, 2 otherwise.
    Distributes total reduction (n_his-1) as evenly as possible.

    Examples:
      n_his=30 → [9, 8, 8, 8]   T: 30→22→15→8→1
      n_his= 7 → [3, 3, 3]      T:  7→ 5→ 3→1
      n_his= 4 → [2, 3]         T:  4→ 3→1
    """
    if n_his >= 20:
        n_layers = 4
    elif n_his >= 4:
        n_layers = 3
    else:
        n_layers = 2
    total_red = n_his - 1
    base_r    = total_red // n_layers
    rem       = total_red  % n_layers
    # first `rem` layers get one extra reduction step
    kernels = [base_r + 2] * rem + [base_r + 1] * (n_layers - rem)
    # verify
    assert sum(k - 1 for k in kernels) == n_his - 1, \
        f"Kernel schedule {kernels} does not reduce T={n_his} to 1"
    return kernels


GLOBAL_KERNELS = _auto_global_kernels(n_his)
print(f'>> Global CNN kernels: {GLOBAL_KERNELS}  '
      f'(T={n_his}→{"→".join(str(n_his - sum(k-1 for k in GLOBAL_KERNELS[:i+1])) for i in range(len(GLOBAL_KERNELS)))})')

_ltag     = '' if args.loss == 'mae' else f'_{args.loss}'   # MAE keeps original dir
model_dir = f'./checkpoints/{ds}_sthsl{_ltag}/'
pred_path = f'./predictions/y_{ds}_sthsl{_ltag}_{step_p}.npy'
os.makedirs(model_dir, exist_ok=True)
os.makedirs('./predictions', exist_ok=True)

# ── data ──────────────────────────────────────────────────────────────────────
PeMS = data_gen_crime(
    pjoin('./data', f'{ds}_V.csv'),
    n_train_days=args.n_train_days,
    n_val_days=args.n_val_days,
    n_test_days=args.n_test_days,
    n_route=N,
    n_frame=n_his + 1,
)
print(f'>> Mean={PeMS.mean:.4f}  Std={PeMS.std:.4f}')
print(f'>> n_his={n_his}  step_p={step_p}')
print(f'>> latdim={L}  hyper_num={H}  nce_samples={Ns}  cr={args.cr}  ir={args.ir}')

x_train_raw = PeMS.get_data('train')
x_test      = PeMS.get_data('test')

# SAEA protocol: swap label at position n_his to the step_p-ahead target
x_train_ = x_train_raw[:, :n_his + 1, :, :].copy()
x_train_[:, n_his, :, :] = x_train_raw[:, n_his + step_p - 1, :, :]

# ── model helpers ──────────────────────────────────────────────────────────────

def leaky_relu(x, alpha=0.2):
    return tf.nn.leaky_relu(x, alpha=alpha)


def tem_cnn_local(x, name, ks, drop_rate, is_train):
    """
    Local temporal CNN block.
    Original: 4 parallel Conv3d each collapsing cateNum→1, concat → residual.
    Adaptation for C=1: single conv2d 'SAME' + dropout + leaky_relu(h + residual).
    Order faithfully matches original: conv → dropout → (cat+residual) → leaky_relu.
    (4-parallel would produce dim 4× residual dim with C=1 — shape mismatch)
    x: (B, N, T, L) → (B, N, T, L)
    """
    with tf.variable_scope(name):
        h = tf.layers.conv2d(x, L, (1, ks), padding='same',
                             activation=None,
                             kernel_initializer=tf.glorot_uniform_initializer())
        h = tf.layers.dropout(h, rate=drop_rate, training=is_train)
    return leaky_relu(h + x)


def hypergraph_agg(x, adj):
    """
    Hypergraph aggregation: node→hyperedge→node.
    Faithful to Hypergraph.forward() in original.
    Note: original defines self.Conv (Conv3d k=1) but never calls it — omitted.

    x:   (B, N, T, L)
    adj: (T, H, N)  learnable, passed as argument (shared for pos and neg)
    Returns: (B, N, T, L)
    """
    x_t    = tf.transpose(x, [0, 3, 2, 1])                        # (B, L, T, N)
    # node→hyperedge: sum_n adj[t,h,n] * x[b,l,t,n] → (B, L, T, H)
    h_edge = leaky_relu(tf.einsum('thn,bltn->blth', adj, x_t))
    adj_t  = tf.transpose(adj, [0, 2, 1])                          # (T, N, H)
    # hyperedge→node: sum_h adj_t[t,n,h] * h_edge[b,l,t,h] → (B, L, T, N)
    ret    = leaky_relu(tf.einsum('tnh,blth->bltn', adj_t, h_edge))
    return tf.transpose(ret, [0, 3, 2, 1])                         # (B, N, T, L)


def hypergraph_infomax(x_pos, x_neg, adj):
    """
    DGI-style Infomax on the hypergraph.
    x_pos, x_neg: (B, N, T, L)  — positive and spatially-shuffled negative
    adj:          (T, H, N)     — shared between pos and neg (same module in paper)
    Returns:
      h_pos : (B, N, T, L)   hypergraph output for positive
      logits: (B, 1, 2*N)    [pos_scores | neg_scores] for BCE loss
    """
    with tf.variable_scope('hyper_infomax'):
        # Both pos and neg use the same adj; no extra variables inside hypergraph_agg
        h_pos = hypergraph_agg(x_pos, adj)                         # (B, N, T, L)
        h_neg = hypergraph_agg(x_neg, adj)                         # (B, N, T, L)

        # Readout: mean over N → sigmoid → summary (B, 1, T, L)
        summary = tf.sigmoid(
            tf.reduce_mean(h_pos, axis=1, keepdims=True))           # (B, 1, T, L)
        # Expand to match h shape for bilinear
        score   = tf.tile(summary, [1, N, 1, 1])                    # (B, N, T, L)

        # Bilinear discriminator: f_k(h, score) = sum_l h_l * (score @ W)_l
        # Equivalent to nn.Bilinear(L, L, 1) with xavier init
        W_d = tf.get_variable('disc_W', [L, L],
                              initializer=tf.glorot_uniform_initializer())
        b_d = tf.get_variable('disc_b', [1],
                              initializer=tf.zeros_initializer())

        def bilinear_score(h, s):
            # h, s: (B, N, T, L)
            sW  = tf.tensordot(s, W_d, [[3], [1]])   # s @ W^T → (B, N, T, L)
            out = tf.reduce_sum(h * sW, axis=-1) + b_d  # (B, N, T)
            return tf.reduce_mean(out, axis=2)            # mean over T → (B, N)

        sc_pos = bilinear_score(h_pos, score)              # (B, N)
        sc_neg = bilinear_score(h_neg, score)              # (B, N)
        # cat [pos | neg] → (B, 2N), expand → (B, 1, 2N)
        logits = tf.expand_dims(
            tf.concat([sc_pos, sc_neg], axis=1), axis=1)  # (B, 1, 2N)

    return h_pos, logits


def tem_cnn_global(x, name, ki, drop_rate, is_train):
    """
    Global temporal CNN: conv2d(1, ki) NO padding + dropout + LeakyReLU.
    Order matches original: conv → dropout → leaky_relu.
    x: (B, N, T, L) → (B, N, T-ki+1, L)
    """
    with tf.variable_scope(name):
        h = tf.layers.conv2d(x, L, (1, ki), padding='valid',
                             activation=None,
                             kernel_initializer=tf.glorot_uniform_initializer())
        h = tf.layers.dropout(h, rate=drop_rate, training=is_train)
        h = leaky_relu(h)
    return h


def transform_3d(x, name, is_train):
    """
    Transform_3d: BatchNorm3d + Conv3d(k=1).
    Matches original: BN then 1×1 linear projection, shape preserved.
    x: (B, N, T, L) → (B, N, T, L)
    """
    with tf.variable_scope(name):
        x_bn  = tf.layers.batch_normalization(x, training=is_train)
        x_out = tf.layers.dense(x_bn, L,
                                kernel_initializer=tf.glorot_uniform_initializer())
    return x_out


def infoNCE_loss(q, k, temp, ns):
    """
    InfoNCE contrastive loss, subsampled to ns nodes to avoid (B,T,N,N) OOM.
    Original: q=eb_global (B,L,N,1,C), k=eb_local (B,L,N,T,C).
              q.expand_as(k) → both (B,L,N,T,C), then permute to (B,T,C,N,L).
    Adaptation: q=(B,N,1,L), k=(B,N,T,L), ns nodes sampled each call.

    q: (B, N, 1, L)  — global embedding
    k: (B, N, T, L)  — local embedding
    ns: int           — nodes to subsample (<=N)
    """
    # Subsample the same ns nodes from both q and k using a placeholder index
    node_idx = tf.random_shuffle(tf.range(N))[:ns]              # (ns,)
    q_s = tf.gather(q, node_idx, axis=1)                        # (B, ns, 1, L)
    k_s = tf.gather(k, node_idx, axis=1)                        # (B, ns, T, L)

    q_exp = tf.tile(q_s, [1, 1, n_his, 1])                     # (B, ns, T, L)
    # → (B, T, ns, L)
    q_t = tf.transpose(q_exp, [0, 2, 1, 3])
    k_t = tf.transpose(k_s,   [0, 2, 1, 3])
    q_n = tf.nn.l2_normalize(q_t, axis=-1)                     # (B, T, ns, L)
    k_n = tf.nn.l2_normalize(k_t, axis=-1)

    pos     = tf.exp(tf.reduce_sum(q_n * k_n, axis=-1) / temp) # (B, T, ns)
    neg_mat = tf.exp(
        tf.einsum('btnd,btmd->btnm', q_n, k_n) / temp)         # (B, T, ns, ns)
    neg_sum = tf.reduce_sum(neg_mat, axis=-1)                   # (B, T, ns)
    return tf.reduce_mean(-tf.math.log(pos / (neg_sum + 1e-8)))


# ── build TF graph ──────────────────────────────────────────────────────────
tf.reset_default_graph()

x_ph     = tf.placeholder(tf.float32, [None, n_his + 1, N, 1], name='data_input')
neg_ph   = tf.placeholder(tf.float32, [None, n_his,     N, 1], name='neg_input')
is_train = tf.placeholder(tf.bool,    name='is_training')

# Slice history and target
x_in   = tf.transpose(x_ph[:, :n_his, :, :], [0, 2, 1, 3])   # (B, N, T, 1)
y_true = x_ph[:, n_his, :, 0]                                  # (B, N)
x_neg  = tf.transpose(neg_ph, [0, 2, 1, 3])                    # (B, N, T, 1)

# ── Lift: 1 → latdim  (shared weights for pos and neg, like dimConv_in) ────
with tf.variable_scope('lift', reuse=tf.AUTO_REUSE):
    x_lift     = tf.layers.dense(x_in,  L,
                                 kernel_initializer=tf.glorot_uniform_initializer())
with tf.variable_scope('lift', reuse=True):
    x_neg_lift = tf.layers.dense(x_neg, L,
                                 kernel_initializer=tf.glorot_uniform_initializer())
# x_lift, x_neg_lift: (B, N, T, L)

# ── Local encoder ────────────────────────────────────────────────────────────
with tf.variable_scope('local_enc'):
    lc1      = tem_cnn_local(x_lift, 'tem1', args.kernel_size,
                              args.drop_rate_l, is_train)
    lc2      = tem_cnn_local(lc1,    'tem2', args.kernel_size,
                              args.drop_rate_l, is_train)
    eb_local = transform_3d(lc2, 'transform', is_train)         # (B, N, T, L)
    lc_pool  = tf.reduce_mean(lc2, axis=2)                      # (B, N, L) — mean over T
    out_local = tf.squeeze(
        tf.layers.dense(lc_pool, 1,
                        kernel_initializer=tf.glorot_uniform_initializer()),
        axis=-1)                                                 # (B, N)

# ── Hypergraph adjacency  (T, H, N) — learnable, passed to both pos/neg ────
with tf.variable_scope('hypergraph'):
    adj_hyp = tf.get_variable('adj', [n_his, H, N],
                              initializer=tf.random_normal_initializer(stddev=0.01))

# ── Global encoder ───────────────────────────────────────────────────────────
with tf.variable_scope('global_enc'):
    h_pos, infomax_logits = hypergraph_infomax(
        x_lift, x_neg_lift, adj_hyp)                            # h_pos: (B, N, T, L)

    g = h_pos
    for _ki, _k in enumerate(GLOBAL_KERNELS):
        g = tem_cnn_global(g, f'gtem{_ki+1}', _k,
                           args.drop_rate_g, is_train)
    # g shape: (B, N, 1, L) — T reduced to 1

    eb_global  = transform_3d(g, 'transform', is_train)        # (B, N, 1, L)
    g_squeezed = tf.squeeze(g, axis=2)                          # (B, N, L)
    out_global = tf.squeeze(
        tf.layers.dense(g_squeezed, 1,
                        kernel_initializer=tf.glorot_uniform_initializer()),
        axis=-1)                                                 # (B, N)

# ── Losses ───────────────────────────────────────────────────────────────────
B_dyn = tf.shape(x_ph)[0]

# 1. Infomax BCE
#    Labels: [1…1 (N pos) | 0…0 (N neg)] per (B, 1, 2N)
ones_     = tf.ones( tf.stack([B_dyn, 1, N]), dtype=tf.float32)
zeros_    = tf.zeros(tf.stack([B_dyn, 1, N]), dtype=tf.float32)
im_labels = tf.concat([ones_, zeros_], axis=2)                  # (B, 1, 2N)
im_loss   = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(
        labels=im_labels, logits=infomax_logits))

# 2. InfoNCE (global query vs local key, subsampled to Ns nodes)
nce_loss = infoNCE_loss(eb_global, eb_local, args.temp, Ns)

# 3. Prediction loss — MAE (SAEA protocol) or MSE (original STHSL); flag --loss
if args.loss == 'mse':
    mae_local  = tf.reduce_mean(tf.square(out_local  - y_true))
    mae_global = tf.reduce_mean(tf.square(out_global - y_true))
else:
    mae_local  = tf.reduce_mean(tf.abs(out_local  - y_true))
    mae_global = tf.reduce_mean(tf.abs(out_global - y_true))

# Total
train_loss = (args.ir  * im_loss
            + args.cr  * nce_loss
            + mae_local
            + mae_global)
copy_loss  = tf.reduce_mean(
    tf.abs(x_ph[:, n_his - 1, :, 0] - y_true))

# ── LR decay — SAEA Section 4.5 ──────────────────────────────────────────────
len_train    = x_train_.shape[0]
epoch_step   = int(np.ceil(len_train / args.batch_size))
global_steps = tf.Variable(0, trainable=False)
lr = tf.train.exponential_decay(
    args.lr, global_steps,
    decay_steps=5 * epoch_step, decay_rate=0.7, staircase=True)
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
        for epoch in range(args.epoch):
            t0 = time.time()
            for j, x_batch in enumerate(
                    gen_batch(x_train_, args.batch_size,
                              dynamic_batch=True, shuffle=True)):
                # Shuffle node order for Infomax negative (spatial corruption)
                idx     = np.random.permutation(N)
                neg_bat = x_batch[:, :n_his, idx, :]            # (B, T, N, 1)

                sess.run(train_op,
                         feed_dict={x_ph:     x_batch,
                                    neg_ph:   neg_bat,
                                    is_train: True})
                if j % 50 == 0:
                    tl, cl = sess.run(
                        [train_loss, copy_loss],
                        feed_dict={x_ph:     x_batch,
                                   neg_ph:   neg_bat,
                                   is_train: False})
                    print(f'Epoch {epoch:3d}  step {j:3d}'
                          f'  loss={tl:.4f}  copy={cl:.4f}')
            print(f'Epoch {epoch:3d}  time={time.time()-t0:.1f}s')
            if (epoch + 1) % args.save == 0:
                saver.save(sess, pjoin(model_dir, f'STHSL-{step_p}'),
                           global_step=global_steps)
    print('Training finished!')

# ── test ──────────────────────────────────────────────────────────────────────
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint(model_dir))
    preds = []
    for x_batch in gen_batch(x_test, args.batch_size, dynamic_batch=True):
        idx     = np.random.permutation(N)
        neg_bat = x_batch[:, :n_his, idx, :]
        p = sess.run(out_global,
                     feed_dict={x_ph:     x_batch,
                                neg_ph:   neg_bat,
                                is_train: False})               # (B, N)
        preds.append(p)

pred_arr = np.concatenate(preds, axis=0)[:len(x_test), :, np.newaxis]  # (W, N, 1)
gt       = x_test[:len(pred_arr), n_his + step_p - 1, :, :]
evl      = evaluation(gt, pred_arr, PeMS.get_stats())
print(f'STHSL  MAPE={evl[0]:.3%}  MAE={evl[1]:.4f}  RMSE={evl[2]:.4f}')
np.save(pred_path, pred_arr)
print(f'Saved → {pred_path}')
