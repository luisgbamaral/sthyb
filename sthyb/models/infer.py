"""infer.py — load a trained backbone checkpoint and run frozen-weight inference.

Single source for the backbone catalogue (_MODEL_SPECS / _DEFAULT_SUBDIR): add a
backbone by registering it here.
"""
import os, sys
import numpy as np
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

_MODEL_SPECS = {
    'stgcn':            dict(prefix='STGCN-none-',        tensor=None,
                             ik='keep_prob:0',             feed_val=1.0),
    'stgcn_sparse':     dict(prefix='STGCN-sparse-',      tensor=None,
                             ik='keep_prob:0',             feed_val=1.0),
    'stgcn_structural': dict(prefix='STGCN-structural-',  tensor=None,
                             ik='keep_prob:0',             feed_val=1.0),
    'gwavenet':         dict(prefix='GWaveNet-',           tensor='output/Squeeze:0',
                             ik='is_training:0',           feed_val=False),
    'sthsl':            dict(prefix='STHSL-',              tensor='global_enc/Squeeze_1:0',
                             ik='is_training:0',           feed_val=False),
    # MSE-trained single models (same inference graph, different checkpoint dir)
    'gwavenet_mse':     dict(prefix='GWaveNet-',           tensor='output/Squeeze:0',
                             ik='is_training:0',           feed_val=False),
    'sthsl_mse':        dict(prefix='STHSL-',              tensor='global_enc/Squeeze_1:0',
                             ik='is_training:0',           feed_val=False),
}

_DEFAULT_SUBDIR = {
    'stgcn':            '{ds}',
    'stgcn_sparse':     '{ds}',
    'stgcn_structural': '{ds}',
    'gwavenet':         '{ds}_gwavenet',
    'sthsl':            '{ds}_sthsl',
    'gwavenet_mse':     '{ds}_gwavenet_mse',
    'sthsl_mse':        '{ds}_sthsl_mse',
}

def _find_ckpt(model_dir, model):
    from pathlib import Path
    prefix = _MODEL_SPECS[model]['prefix']
    d = Path(model_dir)
    def _step_of(p):                       # trailing global-step (numeric, not alpha!)
        try:
            return int(p.stem.split('-')[-1])
        except ValueError:
            return -1
    files = sorted(d.glob(f'{prefix}*.meta'), key=_step_of)   # highest step = final ckpt
    if files:
        return str(files[-1])[:-5]
    state = tf.train.get_checkpoint_state(str(d))
    if state:
        for c in state.all_model_checkpoint_paths:
            if os.path.basename(c).startswith(prefix):
                return c
    sys.exit(f"[ERROR] No checkpoint with prefix '{prefix}' in {model_dir}")

def _detect_n_his(ckpt):
    g = tf.Graph()
    with g.as_default():
        tf.train.import_meta_graph(f'{ckpt}.meta')
    dim1 = g.get_tensor_by_name('data_input:0').shape.as_list()[1]
    return None if dim1 is None else dim1 - 1

def infer_split(x_data, n_his, batch_size, ckpt, model):
    """Run frozen-weight inference → (n_windows, N) in z-score space."""
    spec = _MODEL_SPECS[model]
    g = tf.Graph()
    with g.as_default():
        saver = tf.train.import_meta_graph(f'{ckpt}.meta')
    cfg = tf.ConfigProto(); cfg.gpu_options.allow_growth = True
    with tf.Session(graph=g, config=cfg) as sess:
        saver.restore(sess, ckpt)
        fetch = (g.get_collection('y_pred') if spec['tensor'] is None
                 else g.get_tensor_by_name(spec['tensor']))
        ik, fv = spec['ik'], spec['feed_val']
        preds = []
        for s in range(0, len(x_data), batch_size):
            b = x_data[s:s + batch_size, :n_his + 1, :, :]
            p = sess.run(fetch, feed_dict={'data_input:0': b, ik: fv})
            if isinstance(p, list): p = p[0]
            preds.append(p)
    r = np.concatenate(preds, axis=0)
    if r.ndim == 3: r = r[:, :, 0]
    return r
