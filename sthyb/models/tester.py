import os
from sthyb.data.data_utils import gen_batch
from sthyb.utils.math_utils import evaluation
from os.path import join as pjoin

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import time


def multi_pred2(sess, y_pred, seq, batch_size, n_his, dynamic_batch=True):
    '''Single-step prediction (no SAEA phi). Returns (pred_array [n_windows, N, 1], n_windows).'''
    pred_list = []
    for i in gen_batch(seq, min(batch_size, len(seq)), dynamic_batch=dynamic_batch):
        test_seq = np.copy(i[:, 0:n_his + 1, :, :])
        pred = sess.run(y_pred, feed_dict={'data_input:0': test_seq, 'keep_prob:0': 1.0})
        if isinstance(pred, list):
            pred = np.array(pred[0])
        pred_list.append(pred)
    pred_array = np.concatenate(pred_list, axis=0)
    return pred_array, pred_array.shape[0]


def multi_pred(sess, y_pred, phi, seq, batch_size, n_his, dynamic_batch=True):
    '''Single-step prediction with the SAEA phi tensor. Returns (pred_array, n_windows, phi).'''
    pred_list = []
    for i in gen_batch(seq, min(batch_size, len(seq)), dynamic_batch=dynamic_batch):
        test_seq = np.copy(i[:, 0:n_his + 1, :, :])
        pred, n_phi = sess.run([y_pred, phi], feed_dict={'data_input:0': test_seq, 'keep_prob:0': 1.0})
        if isinstance(pred, list):
            pred = np.array(pred[0])
        pred_list.append(pred)
    pred_array = np.concatenate(pred_list, axis=0)
    return pred_array, pred_array.shape[0], n_phi


def model_test(inputs, batch_size, n_his, n_pred, inf_mode, step_p, saea, load_path='./checkpoints/', dataset=''):
    '''
    Load a saved checkpoint and evaluate it on the test set; saves predictions to ./predictions/.
    :param inputs: instance of class Dataset, data source for test.
    :param step_p: int, which prediction step to evaluate.
    :param saea: str, SAEA variant of the checkpoint to load.
    :param load_path: str, directory of the saved model.
    '''
    start_time = time.time()
    # Search load_path first, then parent (fallback for legacy flat layout)
    from pathlib import Path
    state = tf.train.get_checkpoint_state(load_path)
    if state is None:
        parent = str(Path(load_path).parent)
        state = tf.train.get_checkpoint_state(parent)
        if state is None:
            raise FileNotFoundError(f"No checkpoint found in '{load_path}' or '{parent}'.")
        print(f"[INFO] Checkpoint found in parent dir: {parent}")
    # Find the checkpoint matching the current saea variant
    model_path = None
    prefix = f'STGCN-{saea}-'
    for ckpt in state.all_model_checkpoint_paths:
        if os.path.basename(ckpt).startswith(prefix):
            model_path = ckpt
    if model_path is None:
        model_path = state.model_checkpoint_path   # fallback: latest

    test_graph = tf.Graph()
    with test_graph.as_default():
        saver = tf.train.import_meta_graph(pjoin(f'{model_path}.meta'))

    with tf.Session(graph=test_graph) as test_sess:
        saver.restore(test_sess, model_path)
        print(f'>> Loading saved model from {model_path} ...')

        x_test, x_stats = inputs.get_data('test'), inputs.get_stats()
        pred = test_graph.get_collection('y_pred')

        os.makedirs('./predictions', exist_ok=True)
        if saea == 'none':
            y_test, len_test = multi_pred2(test_sess, pred, x_test, batch_size, n_his)
        else:
            phi = test_graph.get_collection('phi')
            y_test, len_test, n_phi = multi_pred(test_sess, pred, phi, x_test, batch_size, n_his)
            np.save(f'./predictions/phi_{dataset}_{saea}_{step_p}.npy', n_phi)
        np.save(f'./predictions/y_{dataset}_{saea}_{step_p}.npy', y_test)

        evl = evaluation(x_test[0:len_test, n_his + step_p - 1, :, :], y_test, x_stats)
        print(f'Time Step {step_p}: MAPE {evl[0]:7.3%}; MAE  {evl[1]:4.3f}; RMSE {evl[2]:6.3f}.')
        print(f'Model Test Time {time.time() - start_time:.3f}s')
    print('Testing model finished!')
