from sthyb.models.layers import *
from os.path import join as pjoin
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def build_model(inputs, n, n_his, Ks, Kt, blocks, saea, keep_prob):
    '''
    Build the STGCN backbone, optionally with the SAEA autocorrelation adjustment.
    :param inputs: placeholder.
    :param n_his: int, size of historical records for training.
    :param Ks: int, kernel size of spatial convolution.
    :param Kt: int, kernel size of temporal convolution.
    :param blocks: list, channel configs of st_conv blocks.
    :param saea: str, adjustment mode — 'none' (plain STGCN), 'sparse', 'structural'
                 or 'structural2' (structural penalty on the W2 graph).
    :param keep_prob: placeholder.
    '''
    mask = tf.compat.v1.get_collection('masked')[0]

    if saea in ('sparse', 'structural', 'structural2'):
        init = 0.1 * tf.math.tanh(tf.random.uniform(shape=[n, n], minval=-1, maxval=1, dtype=tf.float32))
        phi = tf.compat.v1.get_variable('phi', initializer=init, dtype=tf.float32)

    xx = inputs[:, 0:n_his, :, :]
    x = xx + 1e-11   # avoid exact zeros before the autocorrelation transform

    if saea != 'none':
        # position 0 = G_{t-H} - Φ·mean  (Algorithm 1, line 4)
        x_diff = x[:, 1:n_his, :, :] - tf.transpose(tf.tensordot(x[:, 0:n_his - 1, :, :], phi, [[2], [1]]), [0, 1, 3, 2])
        b_mean = tf.expand_dims(tf.math.reduce_mean(xx, axis=1), axis=1)
        b_adj = xx[:, 0:1, :, :] - tf.transpose(tf.tensordot(b_mean, phi, [[2], [1]]), [0, 1, 3, 2])
        x = tf.concat([b_adj, x_diff], 1)

    # Ko: kernel size of temporal convolution in the output layer.
    Ko = n_his
    for i, channels in enumerate(blocks):
        rate = 1 - keep_prob
        x = st_conv_block(x, Ks, Kt, channels, i, rate, act_func='GLU')
        Ko -= 2 * (Kt - 1)

    if Ko > 1:
        y = output_layer(x, Ko, 'output_layer')
        e = y
    else:
        raise ValueError(f'ERROR: kernel size Ko must be greater than 1, but received "{Ko}".')

    # alpha per variant — Table 2 of the paper
    alpha = 100 if saea == 'sparse' else 1000
    if saea == 'none':
        train_loss = tf.nn.l2_loss(y - inputs[:, n_his:n_his + 1, :, :])
        single_pred = y
    else:
        e_ = inputs[:, n_his:n_his + 1, :, :] - tf.transpose(tf.tensordot(inputs[:, n_his - 1:n_his, :, :], phi, [[2], [1]]), [0, 1, 3, 2])
        if saea == 'sparse':
            reg = tf.math.reduce_sum(tf.abs(phi))                       # L1 sparsity
        else:  # structural / structural2
            reg = tf.nn.l2_loss(tf.math.multiply(phi, mask))            # Eq. 19: ||M ⊙ Φ||_2²
        train_loss = tf.nn.l2_loss(e_ - e) + alpha * reg
        single_pred = e[:, :, :, :] + tf.transpose(tf.tensordot(inputs[:, n_his - 1:n_his, :, :], phi, [[2], [1]]), [0, 1, 3, 2])

    tf.add_to_collection(name='copy_loss', value=tf.nn.l2_loss(inputs[:, n_his - 1:n_his, :, :] - inputs[:, n_his:n_his + 1, :, :]))
    tf.add_to_collection(name='test_loss', value=tf.nn.l2_loss(e - inputs[:, n_his:n_his + 1, :, :]))
    single_pred = single_pred[:, 0, :, :]
    tf.add_to_collection(name='y_pred', value=single_pred)
    if saea != 'none':
        tf.add_to_collection(name='phi', value=phi)

    return train_loss, single_pred


def model_save(sess, global_steps, model_name, save_path='./checkpoints/'):
    '''Save the checkpoint of the trained model.'''
    saver = tf.train.Saver(max_to_keep=3)
    prefix_path = saver.save(sess, pjoin(save_path, model_name), global_step=global_steps)
    print(f'<< Saving model to {prefix_path} ...')
