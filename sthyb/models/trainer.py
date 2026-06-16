from sthyb.data.data_utils import gen_batch
from sthyb.models.base_model import build_model, model_save

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import time


def model_train(inputs, blocks, args, model_dir='./checkpoints/'):
    '''
    Train the STGCN/SAEA backbone.
    :param inputs: instance of class Dataset, data source for training.
    :param blocks: list, channel configs of st_conv blocks.
    :param args: parsed argparse namespace.
    :param model_dir: str, directory to save checkpoints.
    '''
    n, n_his = args.n_route, args.n_his
    Ks, Kt = args.ks, args.kt
    batch_size, epoch, opt = args.batch_size, args.epoch, args.opt

    x = tf.compat.v1.placeholder(tf.float32, [None, n_his + 1, n, 1], name='data_input')
    keep_prob = tf.compat.v1.placeholder(tf.float32, name='keep_prob')

    train_loss, pred = build_model(x, n, n_his, Ks, Kt, blocks, args.saea, keep_prob)
    copy_loss = tf.add_n(tf.get_collection('copy_loss'))
    test_loss = tf.add_n(tf.get_collection('test_loss'))

    # Learning-rate schedule: decay 0.7 every 5 epochs.
    global_steps = tf.Variable(0, trainable=False)
    len_train = inputs.get_len('train')
    epoch_step = len_train / batch_size if len_train % batch_size == 0 else int(len_train / batch_size) + 1
    lr = tf.compat.v1.train.exponential_decay(args.lr, global_steps, decay_steps=5 * epoch_step,
                                              decay_rate=0.7, staircase=True)
    step_op = tf.assign_add(global_steps, 1)
    with tf.control_dependencies([step_op]):
        if opt == 'RMSProp':
            train_op = tf.compat.v1.train.RMSPropOptimizer(lr).minimize(train_loss)
        elif opt == 'ADAM':
            train_op = tf.compat.v1.train.AdamOptimizer(lr).minimize(train_loss)
        else:
            raise ValueError(f'ERROR: optimizer "{opt}" is not defined.')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        x_train = inputs.get_data('train')
        # .copy() avoids modifying the original array in-place when step_p > 1
        x_train_ = x_train[:, 0:n_his + 1, :, :].copy()
        x_train_[:, n_his, :, :] = x_train[:, n_his + args.step_p - 1, :, :]
        for i in range(epoch):
            start_time = time.time()
            for j, x_batch in enumerate(gen_batch(x_train_, batch_size, dynamic_batch=True, shuffle=True)):
                sess.run(train_op, feed_dict={x: x_batch[:, 0:n_his + 1, :, :], keep_prob: 1.0})
                if j % 50 == 0:
                    loss_value = sess.run([train_loss, copy_loss, test_loss],
                                          feed_dict={x: x_batch[:, 0:n_his + 1, :, :], keep_prob: 1.0})
                    print(f'Epoch {i:2d}, Step {j:3d}: [{loss_value[0]:.3f}, {loss_value[1]:.3f}, {loss_value[2]:.3f}]')
            print(f'Epoch {i:2d} Training Time {time.time() - start_time:.3f}s')

            if (i + 1) % args.save == 0:
                model_save(sess, global_steps, f'STGCN-{args.saea}-{args.step_p}', save_path=model_dir)
    print('Training model finished!')
