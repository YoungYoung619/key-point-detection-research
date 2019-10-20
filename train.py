"""
Copyright (c) College of Mechatronics and Control Engineering, Shenzhen University.
All rights reserved.

Description :


Author：Team Li
"""
import tensorflow as tf
import os
import config
from datasets import dataset_factory
from utils.data_pileline_utils import prepare_data_train
from model.redundant_point_net import redudant_point_network
from utils.train_utils import loss_for_batch
from utils.logging import logger

tf.app.flags.DEFINE_string(
    'model_name', 'redundant',
    'The name of the architecture to train.')

tf.app.flags.DEFINE_string(
    'checkpoint_dir', '',
    'The path to a checkpoint from which to fine-tune.')

tf.app.flags.DEFINE_string(
    'train_dir', './checkpoint/',
    'Directory where checkpoints are written to.')

tf.app.flags.DEFINE_string(
    'summary_dir', './summary/',
    'Directory where checkpoints are written to.')

tf.app.flags.DEFINE_integer(
    'batch_size', 1, 'The number of samples in each batch.')

tf.app.flags.DEFINE_float('learning_rate', 1e-3, 'Initial learning rate.')

tf.app.flags.DEFINE_integer(
    'f_log_step', 20,
    'The frequency with which logs are print.')

tf.app.flags.DEFINE_integer(
    'f_summary_step', 20,
    'The frequency with which the model is saved, in step.')

tf.app.flags.DEFINE_integer(
    'f_save_step', 2000,
    'The frequency with which summaries are saved, in step.')

tf.app.flags.DEFINE_integer(
    'training_step', None,
    'when training step bigger than training_step, training would stop')

FLAGS = tf.app.flags.FLAGS

global_step = tf.Variable(0, trainable=False, name='global_step')

def build_optimizer(loss, var_list=None):
    """ build total loss, and optimizer.
    Args:
        loss: a tensor represents the loss
        var_list: the variable need to be trained.
    Return:
        a train_ops
    """
    with tf.name_scope("optimize"):
        # learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step,
        #                                            2*70000 / FLAGS.batch_size,
        #                                            0.97, staircase=True)
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        train_ops = optimizer.minimize(loss, global_step=global_step, var_list=var_list)
        # train_ops = optimizer.minimize(loss, global_step=global_step)

        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(update_ops):
        #     optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        #     if var_list == None:
        #         aa = optimizer.compute_gradients(loss)
        #         train_ops = optimizer.minimize(loss, global_step=global_step)
        #     else:
        #         train_ops = optimizer.minimize(loss, global_step=global_step, var_list=var_list)

    return train_ops


if __name__ == '__main__':
    # redudant_point_network()
    with tf.device('/cpu:0'):
        dataset = dataset_factory.get_dataset(
            'pascalvoc_2012', 'train', './datasets/voc2012_tfrecord/')

        img, labels, bboxes = prepare_data_train(dataset, num_readers=4, batch_size=FLAGS.batch_size, shuffle=True)
        batch_img, batch_labels, batch_bboxes = tf.train.batch([img, labels-1, bboxes], batch_size=FLAGS.batch_size,
                                                               num_threads=4, capacity=FLAGS.batch_size*5, dynamic_pad=True)

    net = redudant_point_network(input=batch_img, is_training=True, n_class=config.n_class)
    hm_preds, emb_preds, offset_preds = net.get_output()

    loss = loss_for_batch(hm_preds, emb_preds, offset_preds, batch_bboxes, batch_labels)
    update_op = build_optimizer(loss)
    # merge_op = tf.summary.merge_all()

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
    init = tf.global_variables_initializer()
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        ## create a summary writer ##
        summary_dir = os.path.join(FLAGS.summary_dir)
        writer = tf.summary.FileWriter(FLAGS.summary_dir, sess.graph)

        if ckpt:
            logger.info('loading %s...' % str(ckpt.model_checkpoint_path))
            saver.restore(sess, ckpt.model_checkpoint_path)
            logger.info('Load checkpoint success...')
        else:
            sess.run(init)
            logger.info('TF variables init success...')

        # start queue
        coord = tf.train.Coordinator()
        # start the queues #
        threads = tf.train.start_queue_runners(coord=coord)

        sess.run(init)
        avg_loss = 0.
        while (True):
            op, loss_local, current_step = sess.run([update_op, loss, global_step])

            if FLAGS.f_log_step != None:
                ## caculate average loss ##
                step = current_step % FLAGS.f_log_step
                avg_loss = (avg_loss * step + loss_local) / (step + 1.)
                if current_step % FLAGS.f_log_step == FLAGS.f_log_step - 1:
                    ## print info ##
                    logger.info('Step%s loss:%s' % (str(current_step), str(avg_loss)))
                    avg_loss = 0.

            if FLAGS.f_summary_step != None:
                if current_step % FLAGS.f_summary_step == FLAGS.f_summary_step - 1:
                    ## summary ##
                    # writer.add_summary(mp, current_step)
                    pass

            if FLAGS.f_save_step != None:
                if current_step % FLAGS.f_save_step == FLAGS.f_save_step - 1:
                    ## save model ##
                    logger.info('Saving model...')
                    model_name = os.path.join(FLAGS.train_dir, FLAGS.model_name)
                    saver.save(sess, model_name, global_step=current_step)
                    logger.info('Save model sucess...')

            if FLAGS.training_step != None:
                if current_step >= FLAGS.training_step:
                    logger.info('Exit training...')
                    break

        # terminate the threads #
        coord.request_stop()
        coord.join(threads)