"""
Copyright (c) College of Mechatronics and Control Engineering, Shenzhen University.
All rights reserved.

Description :


Author：Team Li
"""
import tensorflow as tf

import config
from datasets import dataset_factory
from utils.data_pileline_utils import prepare_data_train
from model.redundant_point_net import redudant_point_network
from utils.train_utils import loss_for_batch

if __name__ == '__main__':
    # redudant_point_network()
    bs = 1
    with tf.device('/cpu:0'):
        dataset = dataset_factory.get_dataset(
            'pascalvoc_2012', 'train', './datasets/voc2012_tfrecord/')

        img, labels, bboxes = prepare_data_train(dataset, num_readers=4, batch_size=bs, shuffle=False)
        batch_img, batch_labels, batch_bboxes = tf.train.batch([img, labels-1, bboxes], batch_size=bs, num_threads=4, capacity=bs*10, dynamic_pad=True)

    net = redudant_point_network(input=batch_img, is_training=True, n_class=config.n_class)
    hm_preds, emb_preds, offset_preds = net.get_output()

    loss = loss_for_batch(hm_preds, emb_preds, offset_preds, batch_bboxes, batch_labels)

    init = tf.global_variables_initializer()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # start queue
        coord = tf.train.Coordinator()
        # start the queues #
        threads = tf.train.start_queue_runners(coord=coord)

        sess.run(init)
        while (True):
            l = sess.run(loss)
            print(l)
            pass

        # terminate the threads #
        coord.request_stop()
        coord.join(threads)