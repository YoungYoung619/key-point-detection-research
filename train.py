"""
Copyright (c) College of Mechatronics and Control Engineering, Shenzhen University.
All rights reserved.

Description :


Author：Team Li
"""
import tensorflow as tf

from model.redundant_point_net import redudant_point_network

from dataset import dataset_factory
from utils.data_pileline_utils import prepare_data_train

if __name__ == '__main__':
    # redudant_point_network()

    with tf.device('/cpu:0'):
        dataset = dataset_factory.get_dataset(
            'pascalvoc_2012', 'train', './dataset/voc2012_tfrecord/')

        img, labels, bboxes = prepare_data_train(dataset, num_readers=4, batch_size=10, shuffle=True)
        batch_img, batch_labels, batch_bboxes = tf.train.batch([img, labels, bboxes], batch_size=10, num_threads=4, capacity=50, dynamic_pad=True)
        pass
    pass

