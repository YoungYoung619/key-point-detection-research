"""
Copyright (c) College of Mechatronics and Control Engineering, Shenzhen University.
All rights reserved.

Description :


Author：Team Li
"""
import tensorflow as tf
import numpy as np
import cv2

from datasets import dataset_factory

from utils.data_pileline_utils import *
from utils.vis_utils import visualize_boxes_and_labels_on_image_array


if __name__ == '__main__':
    with tf.device('/cpu:0'):
        dataset = dataset_factory.get_dataset(
            'pascalvoc_2012', 'train', './dataset/voc2012_tfrecord/')

        img, labels, bboxes = prepare_data_train(dataset, num_readers=4, batch_size=10, shuffle=True)



    with tf.Session() as sess:
        # start queue
        coord = tf.train.Coordinator()
        # start the queues #
        threads = tf.train.start_queue_runners(coord=coord)

        while(True):

            img_, labels_, bboxes_ = sess.run([img, labels, bboxes])

            img_ = img_.astype(np.uint8)
            bboxes_[:, 0] = np.int32(bboxes_[:, 0]*config.img_size[0])
            bboxes_[:, 1] = np.int32(bboxes_[:, 1] * config.img_size[1])
            bboxes_[:, 2] = np.int32(bboxes_[:, 2] * config.img_size[0])
            bboxes_[:, 3] = np.int32(bboxes_[:, 3] * config.img_size[1])

            labels_ = labels_
            scores = np.ones(shape=bboxes_.shape[0])
            img_ = visualize_boxes_and_labels_on_image_array(img_, bboxes_, labels_, scores, config.category_index, skip_scores=True)

            img_ = cv2.cvtColor(img_, cv2.COLOR_RGB2BGR)
            print(labels_)
            cv2.imshow('test', img_)
            cv2.waitKey()
            cv2.destroyAllWindows()
            pass

        # terminate the threads #
        coord.request_stop()
        coord.join(threads)