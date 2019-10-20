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
from model.redundant_point_net import redudant_point_network
from utils.train_utils import *

from utils.logging import logger

def vis_heat_map(heat_gts, labels):

    labels = set(labels)
    for i, index in enumerate(labels):
        cv2.imshow('top_left:'+config.category_index[index+1]['name']+'_'+str(i), heat_gts[0][:,:,index])
        # cv2.imshow('top_right:' + config.category_index[index + 1]['name']+'_'+str(i), heat_gts[1][:,:,index])
        # cv2.imshow('bottom_left:' + config.category_index[index + 1]['name']+'_'+str(i), heat_gts[2][:,:,index])
        # cv2.imshow('bottom_right:' + config.category_index[index + 1]['name']+'_'+str(i), heat_gts[3][:,:,index])
        # cv2.imshow('center:' + config.category_index[index + 1]['name']+'_'+str(i), heat_gts[4][:,:,index])

def vis_all__heat_map(heat_gts):
    top_left = np.max(heat_gts[0], axis=-1)
    cv2.imshow('top_left:', top_left)
    logger.info('n_pos_top_left:'+str(np.sum(np.int32(np.equal(top_left, 1)))))

    top_right = np.max(heat_gts[1], axis=-1)
    cv2.imshow('top_right:', top_right)
    logger.info('n_pos_top_right:' + str(np.sum(np.int32(np.equal(top_right, 1)))))

    bottom_left = np.max(heat_gts[2], axis=-1)
    cv2.imshow('bottom_left:', bottom_left)
    logger.info('n_pos_bottom_left:' + str(np.sum(np.int32(np.equal(bottom_left, 1)))))

    bottom_right = np.max(heat_gts[3], axis=-1)
    cv2.imshow('bottom_right:', bottom_right)
    logger.info('n_pos_bottom_right:' + str(np.sum(np.int32(np.equal(bottom_right, 1)))))

    center = np.max(heat_gts[4], axis=-1)
    cv2.imshow('center:', center)
    logger.info('n_pos_center:' + str(np.sum(np.int32(np.equal(center, 1)))))
    logger.info('---------------------------------------------')
    logger.info('---------------------------------------------')


if __name__ == '__main__':
    bs = 1
    with tf.device('/cpu:0'):
        dataset = dataset_factory.get_dataset(
            'pascalvoc_2012', 'train', './datasets/voc2012_tfrecord/')

        img, labels, bboxes = prepare_data_train(dataset, num_readers=4, batch_size=bs, shuffle=True)
        batch_img, batch_labels, batch_bboxes = tf.train.batch([img, labels - 1, bboxes], batch_size=bs,
                                                               num_threads=4, capacity=bs * 5, dynamic_pad=True)

    net = redudant_point_network(input=batch_img, is_training=True, n_class=config.n_class)
    hm_preds, emb_preds, offset_preds = net.get_output()
    hm_gts, emb_pair, offsets, points_pos_mask = encode_for_one_img(hm_preds, emb_preds, offset_preds, batch_bboxes[0], batch_labels[0])

    pos_hm_loss = hm_pos_loss_for_one_bbox(hm_preds, points_pos_mask, batch_bboxes[0], batch_labels[0], 0)
    neg_hm_loss = hm_neg_loss_for_one_img(hm_preds, hm_gts)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        # start queue
        coord = tf.train.Coordinator()
        # start the queues #
        threads = tf.train.start_queue_runners(coord=coord)
        while(True):

            img_, labels_, bboxes_, hm_gts_, points_pos_mask_, pos_hm_loss_, neg_hm_loss_ = sess.run([batch_img, batch_labels, batch_bboxes,
                                                                                        hm_gts, points_pos_mask, pos_hm_loss,
                                                                                        neg_hm_loss])
            logger.info('neg_hm_loss_:'+str(neg_hm_loss_))


            img_ = img_[0]
            labels_ = labels_[0]
            bboxes_ = bboxes_[0]

            vis_all__heat_map(hm_gts_)

            img_ = img_.astype(np.uint8)
            bboxes_[:, 0] = np.int32(bboxes_[:, 0]*config.img_size[0])
            bboxes_[:, 1] = np.int32(bboxes_[:, 1] * config.img_size[1])
            bboxes_[:, 2] = np.int32(bboxes_[:, 2] * config.img_size[0])
            bboxes_[:, 3] = np.int32(bboxes_[:, 3] * config.img_size[1])

            labels_ = labels_
            scores = np.ones(shape=bboxes_.shape[0])
            img_ = visualize_boxes_and_labels_on_image_array(img_, bboxes_, labels_+1, scores, config.category_index, skip_scores=True)

            img_ = cv2.cvtColor(img_, cv2.COLOR_RGB2BGR)
            cv2.imshow('test', img_)
            cv2.waitKey()
            cv2.destroyAllWindows()
            pass

        # terminate the threads #
        coord.request_stop()
        coord.join(threads)