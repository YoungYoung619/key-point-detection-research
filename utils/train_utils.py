"""
Copyright (c) College of Mechatronics and Control Engineering, Shenzhen University.
All rights reserved.

Description :
some utils used for training

Author：Team Li
"""
import tensorflow as tf
import numpy as np
import config
import cv2

from utils.kp2hm_utils import heat_map_tf

def get_feedback_hm_loss_for_each_bbox(hm_pred, bbox, class_index, radius_radio=3):
    """get feed back loss for each bbox
    Args:
        hm_pred: net predition for heat map, with a shape [1, 128, 128, n_class], value is in [0., 1.]
        bbox: tensor represent the bbox [ymin, xmin, ymax, xmax], value is in [0., 1.]
        class_index: tensor represent the class index
        radius_radio: consideration range for loss calculation
    Return:
        a tensor with the shape (6, ), represents the different key point loss for heat map
    """
    def get_loss_for_kp(point):
        """ ..
        Args:
            point: a tensor with the shape (2,), mean (x, y), tf.float, value between [0, 1]
        Return:
            feedback loss for a key point
        """
        bbox_min_len = tf.cast(tf.minimum((bbox[2]-bbox[0])*config.hm_size[0]/radius_radio,
                                          (bbox[3]-bbox[1])*config.hm_size[1]/radius_radio), tf.int32)

        y_index = tf.cast(point[1] * config.hm_size[0], tf.int32)
        x_index = tf.cast(point[0] * config.hm_size[1], tf.int32)

        pos_pred = tf.gather_nd(hm_pred, [0, y_index, x_index, class_index])
        consider_radius = tf.maximum(bbox_min_len, config.min_radius_for_feedback_cal)

        y_begin = tf.maximum(y_index-consider_radius, 0)
        x_begin = tf.maximum(x_index-consider_radius, 0)
        size_y = tf.minimum(consider_radius + y_index - y_begin, config.hm_size[0] - y_begin)
        size_x = tf.minimum(consider_radius + x_index - x_begin, config.hm_size[1] - x_begin)
        consider_range_pred = tf.squeeze(tf.slice(hm_pred, begin=(0, y_begin, x_begin, class_index), size=(1, size_y, size_x, 1)))

        pos_loss = -tf.pow(1 - pos_pred, config.focal_loss_alpha)*tf.log(pos_pred)

        hm_gt = heat_map_tf(config.hm_size, point=point, sigma=consider_radius)
        consider_range_gt = tf.slice(hm_gt, begin=(y_begin, x_begin), size=(size_y, size_x))
        neg_mask = tf.cast(tf.less(consider_range_gt, 1.), tf.float32)
        neg_loss = -tf.pow(1. - consider_range_gt, config.focal_loss_belta)*tf.pow(consider_range_pred, config.focal_loss_alpha)*tf.log(1 - consider_range_pred*neg_mask)
        neg_loss = tf.reduce_mean(neg_loss)

        return pos_loss+neg_loss

    ## points in x, y cordinate
    left_top = tf.stack([bbox[1], bbox[0]], axis=0)
    left_top_loss = get_loss_for_kp(left_top)

    right_top = tf.stack([bbox[3], bbox[0]], axis=0)
    right_top_loss = get_loss_for_kp(right_top)

    left_bottom = tf.stack([bbox[1], bbox[2]], axis=0)
    left_bottom_loss = get_loss_for_kp(left_bottom)

    right_bottom = tf.stack([bbox[3], bbox[2]], axis=0)
    right_bottom_loss = get_loss_for_kp(right_bottom)

    center = tf.stack([(bbox[1]+bbox[3])/2., (bbox[0]+bbox[2])/2.], axis=0)
    center_loss = get_loss_for_kp(center)

    ## different groups loss
    group_1 = left_top_loss + right_bottom_loss

    group_2 = right_top_loss + left_bottom_loss

    group_3 = center_loss + left_top_loss

    group_4 = center_loss + right_top_loss

    group_5 = center_loss + left_bottom_loss

    group_6 = center_loss + right_bottom_loss
    return tf.stack([group_1, group_2, group_3, group_4, group_5, group_6], axis=0)




if __name__ == '__main__':
    net_pred = tf.placeholder(shape=(None, 128, 128, 80), dtype=tf.float32)
    bbox = tf.placeholder(shape=(4), dtype=tf.float32)
    pl, nl = get_feedback_hm_loss_for_each_bbox(net_pred, bbox=bbox, class_index=23)


    with tf.Session() as sess:
        for i in range(100):
            c = np.random.uniform(0, 1, (10, 128, 128, 80))

            b = np.random.uniform(0, 1, 2)
            xmin = np.min(b)
            xmax = np.max(b)
            cc = np.random.uniform(0, 1, 2)
            ymin = np.min(cc)
            ymax = np.max(cc)

            apll, nll = sess.run([pl, nl], feed_dict={net_pred: c, bbox:np.array([ymin, xmin, ymax, xmax])})
            pass

    pass