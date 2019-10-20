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

from utils.tf_extended import tf_utils
from utils.kp2hm_utils import heat_map_tf
from utils.tf_triplet_loss import triplet_semihard_loss

group_index_dict = {'lt_rb': 0, 'rt_lb': 1, 'c_lt': 2,
                    'c_rt': 3, 'c_lb': 4, 'c_rb': 5}


def loss_for_batch(b_hm_preds, b_emb_preds, b_offset_preds, b_bboxes, b_class_indexs):
    def get_hm_preds(index):
        hm_preds = []
        for b_hm_pred in b_hm_preds:
            hm_preds.append(tf.gather(b_hm_pred, [index]))
        return hm_preds

    def get_emb_preds(index):
        emb_preds = []
        for b_emb_pred in b_emb_preds:
            emb_preds.append(tf.gather(b_emb_pred, [index]))
        return emb_preds

    def get_offset_preds(index):
        offset_preds = []
        for b_offset_pred in b_offset_preds:
            offset_preds.append(tf.gather(b_offset_pred, [index]))
        return offset_preds

    def tfw_condition(i, total_loss):
        return tf.less(i, tf.shape(b_bboxes)[0])

    def tfw_body(i, total_loss):
        loss = loss_for_one_img(get_hm_preds(i), get_emb_preds(i), get_offset_preds(i), b_bboxes[i], b_class_indexs[i])
        loss = tf.concat([total_loss, tf.expand_dims(loss, dim=0)], axis=0)
        return i+1, loss

    i = tf.constant(1)
    loss = loss_for_one_img(get_hm_preds(0), get_emb_preds(0), get_offset_preds(0), b_bboxes[0], b_class_indexs[0])
    loss = tf.expand_dims(loss, axis=0)

    i, loss_ = tf.while_loop(tfw_condition, tfw_body, [i, loss], shape_invariants=[i.get_shape(), tf.TensorShape([None])])
    return tf.reduce_mean(loss_)


def loss_for_one_img(hm_preds, emb_preds, offset_preds, bboxes, class_indexs):
    """calculate the loss for one img
    Args:
        hm_pred: a list of heat map prediction, each elemt with a shape [1, 128, 128, n_class]
        emb_pred: a list of embedding prediction, each elemt with a shape [1, 128, 128, 1]
        offset_pred: a list of offset prediction, each elemt with a shape [1 ,128, 128, 2]
        bbox: ground truth bbox, [ymin, xmin, ymax, xmax], with a shape (n_bboxes, 4)
        class_indexs: class index, with a shape (n_bboxes,)
    Return:
        a total loss for training
    """
    hm_gts, embedding_pairs, offset_pairs, points_mask = encode_for_one_img(hm_preds, emb_preds, offset_preds, bboxes,
                                                                             class_indexs)

    ## heat map loss
    pos_hm_loss = hm_pos_loss_for_one_img(hm_preds, points_mask, bboxes, class_indexs)
    # tf.summary.scalar('pos_hm_loss', pos_hm_loss)

    neg_hm_loss = hm_neg_loss_for_one_img(hm_preds, hm_gts)
    # tf.summary.scalar('neg_hm_loss', neg_hm_loss)

    ## embedding loss
    embedding_loss = embedding_pairs_loss_for_one_img(embedding_pairs)
    # tf.summary.scalar('embedding_loss', embedding_loss)

    ## offset loss
    offset_loss = offset_loss_for_one_img(offset_pairs, points_mask)
    # tf.summary.scalar('offset_loss', offset_loss)

    return pos_hm_loss + neg_hm_loss/30. + 0.1*embedding_loss + offset_loss


def offset_loss_for_one_img(offset_pairs, points_mask):
    def _smooth_l1(x):
        """Smoothed absolute function. Useful to compute an L1 smooth error.
        Define as:
            x^2 / 2         if abs(x) < 1
            abs(x) - 0.5    if abs(x) > 1
        We use here a differentiable definition using min(x) and abs(x). Clearly
        not optimal, but good enough for our purpose!
        """
        absx = tf.abs(x)
        minx = tf.minimum(absx, 1)
        r = 0.5 * ((absx - 1) * minx + absx)  ## smooth_l1
        return r

    offset_loss = []
    for offset_pair, point_mask in zip(offset_pairs, points_mask):
        offset_diff_in_xy = tf.reduce_sum(offset_pair[:, :, 0] - offset_pair[:, :, 1], axis=-1)
        offset_diff_in_xy *= point_mask
        offset_loss.append(tf.reduce_sum(_smooth_l1(offset_diff_in_xy)))

    offset_loss = tf.stack(offset_loss, axis=0)
    offset_loss = tf.reduce_sum(offset_loss)
    return offset_loss


def embedding_pairs_loss_for_one_img(embedding_pairs):
    """calculate the loss of embedding pairs
    Args:
        embedding_pairs: a tensor with a shape (n_bboxes, 2)
    """
    labels = tf.linspace(0., tf.cast(tf.shape(embedding_pairs)[0] - 1, tf.float32), tf.shape(embedding_pairs)[0])
    labels = tf.stack([labels, labels], axis=-1)
    labels = tf.cast(tf.reshape(labels, shape=[-1]), dtype=tf.int32)
    embedding_ = tf.expand_dims(tf.reshape(embedding_pairs, shape=[-1]), axis=-1)
    return triplet_semihard_loss(labels, embedding_, margin=1.0)


def hm_pos_loss_for_one_bbox(hm_preds, points_mask, bboxes, class_indexs, index):
    """ pos relative loss calculation for one box
    Args:
        hm_pred: a list of heat map prediction, each elemt with a shape [1, 128, 128, n_class]
        points_mask: a list of mask, each elemt with a shape (n_bboxes, ), denotes whether a key point is a pos
        bbox: ground truth bbox, [ymin, xmin, ymax, xmax], with a shape (n_bboxes, 4)
        class_index: class index, with a shape (n_bboxes,)
        index: indicate which bbox to be calculated
    Return:
        a list of heat map ground truth, each elemt with a shape [[1, 128, 128, n_class]]
        a tensor represents embedding pair, [key_point_1_embeding, key_point_2_embeding]
        a list of offset pair for 5 different key points, each with shape (2,), [offset_preds, offset_gt]
        a list of mask, each with a shape (1,), represents whether a key point is be selective.
    """
    bbox = tf.gather(bboxes, index)
    class_index = tf.gather(class_indexs, [index])

    ## points in x, y cordinate
    left_top = tf.stack([bbox[1], bbox[0]], axis=0) * config.hm_size[::-1]
    left_top_int = tf.cast(left_top, tf.int32)  ## xy cord

    right_top = tf.stack([bbox[3], bbox[0]], axis=0) * config.hm_size[::-1]
    right_top_int = tf.cast(right_top, tf.int32)  ## xy cord

    left_bottom = tf.stack([bbox[1], bbox[2]], axis=0) * config.hm_size[::-1]
    left_bottom_int = tf.cast(left_bottom, tf.int32)  ## xy cord

    right_bottom = tf.stack([bbox[3], bbox[2]], axis=0) * config.hm_size[::-1]
    right_bottom_int = tf.cast(right_bottom, tf.int32)  ## xy cord

    center = tf.stack([(bbox[1] + bbox[3]) / 2., (bbox[0] + bbox[2]) / 2.], axis=0) * config.hm_size[::-1]
    center_int = tf.cast(center, tf.int32)  ## xy cord

    points_cord = [left_top_int, right_top_int, left_bottom_int, right_bottom_int, center_int] ## in x, y cord

    with tf.name_scope('pos_hm_loss'):
        pos_hm_loss = []
        for hm_pred, point_mask, point_cord in zip(hm_preds, points_mask, points_cord):
            hm_pred = tf.gather_nd(hm_pred, [[0, point_cord[1], point_cord[0], tf.cast(class_index[0], tf.int32)]])
            point_mask = tf.gather(point_mask, [index])
            hm_pred = tf.clip_by_value(hm_pred, 1e-10, 1.)
            pos_hm_loss.append(-tf.pow((1.-hm_pred), config.focal_loss_alpha) * tf.log(hm_pred) * point_mask)
        pos_hm_loss = tf.concat(pos_hm_loss, axis=0)
        pos_hm_loss = tf.reduce_sum(pos_hm_loss)

    return pos_hm_loss


def hm_neg_loss_for_one_img(hm_preds, hm_gts):
    """calculate the neg loss for heat map"""
    neg_loss = []
    for hm_pred, hm_gt in zip(hm_preds, hm_gts):
        neg_mask = tf.cast(tf.less(hm_gt, 1.), tf.float32)
        hm_pred = tf.squeeze(hm_pred)
        hm_pred = tf.clip_by_value(hm_pred, 1e-10, 1.)
        # neg_pred = tf.clip_by_value(1. - hm_pred, 1e-10, 1.)
        neg_pred = 1. - hm_pred
        neg_loss.append(-tf.pow(1. - hm_gt, config.focal_loss_belta)*tf.pow(hm_pred, config.focal_loss_alpha)*tf.log(neg_pred)*neg_mask)

    neg_loss = tf.reduce_sum(tf.stack(neg_loss, axis=0))
    return neg_loss


def hm_pos_loss_for_one_img(hm_preds, points_mask, bboxes, class_indexs):
    """calculate the pos loss for heat map"""
    ########## pos loss for hm #############
    def tfw_condition_pos_hm(i, pos_hm_loss):
        return tf.less(i, tf.shape(bboxes)[0])

    def tfw_body_pos_hm(i, pos_hm_loss):
        pos_hm_loss_ = hm_pos_loss_for_one_bbox(hm_preds, points_mask, bboxes, class_indexs, i)
        pos_hm_loss = tf.concat([pos_hm_loss, tf.expand_dims(pos_hm_loss_, axis=0)], axis=0)
        return i+1, pos_hm_loss

    i = tf.constant(1)
    pos_hm_loss = hm_pos_loss_for_one_bbox(hm_preds, points_mask, bboxes, class_indexs, 0)
    pos_hm_loss = tf.expand_dims(pos_hm_loss, axis=0)

    i, pos_hm_loss = tf.while_loop(tfw_condition_pos_hm, tfw_body_pos_hm, [i, pos_hm_loss], shape_invariants=[i.get_shape(),
                                                                                             tf.TensorShape([None])])
    pos_hm_loss = tf.reduce_sum(pos_hm_loss)
    ########## pos loss for hm #############

    return pos_hm_loss


def encode_for_one_img(hm_preds, emb_preds, offset_preds, bboxes, class_indexs, radius_radio=6):
    """encode for one img, the tf while_loop is so disgusting!!
    Args:
        hm_pred: a list of heat map prediction, each elemt with a shape [1, 128, 128, n_class]
        emb_pred: a list of embedding prediction, each elemt with a shape [1, 128, 128, 1]
        offset_pred: a list of offset prediction, each elemt with a shape [1 ,128, 128, 2]
        bbox: ground truth bbox, [ymin, xmin, ymax, xmax], with a shape (n_bboxes, 4)
        class_indexs: class index, with a shape (n_bboxes, )
    Return:
        a list of heat map ground truth, each elemt with a shape (128, 128, n_class)
        a tensor represents the embedding, with a shape (n_bboxes, 2), [key_point_1_embeding, key_point_2_embeding]
        a list of offset pairs for 5 different key points, each with shape (n_bboxes, 2, 2), elemt[i] is [offset_preds∈R^2, offset_gt∈R^2]
        a list of masks, each with a shape (n_bboxes,), represents whether a key point is be selective.
    """
    def tfw_condition(i, *args):
        return tf.less(i, tf.shape(bboxes)[0])

    def tfw_body(i, *args):
        hm_gts, embedding, offset_pairs, points_mask = tf_utils.reshape_list(args, shape=[5, 1, 5, 5])
        hm_gts_, emb_pair_, offset_pairs_, points_mask_ = encode_for_one_bbox(hm_preds, emb_preds, offset_preds, bboxes[i],
                                                                              class_indexs[i], radius_radio)
        temp_hm_gt = []
        for hm_gt, hm_gt_ in zip(hm_gts, hm_gts_):
            temp_hm_gt.append(tf.maximum(hm_gt, hm_gt_))

        emb_ = tf.concat([embedding, emb_pair_], axis=0)

        temp_offset = []
        for offset_pair, offset_pair_ in zip(offset_pairs, offset_pairs_):
            temp_offset.append(tf.concat([offset_pair, offset_pair_], axis=0))

        temp_point_mask = []
        for point_mask, point_mask_ in zip(points_mask, points_mask_):
            temp_point_mask.append(tf.concat([point_mask, point_mask_], axis=0))

        return [i+1] + temp_hm_gt + [emb_] + temp_offset + temp_point_mask

    i = tf.constant(1)
    hm_gts_, embedding_pair, offset_pairs, points_mask = encode_for_one_bbox(hm_preds, emb_preds, offset_preds, bboxes[0],
                                                                          class_indexs[0], radius_radio)

    shape_invariants = [i.get_shape()] + [hm_gts_[0].get_shape()] * 5 + [tf.TensorShape((None, embedding_pair.get_shape()[1]))] + \
                       [tf.TensorShape((None, offset_pairs[0].get_shape().as_list()[1], offset_pairs[0].get_shape().as_list()[2]))] * 5 +\
                       [tf.TensorShape([None])] * 5
    loop_vars = [i] + hm_gts_ + [embedding_pair] + offset_pairs + points_mask
    i, hm_gts, embedding_pairs, offset_pairs, points_mask = tf_utils.reshape_list(tf.while_loop(tfw_condition, tfw_body, loop_vars,
                                                                                                shape_invariants=shape_invariants), shape=[1, 5, 1, 5, 5])
    return hm_gts, embedding_pairs, offset_pairs, points_mask


def encode_for_one_bbox(hm_preds, emb_preds, offset_preds, bbox, class_index, radius_radio=3):
    """encode for one bbox
    Args:
        hm_pred: a list of heat map prediction, each elemt with a shape [1, 128, 128, n_class]
        emb_pred: a list of embedding prediction, each elemt with a shape [1, 128, 128, 1]
        offset_pred: a list of offset prediction, each elemt with a shape [1 ,128, 128, 2]
        bbox: ground truth bbox, [ymin, xmin, ymax, xmax], with a shape (4,)
        class_index: class index, with a shape (1,)
    Return:
        a list of heat map ground truth, each elemt with a shape (128, 128, n_class)
        a tensor represents embedding pair, with a shape (1, 2)[key_point_1_embeding, key_point_2_embeding]
        a list of offset pair for 5 different key points, each with shape (1, 2, 2), the last dim is [offset_preds_x(y), offset_gt_x(y)]
        a list of mask, each with a shape (1,), represents whether a key point is be selective.
    """
    ## heat map ground truth
    # bbox_min_len = tf.cast(tf.minimum((bbox[2] - bbox[0]) * config.hm_size[0] / radius_radio,
    #                                   (bbox[3] - bbox[1]) * config.hm_size[1] / radius_radio), tf.int32)
    # consider_radius = tf.maximum(bbox_min_len, config.min_radius_for_feedback_cal)
    det_size = tf.stack([bbox[2]-bbox[0], bbox[3]-bbox[1]], axis=0)*config.hm_size
    consider_radius = gaussian_radius(det_size, min_overlap=0.7)
    consider_radius = tf.maximum(consider_radius, config.min_radius_for_feedback_cal)

    ## points in x, y cordinate
    ones = tf.ones(shape=(config.hm_size[0], config.hm_size[1], config.n_class))
    class_one_hot = tf.one_hot(class_index, config.n_class)

    left_top = tf.stack([bbox[1], bbox[0]], axis=0) * config.hm_size[::-1]
    left_top_int = tf.cast(left_top, tf.int32)  ## xy cord
    lt_hm_gt = heat_map_tf(config.hm_size, left_top_int, consider_radius/radius_radio)
    lt_hm_gt = tf.expand_dims(lt_hm_gt, dim=-1) * ones * class_one_hot

    right_top = tf.stack([bbox[3], bbox[0]], axis=0) * config.hm_size[::-1]
    right_top_int = tf.cast(right_top, tf.int32)  ## xy cord
    rt_hm_gt = heat_map_tf(config.hm_size, right_top_int, consider_radius/radius_radio)
    rt_hm_gt = tf.expand_dims(rt_hm_gt, dim=-1) * ones * class_one_hot


    left_bottom = tf.stack([bbox[1], bbox[2]], axis=0) * config.hm_size[::-1]
    left_bottom_int = tf.cast(left_bottom, tf.int32)  ## xy cord
    lb_hm_gt = heat_map_tf(config.hm_size, left_bottom_int, consider_radius/radius_radio)
    lb_hm_gt = tf.expand_dims(lb_hm_gt, dim=-1) * ones * class_one_hot

    right_bottom = tf.stack([bbox[3], bbox[2]], axis=0) * config.hm_size[::-1]
    right_bottom_int = tf.cast(right_bottom, tf.int32)  ## xy cord
    rb_hm_gt = heat_map_tf(config.hm_size, right_bottom_int, consider_radius/radius_radio)
    rb_hm_gt = tf.expand_dims(rb_hm_gt, dim=-1) * ones * class_one_hot

    center = tf.stack([(bbox[1] + bbox[3]) / 2., (bbox[0] + bbox[2]) / 2.], axis=0)  * config.hm_size[::-1]
    center_int = tf.cast(center, tf.int32)  ## xy cord
    c_hm_gt = heat_map_tf(config.hm_size, center_int, consider_radius/radius_radio)
    c_hm_gt = tf.expand_dims(c_hm_gt, dim=-1) * ones * class_one_hot

    hm_gts = [lt_hm_gt, rt_hm_gt, lb_hm_gt, rb_hm_gt, c_hm_gt]

    ## different group loss calculation
    g_loss = get_feedback_hm_loss_for_one_bbox(hm_preds, hm_gts, bbox, class_index, consider_radius)
    g_index = tf.argmax(g_loss)

    ## select mining point pos mask
    lt_mask = tf.cast(tf.logical_or(tf.equal(g_index, group_index_dict['lt_rb']),
                                    tf.equal(g_index, group_index_dict['c_lt'])), tf.float32)
    rt_mask = tf.cast(tf.logical_or(tf.equal(g_index, group_index_dict['rt_lb']),
                                    tf.equal(g_index, group_index_dict['c_rt'])), tf.float32)
    lb_mask = tf.cast(tf.logical_or(tf.equal(g_index, group_index_dict['rt_lb']),
                                    tf.equal(g_index, group_index_dict['c_lb'])), tf.float32)
    rb_mask = tf.cast(tf.logical_or(tf.equal(g_index, group_index_dict['lt_rb']),
                                    tf.equal(g_index, group_index_dict['c_rb'])), tf.float32)
    c_mask = tf.cast(tf.logical_not(tf.logical_or(tf.equal(g_index, group_index_dict['lt_rb']),
                                                  tf.equal(g_index, group_index_dict['rt_lb']))), tf.float32)

    points_pos_mask = [lt_mask, rt_mask, lb_mask, rb_mask, c_mask] ## each step, 2 in it must be 1, other 0

    ## select the corresponding embedding to be pairs
    lt_embedding = tf.gather_nd(emb_preds[0], [[0, left_top_int[1], left_top_int[0], 0]])
    rt_embedding = tf.gather_nd(emb_preds[1], [[0, right_top_int[1], right_top_int[0], 0]])
    lb_embedding = tf.gather_nd(emb_preds[2], [[0, left_bottom_int[1], left_bottom_int[0], 0]])
    rb_embedding = tf.gather_nd(emb_preds[3], [[0, right_bottom_int[1], right_bottom_int[0], 0]])
    c_embedding = tf.gather_nd(emb_preds[4], [[0, center_int[1], center_int[0], 0]])

    lt_rb_emb_pair = tf.concat([lt_embedding, rb_embedding], axis=0)
    rt_lb_emb_pair = tf.concat([rt_embedding, lb_embedding], axis=0)
    c_lt_emb_pair = tf.concat([c_embedding, lt_embedding], axis=0)
    c_rt_emb_pair = tf.concat([c_embedding, rt_embedding], axis=0)
    c_lb_emb_pair = tf.concat([c_embedding, lb_embedding], axis=0)
    c_rb_emb_pair = tf.concat([c_embedding, rb_embedding], axis=0)

    lt_rb_emb_pair = lt_rb_emb_pair * tf.cast(tf.equal(g_index, group_index_dict['lt_rb']), tf.float32)
    rt_lb_emb_pair = rt_lb_emb_pair * tf.cast(tf.equal(g_index, group_index_dict['rt_lb']), tf.float32)
    c_lt_emb_pair = c_lt_emb_pair * tf.cast(tf.equal(g_index, group_index_dict['c_lt']), tf.float32)
    c_rt_emb_pair = c_rt_emb_pair * tf.cast(tf.equal(g_index, group_index_dict['c_rt']), tf.float32)
    c_lb_emb_pair = c_lb_emb_pair * tf.cast(tf.equal(g_index, group_index_dict['c_lb']), tf.float32)
    c_rb_emb_pair = c_rb_emb_pair * tf.cast(tf.equal(g_index, group_index_dict['c_rb']), tf.float32)

    emb_pair = lt_rb_emb_pair + rt_lb_emb_pair + c_lt_emb_pair + c_rt_emb_pair + c_lb_emb_pair + c_rb_emb_pair
    emb_pair = tf.expand_dims(emb_pair, axis=0)

    ## select the offset for pos
    lt_offset_pred = tf.squeeze(tf.gather_nd(offset_preds[0], [[0, left_top_int[1], left_top_int[0]]]))
    rt_offset_pred = tf.squeeze(tf.gather_nd(offset_preds[1], [[0, right_top_int[1], right_top_int[0]]]))
    lb_offset_pred = tf.squeeze(tf.gather_nd(offset_preds[2], [[0, left_bottom_int[1], left_bottom_int[0]]]))
    rb_offset_pred = tf.squeeze(tf.gather_nd(offset_preds[3], [[0, right_bottom_int[1], right_bottom_int[0]]]))
    c_offset_pred = tf.squeeze(tf.gather_nd(offset_preds[4], [[0, center_int[1], center_int[0]]]))

    lt_offset = left_top - tf.cast(left_top_int, tf.float32)
    rt_offset = right_top - tf.cast(right_top_int, tf.float32)
    lb_offset = left_bottom - tf.cast(left_bottom_int, tf.float32)
    rb_offset = right_bottom - tf.cast(right_bottom_int, tf.float32)
    c_offset = center - tf.cast(center_int, tf.float32)

    lt_offset_pair = tf.expand_dims(tf.stack([lt_offset_pred, lt_offset], axis=-1), axis=0)
    rt_offset_pair = tf.expand_dims(tf.stack([rt_offset_pred, rt_offset], axis=-1), axis=0)
    lb_offset_pair = tf.expand_dims(tf.stack([lb_offset_pred, lb_offset], axis=-1), axis=0)
    rb_offset_pair = tf.expand_dims(tf.stack([rb_offset_pred, rb_offset], axis=-1), axis=0)
    c_offset_pair = tf.expand_dims(tf.stack([c_offset_pred, c_offset], axis=0), axis=0)
    offsets = [lt_offset_pair, rt_offset_pair, lb_offset_pair, rb_offset_pair, c_offset_pair]

    return hm_gts, emb_pair, offsets, points_pos_mask


# def get_feedback_hm_loss_for_one_img(hm_pred, bboxes, classes):
#     """get feedback loss for one img
#     Args:
#         hm_pred: tensor, the net predition for the img in heat map, shape is (1, 128, 128, n_class)
#         bboxes: tensor, the groundtruth bbxoes for the img, val in [0, 1], [ymin, xmin, ymax, xmax], shape is (n_bboxes, 4)
#         classes: tensor, class index for the img, shape is (n_bboxes,)
#     Return:
#         group loss for each bbox, shape is (n_bboxes, 6)
#     """
#     def tfw_condition(i, group_loss):
#         return tf.less(i, tf.shape(bboxes)[0])
#
#     def tfw_body(i, group_loss):
#         group_loss_ = get_feedback_hm_loss_for_one_bbox(hm_pred, bboxes[i], classes[i])
#         group_loss_ = tf.concat([group_loss, tf.expand_dims(group_loss_, dim=0)], axis=0)
#         return i+1, group_loss_
#
#     i = tf.constant(1)
#     group_loss = get_feedback_hm_loss_for_one_bbox(hm_pred, bboxes[0], classes[0])
#     group_loss = tf.expand_dims(group_loss, dim=0)
#     i, group_loss = tf.while_loop(tfw_condition, tfw_body, [i, group_loss],
#                                   shape_invariants=[i.get_shape(), tf.TensorShape([None, group_loss.get_shape()[1]])])
#
#     return group_loss


def get_feedback_hm_loss_for_one_bbox(hm_preds, hm_gts, bbox, class_index, consider_radius):
    """get feed back loss for each bbox
    Args:
        hm_preds: a list of net predition for heat map, each elment with a shape [1, 128, 128, n_class], value is in [0., 1.]
        hm_gts: a list of net groundtruth for heat map, each elment with a shape [1, 128, 128, n_class], value is in [0., 1.]
        bbox: tensor represent the bbox [ymin, xmin, ymax, xmax], value is in [0., 1.]
        class_index: tensor represent the class index
        consider_radius: consideration range for loss calculation
    Return:
        a tensor with the shape (6, ), represents the different key point loss for heat map
    """
    def get_loss_for_kp(hm_pred, hm_gt, point, consider_radius):
        """ ..
        Args:
            point: a tensor with the shape (2,), in (x, y), tf.float, value between [0, 1]
        Return:
            feedback loss for a key point
        """
        point = point * config.hm_size[::-1]
        y_index = tf.cast(point[1], tf.int32)
        x_index = tf.cast(point[0], tf.int32)

        pos_pred = tf.gather_nd(hm_pred, [[0, y_index, x_index, tf.cast(class_index, tf.int32)]])

        y_begin = tf.maximum(y_index-consider_radius, 0)
        x_begin = tf.maximum(x_index-consider_radius, 0)
        size_y = tf.minimum(consider_radius + y_index - y_begin, config.hm_size[0] - y_begin)
        size_x = tf.minimum(consider_radius + x_index - x_begin, config.hm_size[1] - x_begin)
        consider_range_pred = tf.squeeze(tf.slice(hm_pred, begin=(0, y_begin, x_begin, tf.cast(class_index, tf.int32)), size=(1, size_y, size_x, 1)), axis=[0,3])

        # pos_pred = tf.clip_by_value(pos_pred, 1e-10, 1.)
        pos_loss = -tf.pow(1. - pos_pred, config.focal_loss_alpha)*tf.log(pos_pred)

        hm_gt = tf.transpose(hm_gt, perm=[2, 0, 1])
        hm_gt = tf.gather_nd(hm_gt, [class_index])
        consider_range_gt = tf.slice(hm_gt, begin=(y_begin, x_begin), size=(size_y, size_x))
        neg_mask = tf.cast(tf.less(consider_range_gt, 1.), tf.float32)
        # consider_range_pred = tf.clip_by_value(consider_range_pred, 1e-10, 1.)
        # neg_pred = tf.clip_by_value(1. - consider_range_pred, 1e-10, 1.)
        neg_pred = 1. - consider_range_pred
        neg_loss = -tf.pow(1. - consider_range_gt, config.focal_loss_belta)*tf.pow(consider_range_pred, config.focal_loss_alpha)*tf.log(neg_pred)*neg_mask
        neg_loss = tf.reduce_mean(neg_loss)

        return pos_loss + neg_loss

    # ###test
    # left_top = tf.stack([bbox[1], bbox[0]], axis=0)
    # return get_loss_for_kp(hm_preds[0], left_top)


    # ## points in x, y cordinate
    left_top = tf.stack([bbox[1], bbox[0]], axis=0)
    left_top_loss = get_loss_for_kp(hm_preds[0], hm_gts[0], left_top, consider_radius)

    right_top = tf.stack([bbox[3], bbox[0]], axis=0)
    right_top_loss = get_loss_for_kp(hm_preds[1], hm_gts[1], right_top, consider_radius)

    left_bottom = tf.stack([bbox[1], bbox[2]], axis=0)
    left_bottom_loss = get_loss_for_kp(hm_preds[2], hm_gts[2], left_bottom, consider_radius)

    right_bottom = tf.stack([bbox[3], bbox[2]], axis=0)
    right_bottom_loss = get_loss_for_kp(hm_preds[3], hm_gts[3], right_bottom, consider_radius)

    center = tf.stack([(bbox[1]+bbox[3])/2., (bbox[0]+bbox[2])/2.], axis=0)
    center_loss = get_loss_for_kp(hm_preds[4], hm_gts[4], center, consider_radius)

    ## different groups loss
    group_1 = left_top_loss + right_bottom_loss

    group_2 = right_top_loss + left_bottom_loss

    group_3 = center_loss + left_top_loss

    group_4 = center_loss + right_top_loss

    group_5 = center_loss + left_bottom_loss

    group_6 = center_loss + right_bottom_loss
    return tf.stack([group_1, group_2, group_3, group_4, group_5, group_6], axis=0)


def gaussian_radius(det_size, min_overlap=0.7):
    height = det_size[0]
    width = det_size[1]

    a1 = 1.
    b1 = (height + width)
    c1 = width * height * (1. - min_overlap) / (1. + min_overlap)
    sq1 = tf.sqrt(tf.square(b1) - 4. * a1 * c1)
    r1 = (b1 + sq1) / 2.

    a2 = 4.
    b2 = 2. * (height + width)
    c2 = (1. - min_overlap) * width * height
    sq2 = tf.sqrt(tf.square(b2) - 4. * a2 * c2)
    r2 = (b2 + sq2) / 2.

    a3 = 4. * min_overlap
    b3 = -2. * min_overlap * (height + width)
    c3 = (min_overlap - 1.) * width * height
    sq3 = tf.sqrt(tf.square(b3) - 4. * a3 * c3)
    r3 = (b3 + sq3) / 2.

    r = tf.stack([r1, r2, r3], axis=0)
    return tf.cast(tf.reduce_min(r), tf.int32)

if __name__ == '__main__':
    ## test-1 for get_feedback_hm_loss_for_each_bbox
    # net_pred = tf.placeholder(shape=(None, 128, 128, 80), dtype=tf.float32)
    # bbox = tf.placeholder(shape=(4), dtype=tf.float32)
    # pl = get_feedback_hm_loss_for_one_bbox([net_pred,net_pred,net_pred,net_pred,net_pred], bbox=bbox, class_index=23)
    #
    # with tf.Session() as sess:
    #     for i in range(100):
    #         c = np.random.uniform(0, 1, (10, 128, 128, 80))
    #
    #         b = np.random.uniform(0, 1, 2)
    #         xmin = np.min(b)
    #         xmax = np.max(b)
    #         cc = np.random.uniform(0, 1, 2)
    #         ymin = np.min(cc)
    #         ymax = np.max(cc)
    #
    #         print(np.array([ymin, xmin, ymax, xmax]))
    #         apll = sess.run(pl, feed_dict={net_pred: c, bbox:np.array([ymin, xmin, ymax, xmax])})
    #         print(apll)
    #         # cv2.imshow('test', apll)
    #         # cv2.waitKey()
    #         # cv2.destroyAllWindows()
    #         pass

    ## test-2 for get_feedback_hm_loss_for_one_img
    # net_pred = tf.placeholder(shape=(None, 128, 128, 80), dtype=tf.float32)
    # bboxes = tf.placeholder(shape=(None, 4), dtype=tf.float32)
    # labels = tf.placeholder(shape=(None), dtype=tf.int64)
    # get_feedback_hm_loss_for_one_img(net_pred, bboxes, labels)
    # pass

    ## test-3 encode_for_one_img

    ## test loss_for_one_img
    # net_pred = tf.placeholder(shape=[1, 128, 128, config.n_class], dtype=tf.float32)
    # emb_pred = tf.placeholder(shape=[1, 128, 128, 1], dtype=tf.float32)
    # offset_pred = tf.placeholder(shape=[1, 128, 128, 2], dtype=tf.float32)
    # bboxes = tf.placeholder(shape=[None, 4], dtype=tf.float32)
    # class_indexs = tf.placeholder(shape=[None], dtype=tf.int64)

    ## test loss_for_batch
    net_pred = tf.placeholder(shape=[2, 128, 128, config.n_class], dtype=tf.float32)
    emb_pred = tf.placeholder(shape=[2, 128, 128, 1], dtype=tf.float32)
    offset_pred = tf.placeholder(shape=[2, 128, 128, 2], dtype=tf.float32)
    bboxes = tf.placeholder(shape=[2, None, 4], dtype=tf.float32)
    class_indexs = tf.placeholder(shape=[2, None], dtype=tf.int64)

    # loss_1 = loss_for_one_img([net_pred]*5, [emb_pred]*5, [offset_pred]*5, bboxes, class_indexs)
    loss_2 = loss_for_batch([net_pred]*5, [emb_pred]*5, [offset_pred]*5, bboxes, class_indexs)

    with tf.Session() as sess:
        ## test loss_for_one_img
        # for i in range(100):
        #     net_pred_n = np.random.uniform(-0, 1, [1, 128, 128, config.n_class])
        #     emb_pred_n = np.random.uniform(-0, 1, [1, 128, 128, 1])
        #     offset_pred_n = np.random.uniform(-0, 1, [1, 128, 128, 2])
        #
        #     bboxes_n = []
        #     n_bbox = np.random.randint(1, 5, 1)[0]
        #     for i in range(n_bbox):
        #         c = np.random.uniform(0, 1, (10, 128, 128, 80))
        #
        #         b = np.random.uniform(0, 1, 2)
        #         xmin = np.min(b)
        #         xmax = np.max(b)
        #         cc = np.random.uniform(0, 1, 2)
        #         ymin = np.min(cc)
        #         ymax = np.max(cc)
        #         bboxes_n.append(np.array([ymin, xmin, ymax, xmax]))
        #     bboxes_n = np.array(bboxes_n)
        #     bboxes_n = np.reshape(bboxes_n, (n_bbox,4))
        #
        #     class_indexs_n = np.random.randint(0, config.n_class-1, [n_bbox])
        #
        #     l = sess.run(loss_1, feed_dict={net_pred: net_pred_n, emb_pred: emb_pred_n, offset_pred:offset_pred_n,
        #                                    bboxes: bboxes_n, class_indexs:class_indexs_n})
        #     print(l)

        ## test loss_for_batch_img
        for i in range(100):
            net_pred_n = np.random.uniform(0, 1, [2, 128, 128, config.n_class])
            emb_pred_n = np.random.uniform(0, 1, [2, 128, 128, 1])
            offset_pred_n = np.random.uniform(0, 1, [2, 128, 128, 2])

            bboxes_n = []
            n_bbox = np.random.randint(1, 5, 1)[0]
            for i in range(n_bbox):
                c = np.random.uniform(0, 1, (10, 128, 128, 80))

                b = np.random.uniform(0, 1, 2)
                xmin = np.min(b)
                xmax = np.max(b)
                cc = np.random.uniform(0, 1, 2)
                ymin = np.min(cc)
                ymax = np.max(cc)
                bboxes_n.append(np.array([0.5, 0.5, 0.5, 0.5]))
            bboxes_n = np.array(bboxes_n*2)
            bboxes_n = np.reshape(bboxes_n, (2, n_bbox, 4))

            class_indexs_n = np.random.randint(0, config.n_class, [2, n_bbox])

            l = sess.run(loss_2, feed_dict={net_pred: net_pred_n, emb_pred: emb_pred_n, offset_pred:offset_pred_n,
                                           bboxes: bboxes_n, class_indexs:class_indexs_n})
            print(l)

