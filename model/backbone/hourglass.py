"""
Copyright (c) College of Mechatronics and Control Engineering, Shenzhen University.
All rights reserved.

Description :
A modification implement of hourglass as a backbone in our model, which is same as the cornernet's.
Some utils are from https://github.com/makalo/CornerNet

For more details of raw hourglass, see https://arxiv.org/abs/1603.06937
For more details of cornernet, see https://arxiv.org/abs/1808.01244

Author：Team Li
"""
import tensorflow as tf
import numpy as np

## some defualt config described in cornetnet
n_deep = 5
n_dims = [256, 256, 384, 384, 384, 512]
n_res = [2, 2, 2, 2, 2, 4]

def pre_process(input, is_training,  scope='init_process'):
    """a pre process convolution layer for raw input img, which be used to reduce img size by 4 times
    Args:
        input: a input tensor represents a img, with the shape [bs, h, w, 3]
    Return:
        output tensor with the shape[bs, h/4, w/4, 3]
    """
    with tf.variable_scope(scope):
        x = tf.contrib.layers.conv2d(input, 128, 7, 2)
        x = tf.contrib.layers.batch_norm(x, is_training=is_training)
        x = residual(x, 256, strides=(2, 2), scope='residual_start')
        return x


def hourglass(input, n_deep=n_deep, n_res=n_res, n_dims=n_dims, is_training=True, scope='hourglass_5'):
    """an implement of hourglass
    Args:
        input: tf tensor with the shape [bs, h, w, 3], default should be [bs, 511//4, 511//4, 3]
    Return:
        hourglass output tensor with the shape [bs, h, w, 3]
    """
    input = pre_process(input, is_training=is_training)

    with tf.variable_scope(scope):
        curr_res = n_res[0]
        next_res = n_res[1]
        curr_dim = n_dims[0]
        next_dim = n_dims[1]

        up_1 = res_block(input, curr_dim, curr_res, is_training=is_training, scope='up_1')

        half = tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        low_1 = res_block(half, next_dim, curr_res, is_training=is_training, scope='low_1')
        if n_deep > 1:
            low_2 = hourglass(low_1, n_deep - 1, n_res[1:], n_dims[1:], is_training=is_training,
                                   scope='hourglass_%d' % (n_deep - 1))
        else:
            low_2 = res_block(low_1, next_dim, next_res, is_training=is_training, scope='low_2')
        low_3 = res_block(low_2, curr_dim, curr_res, is_training=is_training, scope='low_3')

        up_2 = tf.image.resize_nearest_neighbor(low_3, tf.shape(low_3)[1:3] * 2, name='up_2')
        merge = tf.add(up_1, up_2)
        return merge


def res_block(input, out_dim, n, k=3, is_training=True, scope='res_block'):
    """a res block for hourglass
    Args:
        input: input tensor
        out_dim: output channel number
        n: repeat times
    Return: output tensor
    """
    with tf.variable_scope(scope):
        x = residual(input, out_dim, k=k, is_training=is_training, scope='residual_0')
        for i in range(1, n):
            x = residual(x, out_dim, k=k, is_training=is_training, scope='residual_%d' % i)
        return x


def residual(input, out_dim, k=3, strides=(1, 1), is_training=True, scope='residual'):
    """a residual implement
    Args:
        input: input tensor
        out_dim: output chennel number

    """
    with tf.variable_scope(scope):
        # assert input.get_shape().as_list()[3]==input_dim
        # low layer 3*3>3*3
        x = conv_bn_re(input, out_dim, strides=strides, is_training=is_training, scope='up_1')
        x = conv_bn_re(x, out_dim, use_relu=False, is_training=is_training, scope='up_2')
        # skip,up layer 1*1
        skip = conv_bn_re(input, out_dim, strides=strides, use_relu=False, k=1,
                               is_training=is_training, scope='low')
        # skip+x
        res = tf.nn.relu(tf.add(skip, x))
        return res


def conv_bn_re(input, out_dim, strides=(1, 1), use_relu=True, use_bn=True, k=3, is_training=True,
               scope='conv_bn_re'):
    """ a std conv and bn layer
    Args:
        input: input tensor
        out_dim: output channel number
    """
    with tf.variable_scope(scope):
        # x=tf.contrib.layers.conv2d(input,out_dim,k,stride=strides,activation_fn=None)
        x = tf.layers.conv2d(input, out_dim, k, strides=strides, padding='same')
        if use_bn:
            x = tf.contrib.layers.batch_norm(x, is_training=is_training)
        if use_relu:
            x = tf.nn.relu(x)
        return x


if __name__ == '__main__':
    input = tf.placeholder(shape=[None, 511, 511, 3], dtype=tf.float32)
    feats = hourglass(input)
    print('Total trainable parameters:%s' %
                str(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))
    pass