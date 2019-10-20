"""
Copyright (c) College of Mechatronics and Control Engineering, Shenzhen University.
All rights reserved.

Description :


Author：Team Li
"""

from model.backbone.hourglass import hourglass
from model.corner_pooling_utils import *
from model.backbone.hourglass import *

slim = tf.contrib.slim

cornet_pooling_utils = {'top_pooling': top_pooling, 'right_pooling': right_pooling,
                        'bottom_pooling': bottom_pooling, 'left_pooling': left_pooling}

ops_dict = {'top_left': ['top_pooling', 'left_pooling'], 'top_right': ['top_pooling', 'right_pooling'],
                    'bottom_left': ['bottom_pooling', 'left_pooling'], 'bottom_right': ['bottom_pooling', 'right_pooling']}


class redudant_point_network():
    def __init__(self, input, is_training, n_class):
        self.n_class = n_class

        hourglass_feat = hourglass(input, is_training) ## shape [bs, h/4, w/4, 256]

        ## top left corner prediction
        self.tl_heat, self.tl_embeds, self.tl_offset = self.prediction_module(hourglass_feat, is_training, scope='top_left')

        ## top right cornet prediction
        self.tr_heat, self.tr_embeds, self.tr_offset = self.prediction_module(hourglass_feat, is_training, scope='top_right')

        ## bottom left corner prediction
        self.bl_heat, self.bl_embeds, self.bl_offset = self.prediction_module(hourglass_feat, is_training, scope='bottom_left')

        ## bottom right corner prediction
        self.br_heat, self.br_embeds, self.br_offset = self.prediction_module(hourglass_feat, is_training, scope='bottom_right')

        ## center prediction
        self.c_heat, self.c_embeds, self.c_offset = self.prediction_module(hourglass_feat, is_training, scope='center')

        self.output = [[slim.softmax(self.tl_heat), slim.softmax(self.tr_heat), slim.softmax(self.bl_heat), slim.softmax(self.br_heat), slim.softmax(self.c_heat)],
                  [self.tl_embeds, self.tr_embeds, self.bl_embeds, self.br_embeds, self.c_embeds],
                  [self.tl_offset, self.tr_offset, self.bl_offset, self.br_offset, self.c_offset]]
    def get_output(self):
        return  self.output


    def prediction_module(self, input, is_training, scope):
        """prediction module same with the corner net
        Args:
            input: input tensor
            scope: tf var scope
        """
        assert scope in ['top_left', 'top_right', 'bottom_right', 'bottom_left', 'center']

        with tf.variable_scope(scope):
            if scope is not 'center':
                ops = ops_dict[scope]
                feat_basic_0 = conv_bn_re(input, 128, is_training=is_training, scope=ops[0])
                feat_pool_0 = cornet_pooling_utils[ops[0]](feat_basic_0)
                feat_basic_1 = conv_bn_re(input, 128, is_training=is_training, scope=ops[1])
                feat_pool_1 = cornet_pooling_utils[ops[1]](feat_basic_1)

                feat = tf.add(feat_pool_0, feat_pool_1)
                feat = conv_bn_re(feat, 256, use_relu=False, is_training=is_training, scope=ops[0] + '_' + ops[1])
            else:
                feat = conv_bn_re(input, 128, is_training=is_training, scope='center_conv1')
                feat = conv_bn_re(feat, 256, use_relu=False, is_training=is_training, scope='center_conv2')

            skip_tl = conv_bn_re(input, 256, use_relu=False, kernel_size=1, is_training=is_training, scope='skip')
            merge_feat = tf.nn.relu(tf.add(skip_tl, feat))
            merge_feat = conv_bn_re(merge_feat, 256, is_training=is_training, scope='merge_'+scope)

            heat = self.heat_out(merge_feat)
            embedding = self.embedding_out(merge_feat)
            offset = self.offset_out(merge_feat)
            return heat, embedding, offset

    def heat_out(self, input, scope='heat_point_map'):
        with tf.variable_scope(scope):
            x = conv_bn_re(input, input.get_shape().as_list()[-1], use_bn=False)
            x = tf.layers.conv2d(x, self.n_class, 1)
        return x

    def embedding_out(self, input, scope='embedding'):
        with tf.variable_scope(scope):
            x = conv_bn_re(input, input.get_shape().as_list()[-1], use_bn=False)
            x = tf.layers.conv2d(x, 1, 1)
        return x

    def offset_out(self, input, scope='offset'):
        with tf.variable_scope(scope):
            x = conv_bn_re(input, input.get_shape().as_list()[-1], use_bn=False)
            x = tf.layers.conv2d(x, 2, 1)
        return x


if __name__ == '__main__':
    input = tf.placeholder(shape=[None, 511, 511, 3], dtype=tf.float32)
    r_n = redudant_point_network(input, is_training=True, n_class=80)
    print('Total trainable parameters:%s' %
          str(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))
    pass