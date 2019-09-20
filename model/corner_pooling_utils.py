"""
Copyright (c) College of Mechatronics and Control Engineering, Shenzhen University.
All rights reserved.

Description :


Author：Team Li
"""
import tensorflow as tf

def top_pooling(input):
    """ top pooling concept implement
    Args:
        input: input tensor with shape [bs, h, w, c]
    Return
        output tensor with the same shape as input
    """
    #forward
    def forward(input):
        out=tf.expand_dims(tf.reduce_max(input,1),1)
        i=tf.constant(1)
        batch,h,w,c=input.get_shape().as_list()
        def cond(i,out):
            return i < h
        def body(i,out):
            d=tf.expand_dims(tf.reduce_max(input[:,i:,:,:],1),1)
            out=tf.concat((out,d),1)
            i = i + 1
            return i,out
        _,out = tf.while_loop(cond, body, [i,out],shape_invariants= [i.get_shape(), tf.TensorShape([batch,None,w,c])])
        return out

    #backward
    def backward(input,dy):
        zeros=tf.expand_dims(tf.zeros_like(input[:,-1,:,:]),1)
        ones=tf.expand_dims(tf.ones_like(input[:,-1,:,:]),1)
        mask=tf.expand_dims(tf.ones_like(input[:,-1,:,:]),1)
        batch,h,w,c=input.get_shape().as_list()
        i=tf.constant(h-1)

        def cond(i,mask):
            return i > 0
        def body(i,mask):
            max_value=tf.expand_dims(tf.reduce_max(input[:,i:,:,:],1),1)
            temp_mask=tf.where(tf.greater(tf.expand_dims(input[:,i-1,:,:],1),max_value),ones,zeros)
            mask=tf.concat((temp_mask,mask),1)
            i = i - 1
            return i,mask
        _,mask = tf.while_loop(cond, body, [i,mask],shape_invariants= [i.get_shape(), tf.TensorShape([batch,None,w,c])])
        return mask*dy

    @tf.custom_gradient
    def new_grad(x):
        def grad(dy):
            return backward(x,dy)
        return forward(x), grad
    return new_grad(input)


def left_pooling(input):
    """ left pooling concept implement
    Args:
        input: input tensor with shape [bs, h, w, c]
    Return
        output tensor with the same shape as input
    """
    #forward
    def forward(input):
        out=tf.expand_dims(tf.reduce_max(input,2),2)
        i=tf.constant(1)
        batch,h,w,c=input.get_shape().as_list()
        def cond(i,out):
            return i < w
        def body(i,out):
            d=tf.expand_dims(tf.reduce_max(input[:,:,i:,:],2),2)
            out=tf.concat((out,d),2)
            i = i + 1
            return i,out
        _,out = tf.while_loop(cond, body, [i,out],shape_invariants= [i.get_shape(), tf.TensorShape([batch,h,None,c])])
        return out

    #backward
    def backward(input,dy):
        zeros=tf.expand_dims(tf.zeros_like(input[:,:,-1,:]),2)
        ones=tf.expand_dims(tf.ones_like(input[:,:,-1,:]),2)
        mask=tf.expand_dims(tf.ones_like(input[:,:,-1,:]),2)
        batch,h,w,c=input.get_shape().as_list()
        i=tf.constant(w-1)

        def cond(i,mask):
            return i > 0
        def body(i,mask):
            max_value=tf.expand_dims(tf.reduce_max(input[:,:,i:,:],2),2)
            temp_mask=tf.where(tf.greater(tf.expand_dims(input[:,:,i-1,:],2),max_value),ones,zeros)
            mask=tf.concat((temp_mask,mask),2)
            i = i - 1
            return i,mask
        _,mask = tf.while_loop(cond, body, [i,mask],shape_invariants= [i.get_shape(), tf.TensorShape([batch,h,None,c])])
        return mask*dy

    @tf.custom_gradient
    def new_grad(x):
        def grad(dy):
            return backward(x,dy)
        return forward(x), grad
    return new_grad(input)


def bottom_pooling(input):
    """ bottom pooling concept implement
    Args:
        input: input tensor with shape [bs, h, w, c]
    Return
        output tensor with the same shape as input
    """
    #forward
    def forward(input):
        out=tf.expand_dims(tf.reduce_max(input,1),1)
        batch,h,w,c=input.get_shape().as_list()
        i=tf.constant(h-1)

        def cond(i,out):
            return i > 0
        def body(i,out):
            d=tf.expand_dims(tf.reduce_max(input[:,:i,:,:],1),1)
            out=tf.concat((d,out),1)
            i = i - 1
            return i,out
        _,out = tf.while_loop(cond, body, [i,out],shape_invariants= [i.get_shape(), tf.TensorShape([batch,None,w,c])])
        return out

    #backward
    def backward(input,dy):
        zeros=tf.expand_dims(tf.zeros_like(input[:,-1,:,:]),1)
        ones=tf.expand_dims(tf.ones_like(input[:,-1,:,:]),1)
        mask=tf.expand_dims(tf.ones_like(input[:,-1,:,:]),1)
        batch,h,w,c=input.get_shape().as_list()
        i=tf.constant(1)

        def cond(i,mask):
            return i < h
        def body(i,mask):
            max_value=tf.expand_dims(tf.reduce_max(input[:,:i,:,:],1),1)
            temp_mask=tf.where(tf.greater(tf.expand_dims(input[:,i,:,:],1),max_value),ones,zeros)
            mask=tf.concat((mask,temp_mask),1)
            i = i + 1
            return i,mask
        _,mask = tf.while_loop(cond, body, [i,mask],shape_invariants= [i.get_shape(), tf.TensorShape([batch,None,w,c])])
        return mask*dy

    @tf.custom_gradient
    def new_grad(x):
        def grad(dy):
            return backward(x,dy)
        return forward(x), grad
    return new_grad(input)


def right_pooling(input):
    """ right pooling concept implement
    Args:
        input: input tensor with shape [bs, h, w, c]
    Return
        output tensor with the same shape as input
    """
    #forward
    def forward(input):
        out=tf.expand_dims(tf.reduce_max(input,2),2)
        batch,h,w,c=input.get_shape().as_list()
        i=tf.constant(w-1)

        def cond(i,out):
            return i > 0
        def body(i,out):
            d=tf.expand_dims(tf.reduce_max(input[:,:,:i,:],2),2)
            out=tf.concat((d,out),2)
            i = i - 1
            return i,out
        _,out = tf.while_loop(cond, body, [i,out],shape_invariants= [i.get_shape(), tf.TensorShape([batch,h,None,c])])
        return out

    #backward
    def backward(input,dy):
        zeros=tf.expand_dims(tf.zeros_like(input[:,:,-1,:]),2)
        ones=tf.expand_dims(tf.ones_like(input[:,:,-1,:]),2)
        mask=tf.expand_dims(tf.ones_like(input[:,:,-1,:]),2)
        batch,h,w,c=input.get_shape().as_list()
        i=tf.constant(1)

        def cond(i,mask):
            return i < w
        def body(i,mask):
            max_value=tf.expand_dims(tf.reduce_max(input[:,:,:i,:],2),2)
            temp_mask=tf.where(tf.greater(tf.expand_dims(input[:,:,i,:],2),max_value),ones,zeros)
            mask=tf.concat((mask,temp_mask),2)
            i = i + 1
            return i,mask
        _,mask = tf.while_loop(cond, body, [i,mask],shape_invariants= [i.get_shape(), tf.TensorShape([batch,h,None,c])])
        return mask*dy

    @tf.custom_gradient
    def new_grad(x):
        def grad(dy):
            return backward(x,dy)
        return forward(x), grad
    return new_grad(input)


if __name__ == '__main__':
    import numpy as np

    input = tf.placeholder(shape=[None, 50, 50, 3], dtype=tf.float32)
    a = top_pooling(input)

    data = np.random.uniform(0, 1, [2, 50, 50, 3])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        aaa = sess.run(a, feed_dict={input: data})
        pass