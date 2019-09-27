"""
Copyright (c) College of Mechatronics and Control Engineering, Shenzhen University.
All rights reserved.

Description :


Author：Team Li
"""
import tensorflow as tf
import numpy as np


aa = tf.random.uniform(shape=(5, 4, 128, 128), dtype=tf.float32)

bb = tf.placeholder(shape=[None, None, 3], dtype=tf.int64)
cc = bb[0]
indices = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    indices_1 = np.random.randint(0, 4, (5, 3))
    indices_2 = np.random.randint(0, 4, (2, 3))
    indices.append(indices_1)
    indices.append(indices_2)
    ee = sess.run(bb, feed_dict={bb:np.array(indices)})
    pass



# tensor_a = tf.Variable([[1,2,3],[4,5,6],[7,8,9]])
# tensor_b = tf.Variable([[0],[1],[2]],dtype=tf.int32)
# tensor_c = tf.Variable([[0],[0],[0]],dtype=tf.int32)
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(sess.run(tf.batch_gather(tensor_a,tensor_b)))
#     print(sess.run(tf.batch_gather(tensor_a,tensor_c)))


