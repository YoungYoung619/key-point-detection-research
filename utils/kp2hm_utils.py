"""
Copyright (c) College of Mechatronics and Control Engineering, Shenzhen University.
All rights reserved.

Description :
script used to transform the key points to heap map

Author：Team Li
"""
import numpy as np
import time, cv2, random
import tensorflow as tf

def heat_map(img_size, points, sigmas):
    """produce a heat map(gray scale) according the points
    Args:
        img_size: img height and width
        points: ndarray or list, represents the coordinate of points. (x, y)
        sigma: control the heap point range
    return:
        a heap map with the shape (h, w)
    Example:
        aa = heat_map(img_size=(224, 224), points=[[50, 50], [100, 100]], sigma=2)
        cv2.imshow('test', aa)
        cv2.waitKey()
        cv2.destroyAllWindows()
    """
    assert legal_points(img_size, points)

    x = np.arange(0, img_size[1], 1)
    y = np.arange(0, img_size[0], 1)
    z = np.swapaxes(np.array(np.meshgrid(x, y)), axis1=0, axis2=2)
    z = np.swapaxes(z, axis1=0, axis2=1)
    heat_map = np.array([gaussian_2d(z, point, sigma=sigma) for point, sigma in zip(points, sigmas)])
    heat_map = np.max(heat_map, axis=0)
    return heat_map/np.max(heat_map)

def heat_map_tf(img_size, points, sigmas):
    """produce a heat map(gray scale) according the points
    Args:
        img_size: img height and width
        points: ndarray or list, represents the coordinate of points. (x, y)
        sigma: control the heap point range
    return:
        a heap map with the shape (h, w)
    Example:
        aa = heat_map(img_size=(224, 224), points=[[50, 50], [100, 100]], sigma=2)
        cv2.imshow('test', aa)
        cv2.waitKey()
        cv2.destroyAllWindows()
    """
    x = np.arange(0, img_size[1], 1)
    y = np.arange(0, img_size[0], 1)
    z = tf.meshgrid(x, y)
    z = tf.stack(z, axis=-1)
    # heat_map = tf.map_fn(fn=lambda point:gaussian_2d_tf(z, point, ))
    heat_map = np.array([gaussian_2d(z, point, sigma=sigma) for point, sigma in zip(points, sigmas)])
    heat_map = np.max(heat_map, axis=0)
    return heat_map/np.max(heat_map)


def legal_points(img_size, points):
    """judge all points whether in img range
    Args:
        img_size: [img_h, img_w]
        points: a list of point in x,y coordinate.
    Return:
        if all legal, True
        else, False
    """
    for point in points:
        if point[0] < 0 or point[0] > img_size[1]-1:
            return False
        if point[1] < 0 or point[1] > img_size[0]-1:
            return False
    return True


def gaussian_2d(point, mu, sigma):
    """2d gaussion function
    Args:
        point: 2d point coordinate
        mu: 2d mean value
        sigma: the standard deviation in gaussian func
    Return:
        a img with the shape (h, w)
    Example:
        x = np.arange(0, 224, 1)
        y = np.arange(0, 224, 1)
        z = np.swapaxes(np.array(np.meshgrid(x, y)), axis1=0, axis2=2)

        z = gauss_2d(z, mu=(100,100), sigma=2)
        cv2.imshow('test', z)
        cv2.waitKey()
        cv2.destroyAllWindows()
    """
    h = point.shape[0]
    w = point.shape[1]
    point = np.reshape(point, newshape=[-1, 2])
    score = np.exp(-(np.sum(np.square(np.array(point)-np.array(mu)), axis=-1))/(2*sigma**2))

    return np.reshape(score, newshape=[h, w])


def gaussian_2d_tf(point, mu, sigma):
    """2d gaussion function
    Args:
        point: 2d point coordinate
        mu: 2d mean value
        sigma: the standard deviation in gaussian func
    Return:
        a img with the shape (h, w)
    Example:
        x = np.arange(0, 224, 1)
        y = np.arange(0, 224, 1)
        z = np.swapaxes(np.array(np.meshgrid(x, y)), axis1=0, axis2=2)

        z = gauss_2d(z, mu=(100,100), sigma=2)
        cv2.imshow('test', z)
        cv2.waitKey()
        cv2.destroyAllWindows()
    """
    h = point.shape[0]
    w = point.shape[1]
    point = tf.reshape(point, shape=[-1, 2])
    score = tf.exp(-(tf.reduce_sum(tf.square(point - mu), axis=-1))/(2*sigma**2))

    return np.reshape(score, newshape=[h, w])


if __name__ == '__main__':
    # t0 = time.time()
    # for i in range(10000):
    #     ## time consume
    #     # aa = heat_map(img_size=(128, 128), points=[[50, 50]], sigma=2)
    #
    #     ## vis
    #     x = np.random.randint(0, 127, 5)
    #     y = np.random.randint(0, 127, 5)
    #     # print(x)
    #     # print(y)
    #     aa = heat_map(img_size=(128, 128), points=[[x[0], y[0]], [x[1], y[1]], [x[2], y[2]], [x[3], y[3]], [x[4], y[4]]], sigmas=[10, 8, 6, 5, 3])
    #     cv2.imshow('test', aa)
    #     cv2.waitKey()
    #     cv2.destroyAllWindows()
    # print('totol time is ', round(time.time()-t0, 4)) ## 2630 FPS for one point

    # print(np.logical_or(1.1, 0.1))

    points = tf.placeholder(shape=[None, 2], dtype=tf.float32)
    sigmas = tf.placeholder(shape=[None], dtype=tf.float32)
    heat_map_tf((128, 128), points, sigmas)