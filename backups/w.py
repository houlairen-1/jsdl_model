
import os
import sys
import time
import train

import tensorflow as tf
import numpy as np
import random
from jpeg import jpeg

#BATCH_SIZE = 1
IMAGE_SIZE = 256

def process():
    path = '/data/lgq/basic50k/basic50k/basic50k_train/cover/n02783161_16345.jpg'
    x = jpeg(path).getSpatial()
    k = np.linspace(0, 4, 5)
    l = np.linspace(0, 4, 5)
    [k, l] = np.meshgrid(k, l)

    A = np.cos(((2*k+1)*l*np.pi)/10)/np.sqrt(5)
    A[0, :] = A[0, :] / np.sqrt(2)
    A = A*np.sqrt(2)
    A = np.transpose(A)
    weight = np.zeros([5, 5, 25], dtype=np.float32)
    for mode_r in range(5):
        for mode_c in range(5):
            modeIndex = mode_r*5 + mode_c
            DCTbase = np.dot(A[:, mode_r], np.transpose(A[:, mode_c]))
            weight[:, :, modeIndex] = DCTbase

    print weight[:, :, 1]
    w0 = tf.Variable(weight, name="w0")
    w0 = tf.reshape(w0, [5, 5, 25])
    sess = tf.InteractiveSession()
#    with tf.name_scope("conv0"):
#        conv0 = tf.nn.conv2d(x, w0, strides=[1, 1, 1, 1], padding='SAME',name="conv0")
#        print conv0
#        conv0 = conv0 / 1
#        conv0 = tf.nn.relu(conv0+4) - 4
#        conv0 = -tf.nn.relu(4-conv0) + 4
#        print conv0

if __name__ == '__main__':
    process()
        
