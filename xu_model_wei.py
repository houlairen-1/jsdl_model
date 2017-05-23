''' 
a SHI-CNN example using tensorflow library.
This example is using the baseboss 1.01 database 
Author: Weihang Wei
create on :2017-3-20
'''
import os
import sys
import math
import time
import train

from config import Config
import tensorflow as tf
import numpy as np
import random
from scipy import ndimage
from pandas import DataFrame
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import utils
#from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm

BATCH_SIZE = 40
IMAGE_SIZE = 512
NUM_CHANNEL = 1
NUM_LABELS = 2
NUM_ITER = 120000
NUM_SHOWTRAIN = 100 #show result eveary epoch 
NUM_SHOWTEST = 10000


LEARNING_RATE =0.001
LEARNING_RATE_DECAY = 0.1
MOMENTUM = 0.9
decay_step = 10000
activation_func1 = tf.nn.relu
activation_func2 = tf.nn.tanh

c = Config()
c['path1'] = '/data/weiweihang/cover/train'
c['path2'] = '/data/weiweihang/cover/test'
c['path3'] = '/data/weiweihang/suniward_0.4/train'
c['path4'] = '/data/weiweihang/suniward_0.4/test'
c['numdata'] = 5000
c['batchsize'] = BATCH_SIZE


fileList1 = []
for (dirpath,dirnames,filenames) in os.walk(c['path1']):  
    fileList1 = filenames
fileList2 = []
for (dirpath,dirnames,filenames) in os.walk(c['path2']):  
    fileList2 = filenames
c['flist1'] = fileList1
c['flist2'] = fileList2
d = np.load('/home/weiweihang/program/python/tensorflow/xu_model_wei/remark/xu_100000.npz')
def weight_variable(shape,n_layer):
    w_name = 'w%s' %n_layer
    initial = tf.random_normal(shape,mean=0.0,stddev=0.01)
    return tf.Variable(initial, name=w_name)


def conv2d(input,w,n_layer):
    conv_name = 'conv%s' % n_layer
    with tf.name_scope(conv_name):
        conv = tf.nn.conv2d(input,w,strides=[1,1,1,1],padding='SAME',name=conv_name)
    return conv

def relu(input,n_layer):
    relu_name = 'relu%s' % n_layer
    with tf.name_scope(relu_name):
        output = tf.nn.relu(input,name=relu_name)
    return output

def tanh(input,n_layer):
    tanh_name = 'tanh%s' % n_layer
    with tf.name_scope(tanh_name):
        output = tf.nn.tanh(input,name=tanh_name)
    return output

def abs(input,n_layer):
    abs_name = 'abs%s' % n_layer
    with tf.name_scope(abs_name):
        output = tf.abs(input,name=abs_name)
    return output

def avg_pool(input,k,s,pad,n_layer):
    pool_name = 'avgpool%s' % n_layer
    with tf.name_scope(pool_name):
        output = tf.nn.avg_pool(input,ksize=[1,k,k,1],strides=[1,s,s,1],padding=pad,name=pool_name)
    return output

def max_pool(input,k,s,pad):
    output = tf.nn.max_pool(input,ksize=[1,k,k,1],strides=[1,s,s,1],padding=pad)
    return output

def conv_abs_bn_tanh_pool(input,shape,is_train,n_layer):
    conv_name = 'conv_%s'%n_layer
    with tf.name_scope(conv_name):
        w =  weight_variable(shape,n_layer)
        conv = conv2d(input,w,n_layer)
        conv = tf.abs(conv)
        bn = slim.layers.batch_norm(conv,scale=True,is_training=is_train,updates_collections=None)
        bn = tf.nn.tanh(bn)
        pool = max_pool(bn,5,2,'SAME')
    return pool

def conv_bn_act(input,shape,activation,is_train,n_layer):
    conv_name = 'conv_%s'%n_layer
    with tf.name_scope(conv_name):
        w =  weight_variable(shape,n_layer)
        conv = conv2d(input,w,n_layer)
        bn = slim.layers.batch_norm(conv,scale=True,is_training=is_train,updates_collections=None)
        bn = activation(bn)
    return bn

def model(x,is_train):
    with tf.variable_scope("conv0") as scope:
        hpf = np.zeros([5,5,1,1],dtype=np.float32)
        hpf[:,:,0,0] = np.array([[-1,2,-2,2,-1],[2,-6,8,-6,2],[-2,8,-12,8,-2],[2,-6,8,-6,2],[-1,2,-2,2,-1]],dtype=np.float32)/(12*255)
        w0 = tf.Variable(hpf,name="w0")
        conv0 = train.conv2d(x,w0,0)
        #bn0 = train.batch_norm_wrapper(conv0,is_train)             

    with tf.variable_scope("conv1") as scope:
        conv1 = conv_abs_bn_tanh_pool(conv0,[5,5,1,8],is_train,1)
    
    with tf.variable_scope("conv2") as scope:
        conv2 = conv_bn_act(conv1,[5,5,8,16],activation_func2,is_train,2)
        pool2 = train.avg_pool(conv2,5,2,'SAME',2)

    with tf.variable_scope("conv3") as scope:
        conv3 = conv_bn_act(pool2,[1,1,16,32],activation_func1,is_train,3)
        pool3 = train.avg_pool(conv3,5,2,'SAME',3)
 
    with tf.variable_scope("conv4") as scope:
        conv4 = conv_bn_act(pool3,[1,1,32,64],activation_func1,is_train,4)
        pool4 = train.avg_pool(conv4,5,2,'SAME',4)

    with tf.variable_scope("conv5") as scope:
        conv5 = conv_bn_act(pool4,[1,1,64,128],activation_func1,is_train,5)
        pool5 = train.avg_pool(conv5,32,1,'VALID',5)
            
    with tf.variable_scope('fully_connecting') as scope:
        w6 = train.weight_variable([128,2],6)
        bias = tf.Variable(tf.random_normal([2],mean=0.0,stddev=0.01),name="bias" )
        pool_shape = pool5.get_shape().as_list()
        pool_reshape = tf.reshape(pool5, [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
        y_ = tf.matmul(pool_reshape, w6) + bias 

    vars = tf.trainable_variables()
    params = [v for v in vars if ( v.name.startswith('conv1/') or  v.name.startswith('conv2/') or  v.name.startswith('conv3/')\
                                       or  v.name.startswith('conv4/') or  v.name.startswith('conv5/') or  v.name.startswith('fully_connecting/') ) ]
       
    #mean = tf.concat(0,[pop_mean1,pop_mean2,pop_mean3])
    return y_,params

def process():

    with tf.name_scope('input') as scope:
        x = tf.placeholder(tf.float32,shape=[BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNEL])
        y = tf.placeholder(tf.float32,shape=[BATCH_SIZE,NUM_LABELS])
        is_train = tf.placeholder(tf.bool)
   
    y_,params = model(x,is_train)
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    global_step = tf.Variable(0,trainable = False)
    with tf.name_scope('acc'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.scalar_summary('acc',accuracy)
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_, y))
        tf.scalar_summary('loss',loss)
        #decayed_learning_rate=tf.train.exponential_decay(LEARNING_RATE, global_step, decay_step,  LEARNING_RATE_DECAY,staircase=True)
        #opt = tf.train.MomentumOptimizer(decayed_learning_rate,MOMENTUM).minimize(loss,var_list=params,global_step=global_step)
        opt = tf.train.GradientDescentOptimizer(0.001).minimize(loss,var_list=params)

    data_x = np.zeros([BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNEL])
    data_y = np.zeros([BATCH_SIZE,NUM_LABELS])

    #merged = tf.merge_all_summaries()
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        tf.initialize_all_variables().run()
        #saver.restore(sess,"/home/weiweihang/program/python/tensorflow/xu_model_wei/xu_m_wei_10000.ckpt")
	
        count = 0
        for i in range(1,NUM_ITER+1):
            data_x,data_y,count = train.get_data_match(data_x,data_y,c,count,0)
            pre,_,l,temp = sess.run([y_,opt,loss,accuracy],feed_dict={x:data_x,y:data_y,is_train:True})
            #print pre

            if i%NUM_SHOWTRAIN==0:  
                print 'training: batch result'
                print 'epoch:', i
                print 'loss:', l
                print 'accuracy:', temp
                print ' '
               
            if i%(NUM_SHOWTEST)==0:
                saver.save(sess,'/home/weiweihang/program/python/tensorflow/xu_model_wei/xu_m_wei_'+str(i)+'.ckpt')

            if i%NUM_SHOWTEST==0:
                result = np.array([]) #accuracy for training set
                #num = i/NUM_SHOWTEST - 1
                test_count = 0
                while test_count < 5000:
                    data_x,data_y,test_count = train.get_data_match(data_x,data_y,c,test_count,1)
                    l1,temp1 = sess.run([loss,accuracy],feed_dict={x:data_x,y:data_y,is_train:False})
                    result = np.insert(result,0,temp1)
                    if test_count%100==0:
                        print temp1
                print 'Testing accuracy:', np.mean(result)

if __name__ == '__main__':
    process()
        
