import os
import train

from config import Config
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from concat import concat

BATCH_SIZE = 64
IMAGE_SIZE = 256
NUM_CHANNEL = 1
NUM_LABELS = 2
NUM_ITER = 100000
NUM_SHOWTRAIN = 100 #show result eveary epoch 
NUM_SHOWTEST = 10000


LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.9
STEPSIZE = 5000
activation_func1 = tf.nn.relu
activation_func2 = tf.nn.tanh

Q_1 = 1
Q_2 = 2
Q_3 = 4
T = 4

c = Config()
c['path1'] = '/data/lgq/basic50k/basic50k_train/cover'
c['path2'] = '/data/lgq/basic50k/basic50k_test/cover'
c['path3'] = '/data/lgq/basic50k/basic50k_train/stego_j-uniward_40'
c['path4'] = '/data/lgq/basic50k/basic50k_test/stego_j-uniward_40'
c['numdata'] = 25000
c['batchsize'] = BATCH_SIZE


fileList1 = []
for (dirpath,dirnames,filenames) in os.walk(c['path1']):  
    fileList1 = filenames
fileList2 = []
for (dirpath,dirnames,filenames) in os.walk(c['path2']):  
    fileList2 = filenames
c['flist1'] = fileList1
c['flist2'] = fileList2

def weight_variable(shape,n_layer):
    w_name = 'w%s' %n_layer
    initial = tf.random_normal(shape,mean=0.0,stddev=0.01)
    return tf.Variable(initial, name=w_name)


def conv2d(input,w,s,n_layer):
    conv_name = 'conv%s' % n_layer
    with tf.name_scope(conv_name):
        conv = tf.nn.conv2d(input,w,strides=[1,s,s,1],padding='SAME',name=conv_name)
    return conv

def relu(input,n_layer):
    relu_name = 'relu%s' % n_layer
    with tf.name_scope(relu_name):
        output = tf.nn.relu(input,name=relu_name)
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

def conv_abs_bn_relu(input,shape,is_train,n_layer):
    conv_name = 'conv_%s'%n_layer
    with tf.name_scope(conv_name):
        w =  weight_variable(shape,n_layer)
        conv = conv2d(input,w,2,n_layer)
        conv = tf.abs(conv)
        bn = slim.layers.batch_norm(conv,scale=True,is_training=is_train,updates_collections=None)
        bn = tf.nn.relu(bn)
    return bn

def conv_bn_act(input,shape,activation,is_train,n_layer):
    conv_name = 'conv_%s'%n_layer
    with tf.name_scope(conv_name):
        w =  weight_variable(shape,n_layer)
        conv = conv2d(input,w,1,n_layer)
        bn = slim.layers.batch_norm(conv,scale=True,is_training=is_train,updates_collections=None)
        bn = activation(bn)
    return bn

def model(x,is_train):
    with tf.variable_scope("conv0") as scope:
        ############ Get 5x5 DCT base x25 ########################
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
                DCTbase = A[:, mode_r]*np.transpose(A[:, mode_c])
                weight[:, :, modeIndex] = DCTbase
        ##########################################################

        w0 = tf.Variable(weight, name="w0", dtype=tf.float32)
        w0 = tf.reshape(w0, [5, 5, 1, 25])
        conv0 = train.conv2d(x, w0, 0)
        
        print "w0:{0}".format(w0.get_shape())
        conv0_s_1 = tf.abs(tf.round(conv0 / Q_1))
        conv0_s_1 = T - tf.nn.relu(T - conv0_s_1)
        print "conv0_s_1:{0}".format(conv0_s_1)
        conv0_s_2 = tf.abs(tf.round(conv0 / Q_2))
        conv0_s_2 = T - tf.nn.relu(T - conv0_s_2)
        conv0_s_3 = tf.abs(tf.round(conv0 / Q_3))
        conv0_s_3 = T - tf.nn.relu(T - conv0_s_3)
#        print "conv0_s_1:{0}".format(conv0_s_1.get_shape())

    with tf.variable_scope("conv1") as scope:
        conv1_s_1 = conv_abs_bn_relu(conv0_s_1,[5,5,25,8],is_train,1)
        conv1_s_2 = conv_abs_bn_relu(conv0_s_2,[5,5,25,8],is_train,2)
        conv1_s_3 = conv_abs_bn_relu(conv0_s_3,[5,5,25,8],is_train,3)
#        print "conv1:{0}".format(conv1.get_shape())

    with tf.variable_scope("conv2") as scope:
        conv2_s_1 = conv_bn_act(conv1_s_1,[3,3,8,32],activation_func1,is_train,4)
        pool2_s_1 = train.avg_pool(conv2_s_1,5,2,'SAME',5)
        conv2_s_2 = conv_bn_act(conv1_s_2,[3,3,8,32],activation_func1,is_train,6)
        pool2_s_2 = train.avg_pool(conv2_s_2,5,2,'SAME',7)
        conv2_s_3 = conv_bn_act(conv1_s_3,[3,3,8,32],activation_func1,is_train,8)
        pool2_s_3 = train.avg_pool(conv2_s_3,5,2,'SAME',9)
#        print "pool2:{0}".format(pool2.get_shape())

    with tf.variable_scope("conv3") as scope:
        conv3_s_1 = conv_bn_act(pool2_s_1,[1,1,32,128],activation_func1,is_train,10)
        pool3_s_1 = train.avg_pool(conv3_s_1,32,32,'SAME',11)
        pool3_s_1 = tf.reshape(pool3_s_1, [BATCH_SIZE, 512])
        conv3_s_2 = conv_bn_act(pool2_s_2,[1,1,32,128],activation_func1,is_train,12)
        pool3_s_2 = train.avg_pool(conv3_s_2,32,32,'SAME',13)
        pool3_s_2 = tf.reshape(pool3_s_2, [BATCH_SIZE, 512])
        conv3_s_3 = conv_bn_act(pool2_s_3,[1,1,32,128],activation_func1,is_train,14)
        pool3_s_3 = train.avg_pool(conv3_s_3,32,32,'SAME',15)
        pool3_s_3 = tf.reshape(pool3_s_3, [BATCH_SIZE, 512])
        print "pool3_s_1:{0}".format(pool3_s_1.get_shape())

        # concat.py
        # customize tf.concat, values' type change into tf.float32
        pool3 = concat(1,[pool3_s_1, pool3_s_2, pool3_s_3])
#        print "pool3:{0}".format(pool3.get_shape())

    with tf.variable_scope('fc1') as scope:
        fc1_weights = tf.get_variable("weights", shape=[512*3,800],
                             initializer=tf.contrib.layers.xavier_initializer())
        fc1_bias = tf.get_variable("bias", [800],
                             initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(pool3, fc1_weights) + fc1_bias)

    with tf.variable_scope('fc2') as scope:
        fc2_weights = tf.get_variable("weights", shape=[800,400],
                             initializer=tf.contrib.layers.xavier_initializer())
        fc2_bias = tf.get_variable("bias", [400],
                             initializer=tf.constant_initializer(0.1))
        fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_bias)

    with tf.variable_scope('fc3') as scope:
        fc3_weights = tf.get_variable("weights", shape=[400,200],
                             initializer=tf.contrib.layers.xavier_initializer())
        fc3_bias = tf.get_variable("bias", [200],
                             initializer=tf.constant_initializer(0.1))
        fc3 = tf.nn.relu(tf.matmul(fc2, fc3_weights) + fc3_bias)

    with tf.variable_scope('fc4') as scope:
        fc4_weights = tf.get_variable("weights", shape=[200,2],
                             initializer=tf.contrib.layers.xavier_initializer())
        fc4_bias = tf.get_variable("bias", [2],
                             initializer=tf.constant_initializer(0.1))
        y_ = tf.matmul(fc3, fc4_weights) + fc4_bias

    vars = tf.trainable_variables()
    params = [v for v in vars if ( v.name.startswith('conv1/') or  v.name.startswith('conv2/') or  v.name.startswith('conv3/')\
                                   or v.name.startswith('fc1/') or  v.name.startswith('fc2/') or  v.name.startswith('fc3/') 
                                   or  v.name.startswith('fc4/')) ]
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
        tf.summary.scalar('acc',accuracy)
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=y))
        tf.summary.scalar('loss',loss)
        learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, STEPSIZE, LEARNING_RATE_DECAY, staircase=True)
        opt = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,var_list=params, global_step=global_step)

    data_x = np.zeros([BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNEL])
    data_y = np.zeros([BATCH_SIZE,NUM_LABELS])

    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        #saver.restore(sess,"/home/weiweihang/program/python/tensorflow/xu_model_wei/xu_m_wei_10000.ckpt")
	
        count = 0
        for i in range(1,NUM_ITER+1):
            data_x,data_y,count = train.get_data_match(data_x,data_y,c,count,0)
            pre,_,l,temp = sess.run([y_,opt,loss,accuracy],feed_dict={x:data_x,y:data_y,is_train:True})

            if i%NUM_SHOWTRAIN==0:  
                print 'training: batch result'
                print 'epoch:', i
                print 'loss:', l
                print 'accuracy:', temp
                print ' '
               
            if i%(NUM_SHOWTEST)==0:
                saver.save(sess,'/home/lgq/Workspace/jsdl_model/saver/jsdl_m_lgq_'+str(i)+'.ckpt')

            if i%NUM_SHOWTEST==0:
                result = np.array([]) #accuracy for training set
                #num = i/NUM_SHOWTEST - 1
                test_count = 0
                while test_count < 25000:
                    data_x,data_y,test_count = train.get_data_match(data_x,data_y,c,test_count,1)
                    l1,temp1 = sess.run([loss,accuracy],feed_dict={x:data_x,y:data_y,is_train:False})
                    result = np.insert(result,0,temp1)
                    #                    if test_count%100==0:
                    #                        print temp1
                print 'Testing accuracy:', np.mean(result)

if __name__ == '__main__':
    process()
        
