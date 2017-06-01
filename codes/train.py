''' 
some function used for net
list:weight_variable , maxpool , avgpool ,
     conv2d , relu, tanh, BatchNorm ,
     bottleneck,nonbottleneck(resnet),crelu,
     conv_bn_pool_relu,dimention_increase
     separate_conv2d,concat_select,select_cover_conv2d
'''
import os
import sys
sys.path.append('/home/lgq/Workspace/jsdl_model/tools')

from config import Config
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm
import numpy as np
from jpeg import jpeg
import random

# generate weight for net ,the name of weight must be diffierent
def weight_variable(shape,n_layer):
    w_name = 'w%s' %n_layer
    initial = tf.random_normal(shape,mean=0.0,stddev=0.01)
    return tf.Variable(initial, name=w_name)

# generate weight for net ,the name of weight can be same
def _get_variable(name,
                  shape,
                  initializer,
                  weight_decay=0.0,
                  dtype='float',
                  trainable=True):
    if weight_decay > 0:
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        regularizer = None
    return tf.get_variable(name,
                           shape=shape,
                           initializer=initializer,
                           dtype=dtype,
                           regularizer=regularizer,
                           trainable=trainable)


"""
batch_normalization 
arg:input:the size of input is [batch,weight,height,channel]or[batch,weight]
     train_phase:whether or not the layers is in the trainning model
     n_layer: the numbel of layer
output:its size is the same as input
Note: is_training is tf.placeholder(tf.bool) type
such as: is_train = tf.placeholder(tf.bool,name='train_phase')
"""
def BatchNorm(input,train_phase,n_layer):    
    scope_name = 'BN%s' % n_layer
    bn_train = batch_norm(input, decay=0.999, center=True, scale=True,updates_collections=None,is_training=True,reuse=None,scope=scope_name)
    bn_inference = batch_norm(input, decay=0.999, center=True, scale=True,updates_collections=None,is_training=False,reuse=True,scope=scope_name)
    z = tf.cond([train_phase], lambda: bn_train, lambda: bn_inference)
    return z


def conv2d(input,w,n_layer):
    conv_name = 'conv%s' % n_layer
    with tf.name_scope(conv_name):
        conv = tf.nn.conv2d(input,w,strides=[1,1,1,1],padding='SAME',name=conv_name)
    return conv


#some nonlinear operator
#input(input<0)=0  input(input>0)=input
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

#some pooling operator
def avg_pool(input,k,s,pad,n_layer):
    pool_name = 'avgpool%s' % n_layer
    with tf.name_scope(pool_name):
        output = tf.nn.avg_pool(input,ksize=[1,k,k,1],strides=[1,s,s,1],padding=pad,name=pool_name)
    return output

def avg_pool(input,k,s,pad,n_layer):
    pool_name = 'avgpool%s' % n_layer
    with tf.name_scope(pool_name):
        output = tf.nn.avg_pool(input,ksize=[1,k,k,1],strides=[1,s,s,1],padding=pad,name=pool_name)
    return output



#the input through convolution,relu,average pooling and batch normalization in sequence 
def conv_relu_pool_bn(input,shape,is_train,n_layer):
    conv_name = 'conv_%s'%n_layer
    with tf.name_scope(conv_name):
        w =  weight_variable(shape,n_layer)
        conv = tf.nn.relu(conv2d(input,w,n_layer))
        pool = max_pool(conv,3,1,'SAME')
        bn = BatchNorm(pool,is_train,n_layer)
    return bn

# the input through 3x3 conv_relu_pool_bn operator twice 
def non_bottleneck(input,in_channel,is_train,n_layer):
    nonbottleneck_name = 'nonbottleneck%s' % n_layer
    N_layer = n_layer*2
    with tf.name_scope(nonbottleneck_name):
        shortcut = input
        input = conv_relu_pool_bn(input,[3,3,in_channel,in_channel],is_train,N_layer-1)
        input = tf.nn.relu(input)
        input = conv_relu_pool_bn(input,[3,3,in_channel,in_channel],is_train,N_layer)
    return tf.nn.relu(shortcut + input)


        
def bottleneck(input,in_channel,n_layer):
    bottleneck_name = 'bottleneck%s' % n_layer
    with tf.name_scope(bottleneck_name):
        shortcut = input
        input = conv_relu_pool_bn(input,[1,1,in_channel,in_channel],is_train,1)
        input = tf.nn.relu(input)
        input = conv_relu_pool_bn(input,[3,3,in_channel,in_channel],is_train,2)
        input = tf.nn.relu(input)
        input = conv_relu_pool_bn(input,[1,1,in_channel,in_channel],is_train,3)
    return tf.nn.relu(shortcut + input)    



def dimension_increase(input,shape,n_layer):
    dimensionincrease_name = 'dimension_increase%s'%n_layer
    with tf.name_scope(dimensionincrease_name):
        w =  weight_variable(shape,n_layer)
        conv = conv2d(input,w,n_layer)
        maxpool = max_pool(conv,3,2,'SAME')
    return maxpool

def separate_conv2d(input,filter):
    output = []
    num = filter.get_shape()[0] 
    for i in range(num):
        output.append(input[i,:,:,:]*filter[i,:,:,:])
    return tf.pack(output)


# arg: input : its size is [batch,weight,height,channel]
#       filter: its size is  [weight,height]
#out = input(dot multiply)filter       
def select_cover_conv2d(input,filter,batchsize):
    output = []
    keep = tf.ones([batchsize,512,512,1])
    output.append(input[:,:,:,0]*filter[:,:,:,0])
    output.append(input[:,:,:,1]*keep[:,:,:,0])
    return tf.transpose(tf.pack(output),perm=[1,2,3,0])

# out = [input(dot multiply)filter]concat input  
def concat_select(input,filter):
    out = []
    out.append(input[:,:,:,0]*filter[:,:,:,0])
    out = tf.transpose(tf.pack(out),perm=[1,2,3,0])
    output = tf.concat(1,[input,out])
    return output

#out = relu( [input*(-1) concat input] )
def crelu(input,n_layer):
    crelu_name = 'crelu%s' % n_layer
    with tf.name_scope(crelu_name):
	shortcut = input*(-1)
	out = tf.concat(1,[input,shortcut])
    return tf.nn.relu(out , name=crelu_name)

###def get_data_same(inputx,inputy,c,flist,count,type):
###    #c:path1~4,numdata,batchsize,
###    numdata = c['numdata']
###    batchsize = c['batchsize']
###    if type==0:
###        path = c['path1']
###    elif type==1:
###        path = c['path2']
###    elif type==2:
###        path = c['path3']
###    else:
###        path = c['path4']
###
###    if (count%numdata==0 and count!=0):
###        np.set_printoptions(threshold='nan')
###        random.seed(count/numdata)
###        random.shuffle(flist)
###    
###    for j in range(batchsize):
###        imc = jpeg(path+'/'+flist[count]).getSpatial()
###        if (type==0 and type==2):
###            inputy[j,0] = 0
###            inputy[j,1] = 1
###        else:
###            inputy[j,0] = 1
###            inputy[j,1] = 0
###        count = count+1
###        inputx[j,:,:,0] = imc.astype(np.float32)
###    return inputx,inputy,count

def get_data_match(inputx,inputy,c,count,type):   
    #type:'0'is training,'1'is testing
    numdata = c['numdata']
    batchsize = c['batchsize']
    
    if (count%numdata==0 and count!=0):
        np.set_printoptions(threshold='nan')
        random.seed(count/numdata)
        random.shuffle(c['flist1'])
        
    flist1 = c['flist1']
    flist2 = c['flist2']
    if type ==0:
        num = count % numdata
        for j in range(batchsize):
            if j%2==0:
                imc = jpeg(c['path1']+'/'+flist1[num]).getSpatial()
                inputy[j,0] = 0
                inputy[j,1] = 1
            else:
                imc = jpeg(c['path3']+'/'+flist1[num]).getSpatial()
                inputy[j,0] = 1
                inputy[j,1] = 0
                count = count+1
                num = count % numdata
            inputx[j,:,:,0] = imc.astype(np.float32)
    else:
        num = count % numdata
        for j in range(batchsize):
            if j%2==0:
                imc = jpeg(c['path2']+'/'+flist2[num]).getSpatial()
                inputy[j,0] = 0
                inputy[j,1] = 1
            else:
                imc = jpeg(c['path4']+'/'+flist2[num]).getSpatial()
                inputy[j,0] = 1
                inputy[j,1] = 0
                count = count+1
                num = count % numdata
            inputx[j,:,:,0] = imc.astype(np.float32)
        
    return inputx,inputy,count

###def get_map_match(inputx,inputy,inputmap,c,count,type):   
###    #type:'0'is training,'1'is testing
###    numdata = c['numdata']
###    batchsize = c['batchsize']
###    
###    if (count%numdata==0 and count!=0):
###        np.set_printoptions(threshold='nan')
###        random.seed(count/numdata)
###        random.shuffle(c['flist1'])
###        
###    flist1 = c['flist1']
###    flist2 = c['flist2']
###    if type ==0:
###        num = count % numdata
###        for j in range(batchsize):
###            imap = jpeg(c['path5']+'/'+flist1[num]).getSpatial()
###            if j%2==0:
###                imc = jpeg(c['path1']+'/'+flist1[num]).getSpatial()
###                inputy[j,0] = 0
###                inputy[j,1] = 1
###            else:
###                imc = jpeg(c['path3']+'/'+flist1[num]).getSpatial()
###                inputy[j,0] = 1
###                inputy[j,1] = 0
###                count = count+1
###                num = num+1
###            inputx[j,:,:,0] = imc.astype(np.float32)
###            inputmap[j,:,:,0] = imap.astype(np.float32)
###    else:
###         for j in range(batchsize):
###            imap = jpeg(c['path5']+'/'+flist2[count]).getSpatial()
###            if j%2==0:
###                imc = jpeg(c['path2']+'/'+flist2[count]).getSpatial()
###                inputy[j,0] = 0
###                inputy[j,1] = 1
###            else:
###                imc = jpeg(c['path4']+'/'+flist2[count]).getSpatial()
###                inputy[j,0] = 1
###                inputy[j,1] = 0
###                count = count+1
###            inputx[j,:,:,0] = imc.astype(np.float32)
###            inputmap[j,:,:,0] = imap.astype(np.float32)
###        
###    return inputx,inputy,inputmap,count

def batch_norm_wrapper(inputs, is_training, decay = 0.999,epsilon = 0.001):

    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training is not None:
        batch_mean, batch_var = tf.nn.moments(inputs,[0,1,2])
        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                                             pop_mean, pop_var, beta, scale, epsilon)
    else:
        return tf.nn.batch_normalization(inputs,
                                         pop_mean, pop_var, beta, scale, epsilon)


def batch_norm_train(input,n_layer,decay = 0.999):
    moments_name = 'moments%s' % n_layer
    beta_name = 'bata%s' % n_layer
    gamma_name = 'gamma%s' % n_layer
    popmean_name = 'popmean%s' % n_layer
    popvar_name = 'popvar%s' % n_layer
    
    batch_mean,batch_var = tf.nn.moments(input,[0,1,2],name=moments_name)
    beta = tf.Variable(tf.zeros([input.get_shape()[-1]]),name=beta_name)
    gamma = tf.Variable(tf.ones([input.get_shape()[-1]]),name=gamma_name)
    pop_mean = tf.Variable(tf.zeros([input.get_shape()[-1]]), trainable=False,name=popmean_name)
    pop_var = tf.Variable(tf.ones([input.get_shape()[-1]]), trainable=False,name=popvar_name)
    train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
    train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
    with tf.control_dependencies([train_mean, train_var]):
        output = tf.nn.batch_normalization(input,batch_mean,batch_var,beta,gamma,1e-3)
    return output

def batch_norm_test(input,n_layer):
    moments_name = 'moments%s' % n_layer
    beta_name = 'bata%s' % n_layer
    gamma_name = 'gamma%s' % n_layer
    popmean_name = 'popmean%s' % n_layer
    popvar_name = 'popvar%s' % n_layer
    
    beta = tf.Variable(tf.zeros([input.get_shape()[-1]]),name=beta_name)
    gamma = tf.Variable(tf.ones([input.get_shape()[-1]]),name=gamma_name)  
    pop_mean = tf.Variable(tf.zeros([input.get_shape()[-1]]), trainable=False, name=popmean_name)
    pop_var = tf.Variable(tf.ones([input.get_shape()[-1]]), trainable=False,name=popvar_name)

    output = tf.nn.batch_normalization(input,pop_mean,pop_var,beta,gamma,1e-3)
    return output

def dimension_increase_train(input,shape,n_layer):
    dimensionincrease_name = 'dimension_increase%s'%n_layer
    with tf.name_scope(dimensionincrease_name):
        w =  weight_variable(shape,n_layer)
        conv = conv2d(input,w,n_layer)
        bn = batch_norm_train(conv,n_layer)
        avgpool = avg_pool(bn,5,2,'SAME',n_layer)
    return avgpool

def dimension_increase_test(input,shape,n_layer):
    dimensionincrease_name = 'dimension_increase%s'%n_layer
    with tf.name_scope(dimensionincrease_name):
        w =  weight_variable(shape,n_layer)
        conv = conv2d(input,w,n_layer)
        bn = batch_norm_test(conv,n_layer)
        avgpool = avg_pool(bn,5,2,'SAME',n_layer)
    return avgpool
