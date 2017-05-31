from concat import concat
import numpy as np
import tensorflow as tf

#a = np.zeros((2,4),dtype=np.float32)
#b = np.ones((2,4),dtype=np.float32)
a = [[96.6, 96.6, 96.6],[96.6, 96.6, 96.6]]
b = [[110.9, 110.9, 110.9],[110.9, 110.9, 110.9]]
x = [[0.7, 0.7, 0.7],[0.7, 0.7, 0.7]]
c = tf.Variable(a)
d = tf.Variable(b)
y = tf.Variable(x)
e = concat(1, [c, d, y], name='concat')
with tf.Session() as sess:
    tf.initialize_all_variables().run()
    print c.eval(session=sess)
    print d.eval(session=sess)
    print y.eval(session=sess)
    print e.eval(session=sess)
