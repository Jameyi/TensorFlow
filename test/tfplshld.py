import tensorflow as tf
import numpy as np

x = tf.placeholder("float",[None,3])
y = x * 2

with tf.Session() as sess:
    x_data = [[1,2,3],[4,5,6]]
    result = sess.run(y,feed_dict={x:x_data})
    print(result)

#============================================
#x = tf.placeholder("float",3)
#y = x * 2

#with tf.Session() as sess:
#    result = sess.run(y,feed_dict={x:[1,2,3]})
#    print(result)


