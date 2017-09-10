# -*- coding: UTF-8 -*- 
import tensorflow as tf
import numpy as np

## Save to file
# remember to define the same dtype and shape when restore
W = tf.Variable([[1,2,3],[3,4,5]], dtype=tf.float32, name='weights')
b = tf.Variable([[1,2,3]], dtype=tf.float32, name='biases')

init= tf.initialize_all_variables()

saver = tf.train.Saver()

# 用 saver 将所有的 variable 保存到定义的路径
with tf.Session() as sess:
   sess.run(init)
   save_path = saver.save(sess, "my_net/save_net.ckpt")
   print("Save to path: ", save_path)

