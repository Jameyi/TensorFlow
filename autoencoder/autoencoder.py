import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def xavier_init(fan_in,fan_out,constant = 1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in,fan_out),minval=low,maxval=high,dtype=tf.float32)

class NoiseAutoencoder(object):
    def __init__(self,n_input,n_hidden,transfer_function=tf.nn.softplus,optimizer=tf.train.AdamOptimizer(),scale=0.1):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.training_scale = scale
        self.weights = dict()
        
        with tf.name_scope('RawInput'):
            self.x = tf.placeholder(tf.float32,[None,self.n_input])
        
        with tf.name_scope('NoiseAdder'):
            self.scale = tf.placeholder(tf.float32)
            self.noise_x = self.x+self.scale*tf.random_normal((n_input,))
        
        with tf.name_scope('Encoder'):
            self.weights['w1'] = tf.Variable(xavier_init(self.n_input,self.n_hidden),name='weight1')
            self.weights['b1'] = tf.Variable(tf.zeros([self.n_hidden],dtype=tf.float32),name='bias1')
            self.hidden = self.transfer(tf.add(tf.matmul(self.noise_x,self.weights['w1']),self.weights['b1']))

        with tf.name_scope('Reconstruction'):
            self.weights['w2'] = tf.Variable(tf.zeros([self.n_hidden,self.n_input],dtype=tf.float32),name='weight2')
            self.weights['b2'] = tf.Variable(tf.zeros([self.n_input],dtype=tf.float32),name='bias2')
            self.reconstruction = tf.add(tf.matmul(self.hidden,self.weights['w2']),self.weights['b2'])

        with tf.name_scope('Cost'):
            self.cost = 0.5*tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction,self.x),2))
        
        with tf.name_scope('Train'):
            self.optimizer = optimizer.minimize(self.cost)
        
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        print("begin to run session...")



AGN_AC = NoiseAutoencoder(n_input=784,
                          n_hidden=200,
                          transfer_function=tf.nn.softplus,
                          optimizer=tf.train.AdamOptimizer(learning_rate=0.01),
                          scale=0.01)

print("Writing to event doucument")
writer = tf.summary.FileWriter(logdir='logs',graph=AGN_AC.sess.graph)
writer.close()


