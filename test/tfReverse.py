import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


filename = "test.jpg"
image = mpimg.imread(filename)
height,width,depth = image.shape

x = tf.Variable(image,name='x')

model = tf.initialize_all_variables()

with tf.Session() as sess:
	
	x = tf.reverse_sequence(x,[width] * height,1)
	sess.run(model)
	result = sess.run(x)

print(result.shape)
plt.imshow(result)
plt.show()
