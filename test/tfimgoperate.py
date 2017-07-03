import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

filename = "test.jpg"
raw_image_data = mpimg.imread(filename)

image = tf.placeholder("uint8",[None,None,3])
slice1 = tf.slice(image,[100,0,0],[300,-1,-1])

with tf.Session() as sess:
    result = sess.run(slice1,feed_dict={image:raw_image_data})
    print(result.shape)

plt.imshow(result)
plt.show()

