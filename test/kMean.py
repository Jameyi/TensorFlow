import tensorflow as tf
import numpy as np

from createsample import *

n_features = 2
n_clusters = 3
n_samples_per_cluster = 500
seed = 700
embiggen_factor = 70


data_centroids,samples = create_samples(n_clusters,n_samples_per_cluster,n_features,embiggen_factor,seed)

initial_centroids = choose_random_centroids(samples,n_clusters)

nearest_indices = assign_to_nearest(samples,initial_centroids)

updated_centroids = update_centroids(samples,nearest_indices,n_clusters)

model = tf.initialize_all_variables()

with tf.Session() as sess:
    sample_values = sess.run(samples)
  #  centroid_values = sess.run(centroids)
    updated_centroid_value = sess.run(updated_centroids)
    print(updated_centroid_value)

plot_clusters(sample_values,updated_centroid_value,n_samples_per_cluster)
