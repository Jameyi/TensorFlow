import tensorflow as tf
import numpy as np

def plot_clusters(all_samples,centroids,n_samples_per_cluster):
    import matplotlib.pyplot as plt

    colour = plt.cm.rainbow(np.linspace(0,1,len(centroids)))
    for i ,centroid in enumerate(centroids):
        samples = all_samples[i*n_samples_per_cluster:(i+1)*n_samples_per_cluster]
#samples[:,0] respect each sample's X position
#samples[:,1] respect each sample's Y position
#Also,centroid[0] is centroid's X posistion 
#centroid[1] is centroid's Y position
        plt.scatter(samples[:,0],samples[:,1],c=colour[i])
        plt.plot(centroid[0],centroid[1],markersize=35,marker="x",color='k',mew=10)
        plt.plot(centroid[0],centroid[1],markersize=30,marker="x",color='m',mew=5)
    plt.show()

# n_clusters=3,n_sample_per_cluster=500,n_features=2
# embiggen_factor=70,seed=seed)
def create_samples(n_clusters,n_sample_per_cluster,n_features,embiggen_factor,seed):
    np.random.seed(seed)
    slices = []
    centroids = []
    #Create samples for each cluster
    for i in range(n_clusters):
        samples = tf.random_normal((n_sample_per_cluster,n_features),mean=0.0,stddev=5.0,dtype=tf.float32,seed=seed,name="cluster_{}".format(i))
        
        current_centroid = (np.random.random((1,n_features))*embiggen_factor)-(embiggen_factor/2)

        centroids.append(current_centroid)
        samples += current_centroid
        slices.append(samples)

    #Create a big "samples" dataset
    samples = tf.concat(0,slices,name='samples')
    centroids = tf.concat(0,centroids,name='centroids')
        
    return centroids,samples

def choose_random_centroids(samples,n_clusters):
    n_samples = tf.shape(samples)[0]
    random_indices = tf.random_shuffle(tf.range(0,n_samples))
    begin = [0,]
    size = [n_clusters,]
    size[0] = n_clusters
    centroid_indices = tf.slice(random_indices,begin,size)
    initial_centroids = tf.gather(samples,centroid_indices)
    return initial_centroids

def assign_to_nearest(samples,centroids):
    expanded_vectors = tf.expand_dims(samples,0)
    expanded_centroids = tf.expand_dims(centroids,1)
    distances = tf.reduce_sum(tf.square(tf.sub(expanded_vectors,expanded_centroids)),2)
    mins = tf.argmin(distances,0)
    nearest_indices = mins
    return nearest_indices

def update_centroids(samples,nearest_indices,n_clusters):
    nearest_indices = tf.to_int32(nearest_indices)
    partitions = tf.dynamic_partition(samples,nearest_indices,n_clusters)
    new_centroids = tf.concat(0,[tf.expand_dims(tf.reduce_mean(partition,0),0) for partition in partitions])
    return new_centroids



