"""

	kmeans


"""

import numpy as np
import pandas as pd

import pickle
import matplotlib.pyplot as plt
import random


def nearest_centroid(vector, matrix_centroid_vectors):

	# specs
	k = matrix_centroid_vectors.shape[0]
	dim = vector.shape[0]

	# find smallest Frobenius norm of row vectors in matrix_distance_vectors
	distance_vector_template = np.zeros(dim)
	matrix_distance_vectors = np.zeros(matrix_centroid_vectors.shape)

	# find distance between given vector and each centroid vector
	for centroid_vector_index in range(0,k):
		diff = np.subtract(vector, matrix_centroid_vectors[centroid_vector_index])
		matrix_distance_vectors[centroid_vector_index,:] = diff

	# find the index of the nearest centroid in matrix_centroid_vectors and return
	centroid_distances = np.linalg.norm(matrix_distance_vectors, axis=1)
	index_nearest_centroid = centroid_distances.argmin()
	return index_nearest_centroid

# returns vector with indeces of closest centroid for each row vector in vector matrix
def nearest_centroid_vector(vector_matrix, matrix_centroid_vectors):

	num_vectors = vector_matrix.shape[0] # num rows (row vectors)
	nc_vector = np.zeros(num_vectors)	 # nearest centroid vector

	for vector_index in range(0, num_vectors):
		vector = vector_matrix[vector_index]
		nc_vector[vector_index] = nearest_centroid(vector, matrix_centroid_vectors)

	return nc_vector

def update_centroids(vector_matrix, matrix_centroid_vectors):  

	k = matrix_centroid_vectors.shape[0]
	num_vectors = vector_matrix.shape[0] # num rows (row vectors)
	nc_vector = nearest_centroid_vector(vector_matrix, matrix_centroid_vectors) # nearest centroid vector

	# find new centroid centers
	for centroid_index in range(0, k):

		# find vectors with the current centroid as their closest centroid
		member_vector_indices = np.where(nc_vector == centroid_index)[0]
		member_vectors = vector_matrix[member_vector_indices]

		# find vector group's center
		updated_centroid_vector = np.mean(member_vectors,axis=0)

		# update matrix of centroid vectors
		matrix_centroid_vectors[centroid_index,:] = updated_centroid_vector


	# return updated centroids
	return matrix_centroid_vectors


def kmeans(vector_matrix, k, max_iters):

	dim = vector_matrix.shape[1]
	maximum = np.max(vector_matrix)
	minimum = np.min(vector_matrix)

	# initilize centroids randomly
	matrix_centroid_vectors = np.random.uniform(low=minimum, high=maximum, size=(k,dim))
	previous_centroids = matrix_centroid_vectors

	# initilize iteration count
	count = 1

	# run algorithm
	while (count <= max_iters):
		# show plot
		plot_clusters(vector_matrix, matrix_centroid_vectors)

		matrix_centroid_vectors = update_centroids(vector_matrix, matrix_centroid_vectors)

		# break if centroid vectors are the same
		#if np.array_equal(matrix_centroid_vectors, previous_centroids):
		#	return matrix_centroid_vectors, count

		previous_centroids = matrix_centroid_vectors

		count += 1

	return matrix_centroid_vectors, count


def plot_clusters(vector_matrix, matrix_centroid_vectors):

	# create plot
	fig = plt.figure()
	ax1 = fig.add_subplot(111)

	# specs
	k = matrix_centroid_vectors.shape[0]
	dim = matrix_centroid_vectors.shape[1]
	nc_vector = nearest_centroid_vector(vector_matrix, matrix_centroid_vectors)

	# 2d support only
	assert dim == 2

	# add current cluster members to plot
	for centroid_index in range(0, k):

		# find vectors with the current centroid as their closest centroid
		member_vector_indices = np.where(nc_vector == centroid_index)[0]

		cluster_members = vector_matrix[member_vector_indices]
		xs = cluster_members[:,0]
		ys = cluster_members[:,1]
		
		# add to plot
		cluster_label = "Cluster %s" % centroid_index
		color=np.random.rand(3,1)
		ax1.scatter(xs, ys, s=40, c=color, label=cluster_label)

		# plot centroid
		x = matrix_centroid_vectors[centroid_index,0]
		y = matrix_centroid_vectors[centroid_index,1]
		ax1.scatter(x, y, s=100, c=color, label=cluster_label)


	# show plot
	plt.show()


def generate_random_k_clusters(k, size, sd=3):
	xs, ys = [], []

	csize = int(size/k)
	r = size%k

	for i in range(0, k-1):
		centerx = random.randint(0, 100)
		centery = random.randint(0, 100)
		xs  += np.random.normal(loc=centerx, scale=sd, size=csize).tolist()
		ys  += np.random.normal(loc=centery, scale=sd, size=csize).tolist()

	centerx = random.randint(0, 100)
	centery = random.randint(0, 100)
	xs += np.random.normal(loc=centerx, scale=sd, size=csize+r).tolist()
	ys += np.random.normal(loc=centery, scale=sd, size=csize+r).tolist()

	# create row vector matrix
	m = list(zip(xs,ys))
	M = np.asarray(m)

	return M


#
# Kmeans
#

k = 3
M = generate_random_k_clusters(k,30)

max_iterations = 5
centroids, iters = kmeans(M, k, max_iterations)
