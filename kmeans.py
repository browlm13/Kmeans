import numpy as np
import pandas as pd

import pickle

def document_vector_matrix(df):

	#
	#	Read Document IDs and Create DocId to Row Map
	#

	# docIDs
	docIDs = []
	for did in df[['docID']].values.tolist():
		docIDs.append(did[0])

	# num rows in matrix 
	nrows = len(docIDs)

	# doc2row map
	rows = range(0, nrows)
	doc2row = dict(zip(docIDs, rows))

	#
	#	Read Document Texts and Create Unique Word to Column Map
	#

	# unique words
	corpus = ''
	document_texts = []
	for sentence in df[['document text']].values.tolist():
		corpus += sentence[0] + ' '
		document_texts.append(sentence[0].split(' '))
	unique_words = list(set(corpus.split(' ')))

	# filter lengths < 1 (TMP MESS)
	unique_words = [w for w in unique_words if len(w) > 0]
	for sentence in document_texts:
		sentence = [w for w in sentence if len(w) > 0]

	# num columns in matrix
	ncols = len(unique_words)

	# word2column map
	columns = range(0, ncols)
	word2column = dict(zip(unique_words, columns))

	#
	#	Create Document Vector Matrix
	#

	# create empty matrix
	M = np.zeros((nrows, ncols), dtype=np.int64)

	# fill matrix
	for row in range(0, nrows):
		for word in document_texts[row]:
			col = word2column[word]
			M[row,col] += 1


	# return M (matrix), doc2row (map), word2column (map)
	return M, doc2row, word2column

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
		matrix_centroid_vectors = update_centroids(vector_matrix, matrix_centroid_vectors)

		# break if centroid vectors are the same
		if np.array_equal(matrix_centroid_vectors, previous_centroids):
			return matrix_centroid_vectors, count

		previous_centroids = matrix_centroid_vectors

		count += 1

	return matrix_centroid_vectors, count

#
# Start
#

# read question 4 csv
doc = "Question_4.csv"
df = pd.read_csv(doc, header=0, dtype={'docID' : np.int64, 'document text':str}) 

# create document vector matrix
M, doc2row, word2column = document_vector_matrix(df)


# testing save and load M for consistant results


# save

pd.DataFrame(data=M).to_csv("M.csv", index=False)

output = open('doc2row.pkl', 'wb')
pickle.dump(doc2row, output)
output.close()

output = open('word2column.pkl', 'wb')
pickle.dump(word2column, output)
output.close()


# load

M = pd.read_csv("M.csv").values

pkl_file = open('doc2row.pkl', 'rb')
doc2row = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open('word2column.pkl', 'rb')
word2column = pickle.load(pkl_file)
pkl_file.close()

print(M)
print(word2column)


#
# Kmeans
#


# note uses row vector matrices
# testing 
num_trials = 10
for i in range(0,num_trials):
	print(kmeans(M, 2, 100))


