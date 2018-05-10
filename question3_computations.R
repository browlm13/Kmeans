
# find the magnitues of each row of a matrix and represent as a vector
magnitudes_of_row_vectors <- function(M){
	magnitudes = sqrt(diag(M%*%t(M)))
	return(magnitudes)
}

# cosine similarities - row correspnds to document vector, column corresponds to centroid vector
cosine_similarities <- function(M,C){
	mag_M = magnitudes_of_row_vectors(M)
	mag_C = magnitudes_of_row_vectors(C)
	# dot product over multiplied magnitudes
	result = (M %*% t(C))/(mag_M %*% t(mag_C))
	return(result)
}

# sqrt squared distances - column correspnds to document vector, row corresponds to centroid vector
errors <- function(M,C){
	
	#replicate()
	#my.array<-array(0,dim=c(3,4,2)) # 2 is number of centroids (k)
	#my.array[,,1] = c(2,0,1)
	#my.array[,,2] = c(0,1,1)
	#t(my.array[,,1])
	#t(my.array[,,2])
	
	k = nrow(C)
	error_matrix = matrix(0,k,nrow(M))
	for (centroid_index in seq(1,k,1)){
		
		# find the sqrt(sqrd error) between centroid vector and each row vector
		C_subtract_i = t(replicate(nrow(M),C[centroid_index,]))
		diff_sqrd = (M - C_subtract_i) * (M - C_subtract_i)
		error_i = sqrt(rowSums(diff_sqrd))
		error_matrix[centroid_index,] <- error_i
	}
	
	# find min distance for each row
	#min_indices = apply(error_matrix, 2, which.min)

	#return(min_indices)
	error_matrix = t(error_matrix)
	return(error_matrix)
}

# find closet centroid for each document vector (euclidean distance)
error_closest_centroid <- function(M,C){
	error_matrix <- errors(M,C)
	
	# find min distance for each row
	min_indices = apply(error_matrix, 1, which.min)

	return(min_indices)
}
	
# find closet centroid for each document vector (cosine similarity)
closet_centroid <- function(M,C){
	
	# cosine similarities 
	A <- data.frame(cosine_similarities(M,C),check.names = FALSE)
	
	# find maximum column index (centroid) for each row (document vector)
	# this corresponds to the closest centroid index for each row
	max_indices = apply(A, 1, which.max)
	return(max_indices)
}

# display kmeans info
display_kmeans_info <-function(M,C,words,iteration){
	#
	# Display iteration information
	#
	cat(paste("\n\n\n\t\tKmeans Iteration:", iteration))
	cat("\n\n")
	
	# display document vector matrix
	print("document vector matrix:")
	DF_dvm <- data.frame(M)
	colnames(DF_dvm) <- words
	rownames(DF_dvm) <- sprintf("Doc[%s]:",seq(1:nrow(M)))
	print(DF_dvm)
	cat("\n")
	
	# display centroid vectors
	print(paste("centroid vector matrix iteration:", iteration))
	DF_cvm <- data.frame(C)
	colnames(DF_cvm) <- words
	rownames(DF_cvm) <- sprintf("Centroid[%s]:",seq(1:nrow(C)))
	print(DF_cvm)
	cat("\n")
	
	
	# display cosine similarity information
	DF_cs <- data.frame(cosine_similarities(M,C))
	colnames(DF_cs) <- sprintf("centroid %s",seq(1:nrow(C)))
	rownames(DF_cs) <- sprintf("Doc[%s]:",seq(1:nrow(M)))
	print("cosine similarities:")
	print(DF_cs)
	cat("\n")
	
	# display sqrt ( sqrd error) information
	DF_es <- data.frame(errors(M,C))
	rownames(DF_es) <- sprintf("Doc[%s]:",seq(1:nrow(M)))
	colnames(DF_es) <- sprintf("centroid %s:",seq(1:nrow(C)))
	print("sqrt(d^2) or errors:")
	print(DF_es)
	
}

# find new centroids by computing the average of its members
find_new_centroid_matrix <- function(M,C,words, iteration){
	
	# display iteration information
	display_kmeans_info(M,C,words,iteration)

	 # L, row centroid matrix - first column is the row index
	# second column is its corresponding centroid index
	#index_vector <- closet_centroid(M,C) # using cosine
	index_vector <- error_closest_centroid(M,C) # using eculidean distance
	L = cbind(seq(1,nrow(M),1),index_vector)
	
	# compute new centroid matrix
	new_C = C
	for (centroid in seq(1,nrow(C),1)){
		
  		# print(paste("Centroid:", centroid))
  		centroid_member_M_indices = subset(L, L[,2] == centroid)[,1]
  		  		
  		# compute new centroid from average (if there is more than one element)
  		if (length(centroid_member_M_indices) > 1){
  			# extract document vectors from m for each centroids members 
  			centroid_members = M[centroid_member_M_indices,]

			# compute vector average for new centroid
  			new_centroid = colSums(centroid_members) /length(centroid_member_M_indices)
  			
  			# set value of new centroid matrix
  			new_C[centroid,] = new_centroid  		
  		}
  		if (length(centroid_member_M_indices) == 1){
  			# extract the document vector from m for the centroid
  			# new centroid is the single document vector
  			# set value of new centroid matrix
  			new_C[centroid,] <- M[centroid_member_M_indices,]
  		}
  		
	}
	
	# return new centroid matrix
	return(new_C)
}

# normalize row vector matrix
normalize_row_vectors <- function(A){
	normalizer_A = sqrt(1/diag(A %*% t(A)))
	A_norm = A * normalizer_A
	return(A_norm)
}

# kmeans
kmeans <- function(M,C_init, words){
	
	# display initial info
	display_kmeans_info(M,C,words,"Initial Before Normalization")
	
	# normalize row vectors in M and C_init
	# for when you compute the average distance for the centroid
	#M <- normalize_row_vectors(M)
	#C_init <- normalize_row_vectors(C_init)
	
	
	count = 1
	prev_C = C_init
	new_C = find_new_centroid_matrix(M, prev_C, words, count)
	while (! identical(new_C, prev_C)){
		count = count + 1
		prev_C = new_C
		new_C = find_new_centroid_matrix(M, prev_C, words, count)
		
	}
	
	# display final k means iteration info
	display_kmeans_info(M,new_C,words,"final")
	
	return(new_C)
}

# run on question 3 data set
C = matrix(c(2,0,1,0,1,1),nrow=2, byrow=TRUE)
#C = matrix(c(1,0,0,1,1,1),nrow=2, byrow=TRUE)
M = matrix(c(2,0,1,1,1,0,0,1,1,0,0,2),nrow=4, byrow=TRUE)
#M = matrix(c(4,4,4,3,3,3,2,2,2,1,1,1),nrow=4, byrow=TRUE)
words = c("[go]", "[SMU]", "[Mustangs]")
kmeans(M, C, words)

#print(error_closest_centroid(M,C))