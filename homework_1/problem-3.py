from sys import argv
# import modules
import matplotlib.pyplot as plt
from matplotlib import style 
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

def load_data():
    # read clust_data
    df = pd.read_csv('clust_data.csv')
    #save data values as a variable
    data = df.values
    # view the first five rows of df
    df.head()
    return data

def run_a_to_c(data):
	#Question (a)
	print("-------Question 3(a)------")
	# apply k-means to the data 15 times
	K = 15
	inertia = np.zeros(15)
	for i in range(1,K+1):
	    # initialize the model
	    kmeans = KMeans(n_clusters=i, random_state=0, n_init=10)
	    # fit the data
	    kmeans.fit(data)
	    assignments = kmeans.predict(data)
	    # store inertia in inertia array
	    centers = np.ndarray((i, data.shape[-1]))
	    for cluster in range(i):
	        #update centers
	        centers[cluster, :] = data[assignments == cluster].mean(0)
	        inertia[i-1] += ((data[assignments == cluster] - centers[cluster]) ** 2).sum()#distance sum of square
	# plot inertia against the number of centers
	plt.plot(range(1,K+1),inertia,'-x')
	plt.savefig("wy818_elbowcurve.jpg")
	print("The elbow point occurs at k=4. \nHence, 4 clusters should be used for this data.")

	#Question (b)
	print("-------Question 3(b)------")
	# reapply k-means with the chosen number of centers
	kmeans = KMeans(n_clusters = 4, random_state=0, n_init=10)
	kmeans.fit(data)
	# count observations in each cluster
	assignments = kmeans.predict(data)
	cluster_id, cluster_count= np.unique(assignments, return_counts=True)
	# calculate inertia
	inertia = 0
	for cluster in range(4):
	    centers = data[assignments == cluster].mean(0)
	    inertia += ((data[assignments==cluster] - centers) ** 2).sum()
	# print out the results
	for i in range(len(cluster_count)):
		print("Cluster %d has %d observations."%(i+1, cluster_count[i]))
	print("Value of inertia is: %f"%(inertia))


	#Question (c)
	print("-------Question 3(c)------")
	# visualize the data
	plt.figure()
	plt.scatter(data[:,0], data[:,1], c=assignments)
	plt.savefig("wy818_scatterplot.jpg")
	print("From the graph, it is not a good clustering.\nWe are only using the first 2 variables, \nhence the scatter plot might not be reliable.")

if __name__=="__main__":
    if argv[1] == "run":
        run_a_to_c(load_data())	







