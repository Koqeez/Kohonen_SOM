import numpy as np, numpy.random

from scipy.spatial import distance
np.set_printoptions(suppress=True)

k = 2
p = 0 
alpha = 0.7

X = np.array([
	[1,1,0,0],
	[0,1,0,0],
	[0,0,1,0],
	[0,0,1,1]])
n = len(X)
d = len(X[0])
addZeros = np.zeros((n,1))
X = np.append(X,addZeros, axis = 1)
print("The some algorithm: \n")
print("The traning data: \n",X)
print("\nTotal number of data: ",n)
print("Total number of features: ",d)
print("Total number of Clusters: ",k)

C = np.zeros((k,d+1))

weight = np.random.rand(n,k)
print("\nThe initial wieght: \n", np.round(weight,2))
for it in range(100):
	for i in range(n):
		distMin = 999999
		for j in range(k):
			dist = np.square(distance.euclidean(weight[:,j], X[i,0:d]))
			if distMin > dist:
				distMin = dist
				jMin = j
		weight[:,jMin] = weight[:,jMin]*(1-alpha) + alpha*X[i,0:d]
	alpha = 0.5*alpha
print("\nThe final weight: \n",np.round(weight,4))

for i in range(n):
	cNumber = np.where(weight[i] == np.amax(weight[i]))
	X[i,d] = cNumber[0]

print("\nThe data with cluster number: \n",X)