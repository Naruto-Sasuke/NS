#!/usr/bin/python
# params are better, or just read the data that has written into disk
from sklearn.cluster import KMeans
import numpy as np
import argparse
from sklearn import preprocessing
from collections import Counter

parser = argparse.ArgumentParser()

parser.add_argument("--cluster_nums", action = "append", help="the number of clusters")
parser.add_argument("--style_layers",help="the number of style layers",type = int)
parser.add_argument("--cluster_method",help="the method of clustering")
parser.add_argument("--image_num",help="the number of images",type=int)

args = parser.parse_args()
#deal with cluster_nums
cluster_nums_arg = args.cluster_nums[0].split(',')
cluster_nums = []
print(len(cluster_nums_arg))
for i in range(len(cluster_nums_arg)):
    cluster_nums.append(int(cluster_nums_arg[i]))

cluster_type = args.cluster_method
nS = args.style_layers
nP = args.image_num

print(cluster_type)

#load all data
data = []
for i in range(nS):
    for j in range(nP):
        print('loading s%d_p%d.npy to data[%d]' %(i+1,j+1,i*nP+j))
        data_tmp = np.load("data/s%d_p%d.npy" %(i+1,j+1)) 
        data.append(data_tmp)

#normalize data
data = preprocessing.normalize(data,norm='l2')


#integrate data according to a certain style layer
num_pixels_arr = []
# data_shaped[i] stored the data of all points in (i+1)th layer 
# so that, data_shaped[i] is the value of NUM*CHANNEL
# NUM means the number of all points in (i+1)th layer
data_shaped = [] 
for i in range(nS):
    num_pixels = 0
    for j in range(nP):
        data_tmp = data[i*nP+j]
        data_reshape = data_tmp.reshape(data_tmp.shape[0],-1).T
        #check data
        print('points:%d, channel:%d' %(data_reshape.shape[0], data_reshape.shape[1]))
        num_pixels_tmp = data_reshape.shape[0]
        if j != 0:
            data_shaped[i] = np.vstack((data_shaped[i], data_reshape))
        else:
            data_shaped.append(data_reshape)
        num_pixels += num_pixels_tmp
    num_pixels_arr.append(num_pixels)
    print('style:%d, points total:%d' %(i+1,num_pixels))

#clustering
'''It's for the single style layer.'''
"""Saved kernels_1.npy stores the first style layer's kernels' data"""

for i in range(nS):
    print('clustering layer %d' %(i+1))
    kernels = np.array([]) #CLUSTER_NUMS[i] * CHANNEL
    labels = np.array([])  #POINTS_NUMS[i], each number indicates the cluster it belonging to.
    counts = np.zeros(cluster_nums[i])  #CLUSTER_NUM, each value indicates how many points it has
    kmeans = KMeans(n_clusters=cluster_nums[i],random_state = 0).fit(data_shaped[i])
    kernels = kmeans.cluster_centers_
    labels = kmeans.labels_ + 1 # starting from 1 to fit lua
    
    labels_items = Counter(kmeans.labels_).items()
    for a in map(lambda x: x, labels_items):
        # get counts[i] the number of (i+1)th cluster points
        counts[a[0]] = a[1]
    print('counts sum:%d, num_pixels:%d' %(sum(counts), num_pixels_arr[i]))
    assert sum(counts) == num_pixels_arr[i],'points number unfit'
    #check kernels,labels and counts
    np.save(('data/kernels_%d.npy' %(i+1)), kernels)
    np.save('data/counts_%d.npy' %(i+1), counts)
    np.save('data/labels_%d.npy' %(i+1), labels)


