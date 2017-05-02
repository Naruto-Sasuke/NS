#!/usr/bin/python
# params are better, or just read the data that has written into disk
import numpy as np
import argparse
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (preprocessing, manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)
from time import time 
from sklearn.cluster import KMeans                  
from sklearn.metrics.pairwise import cosine_distances
def new_euclidean_distances(X, Y = None, Y_norm_squared = None, squared = False):
    return cosine_distances(X,Y)


import sklearn.cluster

sklearn.cluster.k_means_.euclidean_distances = new_euclidean_distances 

#----------------------------------------------------------------------
# Scale and visualize the embedding vectors
# X is the data_ori with dimension_reduction
# kernels are the pointed-clusters.
def plot_embedding(X, labels,kernels,cluster_num, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    k_min, k_max = np.min(kernels, 0), np.max(kernels, 0)
    kernels = (kernels - k_min)/(k_max - k_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(labels[i]),
                 color=plt.cm.Set1(labels[i] / 100.),
                 fontdict={'weight': 'bold', 'size': 8})
    # print(kernels)
    # print(labels)
    # for j in range(cluster_num):
    #     plt.text(kernels[j, 0], kernels[j, 1], str(j),
    #              color=plt.cm.Set1(j/102.),
    #              fontdict={'weight':'bold','size':18})

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)   





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
        #normalize
        data_reshape = preprocessing.normalize(data_reshape,norm='l2')
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
    #kmeans.labels_ is representing the title of each data_shaped[i]
    

    labels_items = Counter(kmeans.labels_).items()
    for a in map(lambda x: x, labels_items):
        # get counts[i] the number of (i+1)th cluster points
        counts[a[0]] = a[1]
    print('counts sum:%d, num_pixels:%d' %(sum(counts), num_pixels_arr[i]))
    assert sum(counts) == num_pixels_arr[i],'points number unfit'
    print(counts)


    #visual t-sne
    t0 = time()
    k_pca = decomposition.TruncatedSVD(n_components=50).fit_transform(data_shaped[i])
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    k_tsne = tsne.fit_transform(k_pca)
    kernels_pca = decomposition.TruncatedSVD(n_components=50).fit_transform(kernels)
    kernels_redu = tsne.fit_transform(kernels_pca)
    plot_embedding(k_tsne, 
                labels,
                kernels_redu,
                cluster_nums[i],
               "t-SNE embedding of the digits (time %.2fs)" %
               (time() - t0))

    # plt.show()

    #check kernels,labels and counts
    np.save(('data/kernels_%d.npy' %(i+1)), kernels)
    np.save('data/counts_%d.npy' %(i+1), counts)
    np.save('data/labels_%d.npy' %(i+1), labels)

 