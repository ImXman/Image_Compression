# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 13:16:09 2019

@author: Yang Xu
"""

import cv2
import numpy as np
import pandas as pd
import seaborn as sns

###############################################################################
##read the image and flatten it into R,G,B 3-column matrix
img = cv2.imread('flowersm.ppm')
img_f=[]
for i in range(img.shape[2]):
    img_f.append(img[:,:,i].flatten())
img_f= np.asarray(img_f).T
img_f = img_f.astype(np.float32)

###############################################################################
##K-means clustering
def kmeans(img,k):
    minv = np.min(img)
    maxv = np.max(img)
    
    ##initialze centroids
    centroid  = np.random.randint(low=minv,high=maxv, size=(k, img.shape[1]))
    centroid = centroid.astype(np.float32)
    
    ##start iteration to cluster pixels
    cluster=np.repeat(0, img.shape[0]).tolist()
    old_cluster = np.random.randint(k, size=(img.shape[0])).tolist()
    while cluster != old_cluster:
        old_cluster = cluster.copy()
        dist=np.empty((img.shape[0],0))
        for i in range(centroid.shape[0]):
            d=img-centroid[i,:]
            d=np.linalg.norm(d, axis=1)
            dist=np.column_stack((dist,d))
        cluster=np.argmin(dist,axis=1).tolist()
        df = pd.DataFrame(np.column_stack((img,cluster)))
        cen = df.groupby([df.iloc[:,-1]], as_index=False).mean()
        ##update centroids
        for j in range(cen.shape[0]):
            centroid[int(cen.iloc[int(j),-1]),:]=cen.iloc[j,:-1].values
                
    return cluster, centroid

###############################################################################
##winner-takes-all
def wta(img,m,l=0.01,epoch=1000,batch=256):
    
    minv = np.min(img)
    maxv = np.max(img)
    
    ##initialze prototypes
    centroid  = np.random.randint(low=minv,high=maxv, size=(m, img.shape[1]))
    centroid = centroid.astype(np.float32)
        
    ##it's native implementation. Therefore, it doesn't the progression of loss
    ##Another problem is if wta reaches to the optimun after the whole epoch
    for i in range(epoch):
        index=np.random.randint(img.shape[0],size=batch)
        samples = img[index,:]
        dist=np.empty((samples.shape[0],0))
        for j in range(centroid.shape[0]):
            d=samples-centroid[j,:]
            d=np.linalg.norm(d, axis=1)
            dist=np.column_stack((dist,d))
        cluster=np.argmin(dist,axis=1).tolist()
        df = pd.DataFrame(np.column_stack((samples,cluster)))
        cen = df.groupby([df.iloc[:,-1]], as_index=False).mean()
        for j in range(cen.shape[0]):
            centroid[int(cen.iloc[int(j),-1]),:]+=l*(cen.iloc[j,:-1].values-\
                     centroid[int(cen.iloc[int(j),-1]),:])
    
    dist=np.empty((img.shape[0],0))
    for i in range(centroid.shape[0]):
        d=img-centroid[i,:]
        d=np.linalg.norm(d, axis=1)
        dist=np.column_stack((dist,d))
    cluster=np.argmin(dist,axis=1).tolist()
    
    return cluster, centroid

###############################################################################
##Kohonen map or self-organizing map
##this is naive implementation of SOM. 
##Therefore, it turns out to be inefficient
##Cautious: decay of learning rate is not considered here
def kohonen_map(img,k=256,l=0.01,epoch=1000,batch=256):
    
    net_size=int(np.sqrt(k))
    m,n = img.shape
    ##initialize the net
    minv = np.min(img)
    maxv = np.max(img)
    net  = np.random.randint(low=minv,high=maxv, size=(net_size,net_size,n))
    net = net.astype(np.float32)
    ##in this project, we use 5X5 gaussian kernal with sigma 1 
    ##to define the influence. Since it's hard coding here, if you want
    ##change influence, you may change those numbers (5,2,1,4) below
    x, y =5,5
    x, y = np.mgrid[-(x//2):(x//2)+1, -(y//2):(y//2)+1]
    g = np.exp(-((x/1)**2+(y/1)**2)/2)
    g = g[2:,2:]
    ##find BMU and update neuron weights
    for i in range(epoch):
        index=np.random.randint(img.shape[0],size=batch)
        samples = img[index,:]
        for j in range(samples.shape[0]):
            ##find BMU
            d=net-samples[j,:]
            d=np.linalg.norm(d, axis=2)  
            bmu = np.where(d==np.min(d))
            bmu = [bmu[0][0],bmu[1][0]]
            ##update neuron weight
            for a in range(net.shape[0]):
                for b in range(net.shape[1]):
                    if abs(a-bmu[0])<=2 and abs(b-bmu[1])<= 2:##4 is also hard coding
                        net[a,b,:]+=l*g[abs(a-bmu[0]),abs(b-bmu[1])]*\
                        (samples[j,:]-net[a,b,:])
                        
    ##assign organized neuron weights to each sample
    ##first flatten the net
    centroid=[]
    for i in range(net.shape[2]):
        centroid.append(net[:,:,i].flatten())
    centroid= np.asarray(centroid).T
    
    dist=np.empty((img.shape[0],0))
    for i in range(centroid.shape[0]):
        d=img-centroid[i,:]
        d=np.linalg.norm(d, axis=1)
        dist=np.column_stack((dist,d))
    cluster=np.argmin(dist,axis=1).tolist()
    
    return net, cluster, centroid

###############################################################################
##quality loss metrics
def quality_loss(ori_img,cp_img):
    
    m,n,d = ori_img.shape
    o=[]
    c=[]
    for i in range(d):
        o.append(ori_img[:,:,i].flatten())
        c.append(cp_img[:,:,i].flatten())
    o= np.asarray(o).T
    c= np.asarray(c).T
    loss=o-c
    loss=np.linalg.norm(loss, axis=1)
        
    return loss

###############################################################################
##compress image
##kmeans
cluster, centroid = kmeans(img=img_f,k=8)
##wta
cluster, centroid = wta(img=img_f,m=256)
##SOM
net, cluster, centroid = kohonen_map(img=img_f,k=256)

img_cp = np.zeros((img_f.shape[0],3))
for i in range(img_cp.shape[0]):
    img_cp[i,:]=centroid[cluster[i],:]
    
rgb=img_cp[:,0].reshape(img.shape[0],img.shape[1])
for i in range(1,3):
    rgb=np.dstack((rgb,img_cp[:,i].reshape(img.shape[0],img.shape[1])))
    

rgb=rgb.astype(np.uint8)
cv2.imwrite("256_compressed_som.jpg", rgb)
cv2.imwrite("orignal.jpg", img)

loss = quality_loss(img,rgb)
pic=sns.distplot(loss)
pic.get_figure().savefig("loss_256_som.jpeg",dpi=1200)
        
        
    
    