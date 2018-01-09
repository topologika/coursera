import numpy as np
import pandas
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import math as math
from skimage.io import imread
from skimage import img_as_float
from skimage import img_as_ubyte

image = imread('parrots.jpg')

# import pylab
# pylab.imshow(image)

image1=img_as_float(image)
n,m = image1.shape[:2]

X = image1.reshape(n*m,3)
n_clusters=11

kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=241).fit(X)

y = kmeans.labels_

Median_center=[]
for i in range(n_clusters):
    Median_center.append(np.median(X[y==i],axis=0))

Median=np.array(Median_center) 
Center = kmeans.cluster_centers_   

X_center = X.copy()
X_median = X.copy()
for i in range(n_clusters):
    X_center[y==i]=Center[i]
    X_median[y==i]=Median[i]
    
MAX=255

image_m = img_as_ubyte(X_median)
image_c = img_as_ubyte(X_center)
image_ = image.reshape(n*m,3)

MSE_c = ((image_-image_c)**2).sum()/image.size
PSNR_c = 10*np.log10(MAX**2/MSE_c)

MSE_m = ((image_-image_m)**2).sum()/image.size
PSNR_m = 10*np.log10(MAX**2/MSE_m)

print(n_clusters,PSNR_c,PSNR_m)


MAX=1
MSE_c = ((X-X_center)**2).sum()/X.size
PSNR_c = 10*np.log10(MAX**2/MSE_c)

MSE_m = ((X-X_median)**2).sum()/X.size
PSNR_m = 10*np.log10(MAX**2/MSE_m)


print(n_clusters,PSNR_c,PSNR_m)