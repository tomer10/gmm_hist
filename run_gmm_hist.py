 # gmm_hist:  calculate Gaussian-Micture-Model on (image) statistics given the image histogram
 #  motivated by availability of histogram for othe purposes &
 #  the acceleration potential comparing to full list of image values
 # Example application: color based image segmentation
 #
 # ToDo: Development steps:
 # 1. Get image, calc histogram
 # 2. K-means clustering on grey levels
 # 3. Test program: segment(im):   image --> hist --> cluster(K-means) --> classify = back prop. pixels to clusters --> Display as segmentation
 #
 # 4. * K-means clustering on histogram
 # 5. K-means on RGB vlues (3D)
 # 6. Get RGB hist (quantize to <256^3 ?)
 # 6. K-means (RGB hist)
 # 7. GMM (RGB hist)
 # ---
 # 8. Auto k-clusters
 #   -  MLE
 #   -  Velassis adaptive

im_url =   'https://upload.wikimedia.org/wikipedia/he/2/24/Lenna.png'
  # https://cdn12.picryl.com/photo/2016/12/31/girl-portrait-face-people-8dd9eb-1024.jpg

# import urllib2 - https://stackoverflow.com/questions/22676/how-do-i-download-a-file-over-http-using-python
# response = urllib2.urlopen('http://www.example.com/')
# html = response.read()


import wget
import cv2 as cv2
# import matplotlib as plt
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import os.path as path


params={'use_color':False}

im_name=path.split(im_url)[1]
if not path.isfile(im_name):
    im_name = wget.download(im_url )

if params['use_color']:
    img = cv2.imread(im_name)[:,:,::-1] # ,mode='RGB')
else:
    img = cv2.imread(im_name, cv2.IMREAD_GRAYSCALE ) # cv2.CV_LOAD_IMAGE_GRAYSCALE ) # cv2.CV_LOAD_IMAGE_GRAYSCALE ) # cv2.CV_LOAD_IMAGE_GRAYSCALE)

# if not params['use_color']:
#     img = cv2.\
#         gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# %pylab inline
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# img=plt.image.imread('your_image.png')
imgplot = plt.imshow(img, cmap='gray')
plt.show()
# plt.closea

ar_hist, ar_val = np.histogram(img, bins=range(256+1)) # , range=None, normed=None, weights=None, density=None)

plt.plot(range(256), ar_hist)
plt.title("hist(img)")
plt.show()

# ---------- KMw
kmeans = KMeans(n_clusters=2, random_state=0).fit(img)
print(kmeans.labels_)

print(kmeans.predict([[31], [128]]))

print(kmeans.cluster_centers_)