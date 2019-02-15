 # gmm_hist:  calculate Gaussian-Micture-Model on (image) statistics given the image histogram
 #  motivated by availability of histogram for othe purposes &
 #  the acceleration potential comparing to full list of image values
 # Example application: color based image segmentation
 #
 # USAGE: EXAMPLE: python run_gmm_hist.py -i https://c402277.ssl.cf1.rackcdn.com/photos/11552/images/hero_small/rsz_namibia_will_burrard_lucas_wwf_us_1.jpg?1462219623
 #
 # ToDo: Development steps:
 # 1. Get image, calc histogram
 # 2. K-means clustering on grey levels
 # 3. Test program: segment(im):   image --> hist --> cluster(K-means) --> classify = back prop. pixels to clusters --> Display as segmentation
 #
 # 4. * K-means clustering on histogram
 # --------------------
 # 5. K-means on RGB vlues (3D)
 # 6. Get RGB hist (quantize to <256^3 ?)
 # 6. K-means (RGB hist)
 # 7. GMM (RGB hist)
 # ---
 # 8. Auto k-clusters
 #   -  MLE
 #   -  Velassis adaptive

IM_URL =   'https://upload.wikimedia.org/wikipedia/he/2/24/Lenna.png'
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
import scipy
import scipy.signal as scipysig

from sklearn.mixture import GaussianMixture
import argparse


def main():
    #--- Parse params ---
    ap = argparse.ArgumentParser("segment image using GMM on histogram")
    ap.add_argument("-i", "--in", required=False, default = IM_URL, help="input imge file or URL")
    args = vars(ap.parse_args())


    params={'use_color':False}

    im_url = args['in']
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

    ar_hist, ar_val = np.histogram(img, density=True, bins=range(256+1)) # , range=None, normed=None, weights=None, density=None)

    plt.figure('Image')
    plt.plot(range(256), ar_hist)
    plt.title("hist(img)")
    plt.show()

    # ar_hist_smooth=scipysig.medfilt(ar_hist, 5)

    ker_d=5
    ar_hist_smooth=np.array([np.median(ar_hist[iVal:iVal+ker_d][ar_hist[iVal:iVal+ker_d]>0]) for iVal in range(256)])
    ar_hist_smooth[np.isnan(ar_hist_smooth)] = 0
    ar_hist_smooth = scipy.ndimage.filters.gaussian_filter(ar_hist_smooth, sigma=1, truncate =1)# .astype(int)
    sum_hist = ar_hist_smooth.sum()
    ar_hist_smooth = np.array([val/sum_hist for val in ar_hist_smooth])

    # ar_hist_smooth=scipy.median(ar_hist, 5)
    fig = plt.figure('histogram')
    plt.plot(range(256), ar_hist_smooth)
    plt.title("hist(img)")
    plt.show()
    # ---------- KMw
    # # ~6[sec]
    # kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(np.reshape(img,(img.size, 1)))
    # # >> > kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    #
    # cluster_centers = np.round(kmeans.cluster_centers_).astype(int)
    #
    # # GMM = GaussianMixture(n_components=num_clusters,
    # #                 covariance_type='full', max_iter=20, random_state=0).fit(np.reshape(img,(img.size, 1)))
    # # GMM.mean, .covariances_
    #
    # fig = plt.figure('histogram')
    # plt.plot(range(256), ar_hist_smooth, kmeans.cluster_centers_, 'b-', cluster_centers, ar_hist_smooth[cluster_centers], 'ro')
    # plt.title("hist(img)")
    # plt.show()

    #======== Hist K-Means ======
    def init(num_clusters, num_val, ar_hist_smooth):
        #  Init rand values from distribution in hist
        ar_val = np.array(range(num_val))
        ar_mean = np.sort(np.random.choice(ar_val, size=num_clusters, replace=False, p=ar_hist_smooth))
        # print(ar_mean)
        return ar_mean

    def maximization(num_val, ar_mean):
        # Maximization: assign samples to closest clusters
        # ar_min_dist: min dist from each sample to closest cluster center

        ar_val = np.array(range(num_val))
        ar_dist = scipy.spatial.distance.cdist(ar_val[:,np.newaxis], ar_mean[:,np.newaxis]) # , metric='euclidean', *args, **kwargs)[source]
        ar_min_dist = np.min(ar_dist, axis=1)
        mean_err = np.mean(ar_min_dist)
        ar_asign = np.argmin(ar_dist, axis=1)

        ar_pi = np.histogram(ar_asign, range(num_clusters+1), density=True, weights = ar_hist_smooth )
        return ar_asign, mean_err, ar_min_dist, ar_dist, ar_pi

    # ---- Expectation ----
    def expectation(num_clusters, num_val, ar_asign):
        #  - Find Mean of each cluster_centers

        ar_val = np.array(range(num_val))
        ar_mean=np.zeros((num_clusters,))
        for iCluster in range(num_clusters):
            ar_mean[iCluster] = np.mean(ar_val[ar_asign == iCluster])
        return ar_mean

    def gmm_hist(num_clusters, num_val, ar_hist_smooth):
        ar_mean = init(num_clusters, num_val, ar_hist_smooth)

        ar_err = np.zeros((num_iter, 1))
        for iIter in range(num_iter):
            ar_asign, mean_err, ar_min_dist, ar_dist, ar_pi = maximization(num_val, ar_mean)
            ar_mean = expectation(num_clusters, num_val, ar_asign)
            ar_err[iIter] = mean_err
            d_err = ar_err[iIter - 1] - ar_err[iIter]
            if iIter > params['min_iter'] and d_err < params['dErr']:
                print('early stop: iter = %d; dErr=%f; minErr = %f \n' % (iIter, d_err, ar_err[iIter]))
                break

        return ar_err[:iIter], ar_mean, ar_asign
    #======== Hist K-Means ======



    num_val = 256
    num_iter = 100
    num_clusters=5

    params={'min_iter':5, 'dErr':1e-4}
    ar_err, ar_mean, ar_asign = gmm_hist(num_clusters, num_val, ar_hist_smooth)

    # apply segmentation by LUT  mapping color to clusters
    im_segment = ar_asign[img]

    fig = plt.figure('segmentation')
    plt.imshow(im_segment)
    plt.show()

    fig = plt.figure('error rate')
    plt.plot(range(len(ar_err)), ar_err)
    # plt.plot(range(num_iter-1), np.diff(ar_err,axis=0))  # dErr
    plt.title("error(iter)")
    plt.show()



    # EM_iter Iterate & measure error (limit err, num_iter)
    # reinit & compare error


    # print(kmeans.labels_)
    # print(kmeans.predict([[31], [128]]))
    # print(kmeans.cluster_centers_)



if __name__ == "__main__":
    main()