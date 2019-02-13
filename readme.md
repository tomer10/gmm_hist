gmm_hist calculate Gaussian-Micture-Model on (image) statistics given the image histogram
motivated by availability of histogram for othe purposes &
 the acceleration potential comparing to full list of image values
Example application: color based image segmentation

 ToDo: Development steps:
 1. Get image, calc histogram
 2. K-means clustering on grey levels
 3. Test program: segment(im):   image --> hist --> cluster(K-means) --> classify = back prop. pixels to clusters --> Display as segmentation

 4. * K-means clustering on histogram
 5. K-means on RGB vlues (3D)
 6. Get RGB hist (quantize to <256^3 ?)
 6. K-means (RGB hist)
 7. GMM (RGB hist)
 ---
 8. Auto k-clusters
   -  MLE
   -  Velassis adaptive

  https://upload.wikimedia.org/wikipedia/he/2/24/Lenna.png
  https://cdn12.picryl.com/photo/2016/12/31/girl-portrait-face-people-8dd9eb-1024.jpg
