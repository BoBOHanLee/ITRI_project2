# import the necessary packages
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import cv2

def color_quanlification(image):

    color_num=3    # n-clusters
    # load the image and grab its width and height
    (h, w) = image.shape[:2]

    # convert the image from the RGB color space to the L*a*b*
    # color space -- since we will be clustering using k-means
    # which is based on the euclidean distance, we'll use the
    # L*a*b* color space where the euclidean distance implies
    # perceptual meaning
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # reshape the image into a feature vector so that k-means
    # can be applied
    image = image.reshape((image.shape[0] * image.shape[1], 3))  #(M, N, 3) image reshapes it into a (M x N, 3) feature vector.

    # apply k-means using the specified number of clusters and
    # then create the quantized image based on the predictions
    clt = MiniBatchKMeans(n_clusters=color_num)   #指定叢數的數量
    labels = clt.fit_predict(image)               #開始分群且“預測”原始圖像中每個像素的量化顏色。 通過確定輸入像素最接近哪個質心來處理該預測。
    quant = clt.cluster_centers_.astype("uint8")[labels]   #創建分群後的圖片

    # reshape the feature vectors to images
    quant = quant.reshape((h, w, 3))
    # convert from L*a*b* to RGB
    quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
    quant = cv2.cvtColor(quant, cv2.COLOR_BGR2GRAY)

    return quant

def middle_color(cq_img):
    maxv=np.amax(cq_img)
    minv=np.amin(cq_img)
    roi=np.empty(cq_img.shape,cq_img.dtype)
    for i in range(cq_img.shape[0]):
        for j in range(cq_img.shape[1]):
            if(cq_img[i,j]!=maxv and cq_img[i,j]!=minv):
                roi[i,j]=255
    return roi