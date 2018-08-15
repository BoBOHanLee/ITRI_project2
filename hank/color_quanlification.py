# import the necessary packages
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import cv2
color_num=4    # n-clusters
def resize_photo(img,x_size,y_size):
    newImg = cv2.resize(img, None, fx=x_size, fy=y_size, interpolation=cv2.INTER_LINEAR)
    return newImg
# load the image and grab its width and height
img = cv2.imread("normal.jpg")
image = resize_photo(img,1.5,1.5)
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
image = image.reshape((h, w, 3))

# convert from L*a*b* to RGB
quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)

# display the images and wait for a keypress
cv2.imshow("image", np.hstack([image, quant]))
cv2.waitKey(0)
