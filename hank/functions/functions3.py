import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans


def Quanlification(img,color_num):
    (h, w) = img.shape[:2]

    # convert the image from the RGB color space to the L*a*b*
    # color space -- since we will be clustering using k-means
    # which is based on the euclidean distance, we'll use the
    # L*a*b* color space where the euclidean distance implies
    # perceptual meaning
    image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # reshape the image into a feature vector so that k-means
    # can be applied
    image = image.reshape(
        (image.shape[0] * image.shape[1], 3))  # (M, N, 3) image reshapes it into a (M x N, 3) feature vector.

    # apply k-means using the specified number of clusters and
    # then create the quantized image based on the predictions
    clt = MiniBatchKMeans(n_clusters=color_num)  # 指定叢數的數量
    labels = clt.fit_predict(image)  # 開始分群且“預測”原始圖像中每個像素的量化顏色。 通過確定輸入像素最接近哪個質心來處理該預測。
    quant = clt.cluster_centers_.astype("uint8")[labels]  # 創建分群後的圖片

    # reshape the feature vectors to images
    quant = quant.reshape((h, w, 3))
    image = image.reshape((h, w, 3))

    # convert from L*a*b* to RGB
    quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
    result=middle_color(quant)
    return  result

def Inhence(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGRA2GRAY)


    img = cv2.GaussianBlur(img, (3, 3), 1)
    #img = np.uint8(np.clip((1.27 * img), 0, 255))
    # Contrast Limited Adaptive Histogram Equalization

    #img = cv2.equalizeHist(img)   #photo 3 is too dark ,so need to use it
    kernel=np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    cl = cv2.filter2D(img, -1, kernel)


    gaussian = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)



    return gaussian


def middle_color(cq_img):    #效果還好
    img = cv2.cvtColor(cq_img, cv2.COLOR_BGRA2GRAY)
    maxv=np.amax(img)
    minv=np.amin(img)
    roi=np.empty(img.shape,img.dtype)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if(img[i,j]!=maxv and img[i,j]!=minv):
               roi[i,j]=255

    roi=cv2.cvtColor(roi,cv2.COLOR_GRAY2BGR)
    return roi




def threshold(img):

    img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

    '''
    im_floodfill = img.copy()

    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
    mor=cv2.morphologyEx(im_floodfill,cv2.MORPH_OPEN,kernel)
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = img.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    # Floodfill from point (0, 0)
    cv2.floodFill(mor, mask, (0, 0), 255)
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    # Combine the two images to get the foreground.
    mor = img | im_floodfill_inv
    '''


    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    #mor = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel1, iterations=3)
    #mor = cv2.morphologyEx(mor, cv2.MORPH_CLOSE, kernel2, iterations=1)
    mor=cv2.erode(img,kernel1,iterations=3)
    mor = cv2.dilate(mor, kernel1, iterations=2)



    mor = cv2.cvtColor(mor, cv2.COLOR_GRAY2BGR)
    return mor


def endpoint_attribution(cx,cy,endpoints):   # Get 4 end points's position
    position=[]

    for i in range(4):
     if endpoints[i][0]>cx and endpoints[i][1]>cy:
          position=np.append(position,'right down')
     elif endpoints[i][0]>cx and endpoints[i][1]<cy:
         position = np.append(position, 'right up')
     elif endpoints[i][0]<cx and endpoints[i][1]<cy:
         position = np.append(position, 'left up')
     else:
         position = np.append(position, 'left down')

    df = pd.DataFrame(position, columns=['endpoint_position'])
    return df






def draw(contours,img_color):
    img_color2 = img_color.copy()

    #  bone number
    # find 4 end points  and other features

    global num_roi

    num_roi = 0  # the num of effective bone's 4 end points
    df=pd.DataFrame()  #final output
    for cnt in contours:
        #end poont
        endpoints = np.zeros((4, 2), dtype=np.int32)


        #area and approx
        area = cv2.contourArea(cnt)
        approx = cv2.approxPolyDP(cnt,15, True)

        # pirimeter
        perimeter = cv2.arcLength(cnt, True)

        # get data  for pd
        if area>120 and np.shape(approx)[0]==4:


         cv2.drawContours(img_color2, [cnt], -1, (0, 0, 255), 1)
         # centroid
         M = cv2.moments(cnt)
         if M['m00'] == 0:
                continue
         cx = int(M['m10'] / M['m00'])
         cy = int(M['m01'] / M['m00'])
         cv2.putText(img_color2, str(num_roi+1), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
         num_roi += 1





         #get all the contours's points
         mask=np.zeros(img_color.shape,np.uint8)
         cv2.drawContours(mask,[cnt],0,255,thickness=cv2.FILLED)
         contourPoints=np.transpose(np.nonzero(mask))
         y = np.reshape(contourPoints[:, 0], (len(contourPoints[:, 0]), 1))
         x = np.reshape(contourPoints[:, 1], (len(contourPoints[:, 1]), 1))
         index = np.full(x.shape, num_roi)
         pixelpoints = np.hstack((index,x,y))
         df_points = pd.DataFrame(pixelpoints,columns=['index','x', 'y'])


         # get all the end points
         for i in range(np.shape(approx)[0]):
             cv2.circle(img_color2, (approx[i][0][0], approx[i][0][1]), 1, [0, 255, 0], 2)
             endpoints[i][0]=approx[i][0][0]
             endpoints[i][1]=approx[i][0][1]



         #panda
         df_feature=pd.DataFrame({'cx':[cx],'cy':[cy],'area':[area],'perimeter':[perimeter]})
         df_endpoints=pd.DataFrame(endpoints,columns=['endpoint_x','endpoint_y'])
         df_all = pd.concat([df_points, df_endpoints, endpoint_attribution(cx, cy, endpoints),df_feature], 1)  #combine in x-direction
         df = df.append(df_all)         #combine in y-direction

    print("找到%d個有效椎骨的座標" % num_roi)
    #print(df)
    df.to_csv("coordinate_forAllContours_points.csv")
    return img_color2




