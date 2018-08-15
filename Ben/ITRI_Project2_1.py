import numpy as np
import pandas as pd
import cv2
import image_processing as ip

def roi(img):
    #CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(img)
    #enhance
    enhance_img=ip.enhance(cl1)
    #otsu threshold
    ret,thr = cv2.threshold(enhance_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #filling hole
    filling_img=ip.filling_hole_tl(thr)
    #opening
    kernel = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(filling_img, cv2.MORPH_OPEN, kernel)
    #contuour
    image, contours, hierarchy = cv2.findContours(opening,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    roi = cv2.drawContours(img, contours, -1, (0,0,255), 1)
    #get features and draw
    df=ip.features(roi,contours)
    print(df)
    #get coordinate
    #ip.coordinate(roi,contours)
    #stacking images side-by-side
    opening=cv2.cvtColor(opening,cv2.COLOR_GRAY2BGR)
    roi=np.hstack((opening,roi))
    '''
    #show
    cv2.imshow("clahe",cl1)
    cv2.imshow("enchance",enhance_img)
    cv2.imshow("otsu",thr)
    cv2.imshow("filling_hole",filling_img)
    cv2.imshow("opening",opening)
    '''
    return roi

def test(img):
    '''
    roi=np.empty(img.shape,img.dtype)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            roi[i,j]=abs(img[i,j]-255)
    '''
    #enhance
    enhance_img=ip.enhance(img)
    #mask
    mask=cv2.inRange(enhance_img,0,63)
    #closing
    kernel = np.ones((3,3),np.uint8)
    dilation=cv2.dilate(mask,kernel)
    dilation=cv2.dilate(dilation,kernel)
    dilation=cv2.dilate(dilation,kernel)
    erosion=cv2.erode(dilation,kernel)
    erosion=cv2.erode(erosion,kernel)
    closing=cv2.erode(erosion,kernel)
    #closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    '''
    #CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(img)
    #gaussianblur
    blur = cv2.GaussianBlur(cl1,(5,5),0)
    #sobel
    soble_img=ip.sobel(blur)
    #otsu threshold
    ret,thr = cv2.threshold(soble_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #enhance
    enhance_img=ip.enhance(img)
    '''
    
    #show
    #cv2.imshow("cl1",cl1)
    cv2.imshow("enhance",enhance_img)
    cv2.imshow("mask",mask)
    cv2.imshow("closing",closing)
    #cv2.imshow("blur",blur)
    #cv2.imshow("sobel",soble_img)
    #cv2.imshow("thr",thr)

    roi=closing
    roi=cv2.cvtColor(roi,cv2.COLOR_GRAY2BGR)
    return roi

#main
img1=cv2.imread("..\\training_img\\normal\\normal2.jpg",0)
#roi_img1=roi(img1)
roi_img1=test(img1)
#stacking images side-by-side
img1=cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
res1=np.hstack((img1,roi_img1))
cv2.imshow("result1",res1)
cv2.imwrite("result.jpg",res1)
cv2.waitKey(0)