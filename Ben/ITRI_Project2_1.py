import numpy as np
import pandas as pd
import cv2
import functions.functions as fc
import functions.color_quanlification as cq

def roi(img):
    #gaussianblur
    blur = cv2.GaussianBlur(img,(5,5),0)
    #color quanlification
    cq_img=cq.color_quanlification(blur)
    #get middle color
    binary_img=cq.middle_color(cq_img)
    #morphology
    kernel = np.ones((3,3),np.uint8)
    opening=cv2.erode(binary_img,kernel,iterations=3)
    opening=cv2.dilate(opening,kernel,iterations=3)
    #opening = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
    #contuour
    image, contours, hierarchy = cv2.findContours(opening,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #draw,features and endpoint
    endpoint_img=fc.endpoint(contours,img)
    #stacking images side-by-side
    opening=cv2.cvtColor(opening,cv2.COLOR_GRAY2BGR)
    roi=np.hstack((opening,endpoint_img))
    
    #show
    cv2.imshow("blur",blur)
    cv2.imshow("cq",cq_img)
    cv2.imshow("binary",binary_img)
    cv2.imshow("endpoint",endpoint_img)
    
    return roi

#main
img1=cv2.imread("..\\training_img\\normal\\normal2.jpg")
roi_img1=roi(np.copy(img1))
#stacking images side-by-side
res=np.hstack((img1,roi_img1))
cv2.imshow("result",res)
cv2.imwrite("result2.jpg",res)
cv2.waitKey(0)