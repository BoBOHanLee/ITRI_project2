import cv2
import numpy as np
import hank.functions.functions3 as fn


#f6 use floodfill will success


image=cv2.imread("/Users/libohan/Documents/GitHub/ITRI_project2/training_img/unnormal/unnormal_f7_dark.jpg")




if np.shape(image)[0]!=512 and np.shape(image)[1]!=512 :   #photo's shape must be 512x512
    image = cv2.resize(image, (512, 512))

image_in=fn.Inhence(image)
image_quan=fn.Quanlification(image_in,3)
image_thresh=fn.threshold(image_quan)

image_cont=cv2.cvtColor(image_thresh,cv2.COLOR_BGRA2GRAY)
__, contours, hierarchy = cv2.findContours(                          #find the outside contour
    image_cont, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

image_draw=fn.draw(contours,image)



cv2.imshow("normal",np.hstack([image_draw,image_in,image_quan,image_thresh]))

k = cv2.waitKey(0)
if k==ord('ｑ'):
    cv2.destroyAllWindows()