import cv2
import numpy as np
import hank.functions.functions3 as fn





image=cv2.imread("/Users/libohan/Documents/GitHub/ITRI_project2/training_img/normal/normal_f8.jpg")


if np.shape(image)[0]!=512 and np.shape(image)[1]!=512 :   #photo's shape must be 512x512
    image = cv2.resize(image, (512, 512))

cv2.imwrite('normal9_before.jpg',image)
image_in=fn.Inhence(image)
image_quan=fn.Quanlification(image_in,3)
image_mor=fn.mor(image_quan)

image_cont=cv2.cvtColor(image_mor,cv2.COLOR_BGRA2GRAY)
__, contours, hierarchy = cv2.findContours(                          #find the outside contour
    image_cont, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

image_draw=fn.draw(contours,image)
temp=np.hstack([image_draw,image_in])
temp2=np.hstack([image_quan,image_mor])

cv2.imwrite('normal9.jpg',image_draw)
#cv2.imshow("unnormal",np.vstack([temp,temp2]))
cv2.imshow("unnormal",np.hstack([temp,temp2]))

k = cv2.waitKey(0)
if k==ord('ｑ'):
    cv2.destroyAllWindows()