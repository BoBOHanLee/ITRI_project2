import cv2
import numpy as np
import hank.functions.functions2 as fn2

# for photo 1
image=cv2.imread("/Users/libohan/Documents/GitHub/ITRI_project2/training_img/unnormal/1.jpg")
image_inh=fn2.Inhence(image)
#quant=fn2.Quanlification(image,color_num)    #This photo doesnt fit quant algorithm
th=fn2.threshold(image_inh)

th_gray=cv2.cvtColor(th,cv2.COLOR_BGRA2GRAY)
__, contours, hierarchy = cv2.findContours(                          #find the outside contour
    th_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)



cv2.imshow("normal",np.hstack([fn2.draw(contours,image), image_inh,th]))

k = cv2.waitKey(0)
if k==ord('ｑ'):
    cv2.destroyAllWindows()