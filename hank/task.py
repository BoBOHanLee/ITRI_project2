import cv2
import numpy as np
import functions_auto as fn
img1 =cv2.imread('normal.jpg',0)

# I take off : manually adjust thresh and find initial roi
#however , contrast and morphology all are used by hands


#thresh
thresh=fn.Inhence_and_threshod(img1)
#Fill up
mor=fn.inner_fill(thresh)

image, contours, hierarchy = cv2.findContours(                          #find the outside contour
    mor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# for showing the picture
img_show=cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
mor_show = cv2.cvtColor(mor, cv2.COLOR_GRAY2BGR)

#draw the bine roi
fn.draw(contours,img_show)
fn.find_roi_coordinate(contours,img_show)



tmp1 = np.hstack((mor_show,img_show))

cv2.imshow("normal",tmp1)

k = cv2.waitKey(0)
if k==ord('ｑ'):
    cv2.destroyAllWindows()