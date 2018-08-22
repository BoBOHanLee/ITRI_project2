import cv2
import numpy as np
import pandas as pd

def Inhence_and_threshod(img):
    # Contrast Limited Adaptive Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(14, 14))
    cl = clahe.apply(img)
    # lowpass filter
    gaussian = cv2.GaussianBlur(cl, (5, 5), 1)
    # THRESH
    __, th = cv2.threshold(cl, 0, 255, cv2.THRESH_BINARY++ cv2.THRESH_OTSU)

    return th

def inner_fill(th):
    # 接滿
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 1))
    mor = cv2.dilate(th, kernel)
    # floodfill external's area
    image1, contours, hierarchy = cv2.findContours(
        mor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mor, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
    # erode
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mor = cv2.erode(mor, kernel)
    #opening
    mor = cv2.morphologyEx(mor, cv2.MORPH_OPEN, kernel)

    return mor

def features(roi,contours):
    index=[]
    centroid_x=[]
    centroid_y=[]
    area=[]
    perimeter=[]
    aspect_ratio=[]
    i=0
    for cnt in contours:
        #centroid
        M=cv2.moments(cnt)
        cx=int(M['m10']/M['m00'])
        cy=int(M['m01']/M['m00'])
        centroid_x=np.append(centroid_x,cx)
        centroid_y=np.append(centroid_y,cy)
        #area
        area=np.append(area,cv2.contourArea(cnt))
        #perimeter
        perimeter=np.append(perimeter,cv2.arcLength(cnt,True))
        #aspect ratio
        x,y,w,h = cv2.boundingRect(cnt)
        aspect_ratio=np.append(aspect_ratio,float(w)/h)
        #index
        index=np.append(index,i)
        cv2.putText(roi,str(i),(cx,cy),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)
        i+=1

    df = pd.DataFrame({"Index":index,
    "Centroid_X":centroid_x,
    "Centroid_Y:":centroid_y,
    "Area":area,
    "Perimeter":perimeter,
    "Aspect_Ratio":aspect_ratio
    })
    df.to_csv("features.csv")
    return df

def coordinate(roi,contours):
    df= pd.DataFrame()
    i=0
    for cnt in contours:
        #get coordinate
        mask = np.zeros(roi.shape,np.uint8)
        cv2.drawContours(mask,[cnt],0,255,-1)
        pixelpoints = np.transpose(np.nonzero(mask))
        #create df
        y=np.reshape(pixelpoints[:,0],(len(pixelpoints[:,0]),1))
        x=np.reshape(pixelpoints[:,1],(len(pixelpoints[:,1]),1))
        index=np.full(x.shape,i)
        pixelpoints=np.hstack((x,y,index))
        df_points=pd.DataFrame(pixelpoints,columns=['x','y','i'])
        df=df.append(df_points)
        i+=1
    print(df)
    df.to_csv("coordinate.csv")