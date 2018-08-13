import numpy as np
import pandas as pd
import cv2

def sobel(img):
    x = cv2.Sobel(img,cv2.CV_16S,1,0)
    y = cv2.Sobel(img,cv2.CV_16S,0,1)
    absX = cv2.convertScaleAbs(x) 
    absY = cv2.convertScaleAbs(y)
    #dst = cv2.addWeighted(absX,0.5,absY,0.5,0)
    dst = cv2.bitwise_and(absX,absY)
    return dst

def enhance(img):
    #sharpen
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpen = cv2.filter2D(img, -1, kernel)
    #gaussianblur
    blur = cv2.GaussianBlur(sharpen,(5,5),0)
    return blur

def filling_hole_tl(img):
    im_floodfill = img.copy()
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w= img.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255)
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    # Combine the two images to get the foreground.
    filled_img = img | im_floodfill_inv
    return filled_img

def filling_hole_br(img):
    im_floodfill = img.copy()
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w= img.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (w-1,h-1), 255)
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    # Combine the two images to get the foreground.
    filled_img = img | im_floodfill_inv
    return filled_img

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