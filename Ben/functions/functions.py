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
    endpoint_img=np.copy(img_color)
    # bone ROI
    for i in range(0, len(contours)):
        area = cv2.contourArea(contours[i])
        x, y, w, h = cv2.boundingRect(contours[i])
        aspect_ratio = float(w) / h
        rect_area = w * h
        extent = float(area) / rect_area
        #delete error area
        
        if area>120:
           cv2.drawContours(img_color, [contours[i]], -1, (0, 0, 255), 1)

    #  bone number
    # find 4 end points  and other features
    num_roi = 0  # the num of effective bone's 4 end points
    df=pd.DataFrame()  #final output
    for cnt in contours:
        #end poont
        endpoints = np.zeros((4, 2), dtype=np.int32)
        #area and approx
        area = cv2.contourArea(cnt)
        approx = cv2.approxPolyDP(cnt, 30, True)
        # pirimeter
        perimeter = cv2.arcLength(cnt, True)
        # get data  for pd
        if area>120 and np.shape(approx)[0]==4:
            # centroid
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                    continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.putText(img_color, str(num_roi+1), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
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
                cv2.circle(img_color, (approx[i][0][0], approx[i][0][1]), 1, [0, 255, 0], 2)
                cv2.circle(endpoint_img, (approx[i][0][0], approx[i][0][1]), 1, [0, 255, 0], 2)
                endpoints[i][0]=approx[i][0][0]
                endpoints[i][1]=approx[i][0][1]

            #panda
            df_feature=pd.DataFrame({'cx':[cx],'cy':[cy],'area':[area],'perimeter':[perimeter]})
            df_endpoints=pd.DataFrame(endpoints,columns=['endpoint_x','endpoint_y'])
            df_all = pd.concat([df_points, df_endpoints, endpoint_attribution(cx, cy, endpoints),df_feature], 1)  #combine in x-direction
            df = df.append(df_all)         #combine in y-direction

    print("找到%d個有效椎骨的座標"%num_roi)
    #print(df)
    df.to_csv("coordinate_forAllContours_points.csv")
    return endpoint_img

def sobel(img):
    x = cv2.Sobel(img,cv2.CV_16S,1,0)
    y = cv2.Sobel(img,cv2.CV_16S,0,1)
    absX = cv2.convertScaleAbs(x) 
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX,0.5,absY,0.5,0)
    #dst = cv2.bitwise_and(absX,absY)
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



