import cv2
import numpy as np
import pandas as pd

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
    num_roi = 0  # the num of effective bone's 4 end points
    df=pd.DataFrame()  #final output
    # find 4 end points  and other features
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h
        rect_area = w * h
        extent = float(area) / rect_area
        #delete error area
        if area>120:
           cv2.drawContours(img_color, [cnt], -1, (0, 0, 255), 1)
        
        #end poont
        endpoints = np.zeros((4, 2), dtype=np.int32)
        #approx
        approx = cv2.approxPolyDP(cnt, 8, True)
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

def endpoint(contours,img):
    img_color=np.copy(img)
    img_size=img.shape[0]*img.shape[1]
    num_roi = 0  # the num of effective bone's 4 end points
    df=pd.DataFrame()  #final output
    for cnt in contours:
        #find 4 corner(tl,bl,br,tr) of minAreaRect
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        corners=np.int32(box)
        #delete error area
        area=cv2.contourArea(cnt)
        rect_area=rect[1][0]*rect[1][1]
        rel_area=area/img_size
        extent=area/rect_area
        aspect_ratio=rect[1][0]/rect[1][1]
        if(rel_area > 1/20 or rel_area < 1/200 or extent < 0.6 or aspect_ratio > 1.5 or aspect_ratio < 0.5):
            continue
        #initialize end points
        num_roi += 1
        M=cv2.moments(cnt)
        cx=int(M['m10']/M['m00'])
        cy=int(M['m01']/M['m00'])
        endpoints=np.full(corners.shape,[cx,cy])
        #calculate end points
        for point in cnt:
            for i in range(0,4):
                if(cv2.norm(corners[i],point[0],cv2.NORM_L2)<cv2.norm(corners[i],endpoints[i],cv2.NORM_L2)):
                    endpoints[i]=point[0]
                else:
                    endpoints[i]=endpoints[i]
        #draw end points
        cv2.drawContours(img_color, [cnt], -1, (0, 0, 255), 1)
        cv2.putText(img_color, str(num_roi), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        for i in range(0,4):
            cv2.circle(img_color, (endpoints[i][0],endpoints[i][1]), 1, [0, 255, 0], 2)
        #get all features
        df=features(img_color,cnt,num_roi,endpoints,df)

    df.to_csv("all_features.csv")
    return img_color

def features(img,cnt,num_roi,endpoints,df):
    #features
    M=cv2.moments(cnt)
    cx=int(M['m10']/M['m00'])
    cy=int(M['m01']/M['m00'])
    area=cv2.contourArea(cnt)
    perimeter=cv2.arcLength(cnt,True)
    #get all the contours's points
    mask=np.zeros(img.shape,np.uint8)
    cv2.drawContours(mask,[cnt],0,255,thickness=cv2.FILLED)
    contourPoints=np.transpose(np.nonzero(mask))
    y = np.reshape(contourPoints[:, 0], (len(contourPoints[:, 0]), 1))
    x = np.reshape(contourPoints[:, 1], (len(contourPoints[:, 1]), 1))
    index = np.full(x.shape, num_roi)
    pixelpoints = np.hstack((index,x,y))
    df_points = pd.DataFrame(pixelpoints,columns=['index','x', 'y'])
    #panda
    df_feature=pd.DataFrame({'cx':[cx],'cy':[cy],'area':[area],'perimeter':[perimeter]})
    df_endpoints=pd.DataFrame(endpoints,columns=['endpoint_x','endpoint_y'])
    #combine in x-direction
    df_all = pd.concat([df_points, df_endpoints, endpoint_attribution(cx, cy, endpoints),df_feature], 1)
    #combine in y-direction
    df = df.append(df_all)        
    return df