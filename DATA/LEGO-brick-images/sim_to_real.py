import cv2
import imutils
import numpy as np
import fnmatch
import os
import logging

matches = []
verbose = True

if verbose:
    level = logging.DEBUG
    log_filename = os.path.join('./', 'logs', 'resize' + '.log')
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    logging.basicConfig(filename=log_filename, level=level, filemode='w', format=format)
else:
    logging.basicConfig(level=level, filemode='w', format=format)
    

logging.getLogger().addHandler(logging.StreamHandler())
directory = '/Users/petesmac/Documents/Machine Learning/Data/LEGO-brick-images/'
pattern = '*.jpg'
for root, dirs, files in os.walk(directory):
    for basename in files:
        if fnmatch.fnmatch(basename, pattern):
            filename = os.path.join(root, basename)
           
            
            print('File ',filename)
            img = cv2.imread(filename)
            frame=img
            print (img.shape)
            newx,newy = 240,240 #new size (w,h)
            height,width,depth = img.shape
            edges = cv2.Canny(img,90,200)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.GaussianBlur(img, (21, 21), 0)
            #newimage = cv2.resize(img,(newx,newy))

            #print (newimage.shape)

            cv2.imshow("original image",img)
            cv2.imshow("resize image",edges)

            blank_image = np.zeros((img.shape), np.uint8)
            blank_image [:] = (255,255,255)      # (B, G, R)
            blank_image = cv2.cvtColor(blank_image, cv2.COLOR_BGR2GRAY)
            frameDelta = cv2.absdiff(blank_image, gray)
            thresh = cv2.threshold(frameDelta, 15, 255, cv2.THRESH_BINARY)[1]

            # dilate the thresholded image to fill in holes, then find contours
            # on thresholded image
            thresh = cv2.dilate(thresh, None, iterations=5)
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            # loop over the contours
            largest_contour = 0
            largest_contourV = 0
            for c in cnts:
                    # if the contour is too small, ignore it
                    print(cv2.contourArea(c))
                    if cv2.contourArea(c) > largest_contourV:
                            #print(type(largest_contourV))
                            largest_contour = c
                            largest_contourV = cv2.contourArea(c) 
                            continue

            #print(type(largest_contour))

            if largest_contourV >0:
                M = cv2.moments(largest_contour)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                Width = frame.shape[1]
                #print(Width)
                # draw the contour and center of the shape on the image
                #cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)
                # compute the bounding box for the contour, draw it on the frame,
                # and update the text
                
                (x, y, w, h) = cv2.boundingRect(largest_contour)
                #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                crop_img = frame[y:y+h, x:x+w]
                if (h > 240) | ( w> 240):
                    if (h >240):
                        scaler = 240/h
                    if (w >240):
                        scaler = 240/w
                    print (scaler)
                    print (crop_img.shape)
                    crop_img = cv2.resize(crop_img, (0,0),fx=scaler,fy=scaler)
                    print (crop_img.shape)
                print(h,w)
                cv2.imshow("cropped", crop_img)
                   
            cv2.imwrite(filename,crop_img)

