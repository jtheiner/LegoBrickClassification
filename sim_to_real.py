import cv2
import imutils
import numpy as np
import fnmatch
import os
import logging
import random

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
background ='/Users/petesmac/Documents/Machine Learning/LegoBrickClassification/Background1.JPG'
for root, dirnames, filenames in os.walk('/Users/petesmac/Documents/Machine Learning/DATA/LEGO-brick-images'):
    for filename in fnmatch.filter(filenames, '*.jpg'):
        print('File %s',root,filename)

        img = cv2.imread(root+'/'+filename)
        newx,newy,depth = img.shape #240,240 #new size (w,h)
        back_Grd = cv2.imread(background)
        bg_x,bg_y,depth =back_Grd.shape
        x=random.randint(0,bg_x-newx)
        y=random.randint(0,bg_y-newy)
        newimage = back_Grd[x:x+newx,y:y+newy]
        #print (newimage.shape)
        
        #newimage = cv2.resize(back_Grd,(newy,newx))
        
        #print (img.shape)
        
        height,width,depth = img.shape
        edges = cv2.Canny(img,80,200)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        

        #print (newimage.shape)

        cv2.imshow("original image",img)
        #cv2.imshow("resize image",edges)
        #Create Black image the same size as the original
        blank_image = np.zeros((img.shape), np.uint8)
        blank_image [:] = (255,255,255)      # (B, G, R)
        blank_image = cv2.cvtColor(blank_image, cv2.COLOR_BGR2GRAY)
        #Detect Diff between Orig. and Black Screen
        frameDelta = cv2.absdiff(blank_image, gray)
        ret,thresh2 = cv2.threshold(frameDelta, 5, 255, cv2.THRESH_BINARY)
        
        # dilate the thresholded image to fill in holes, then find contours
        # on thresholded image
        #thresh = cv2.dilate(thresh2, None, iterations=1)
        thresh=thresh2
        _, cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_LIST,
                cv2.CHAIN_APPROX_SIMPLE)
        
        # Now create a mask of logo and create its inverse mask also
        #Create a mask the same size as the original
        mask = np.zeros(img.shape, dtype=np.uint8)
        mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
        #cv2.drawContours(mask, cnts, 0, 255, -1) # Draw filled contour in mask
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,)*channel_count
        cv2.fillPoly(mask, cnts, ignore_mask_color)
        # apply the mask
        #cv2.imshow("Mask", mask)
        #Invert Mask
        mask_inv = cv2.bitwise_not(mask)
        #print(mask_inv.shape)
        #cv2.imshow("Extract Inv", mask_inv)
        # Now black-out the area of Part in Background Image
        img_bg = cv2.bitwise_and(newimage,newimage,mask = mask_inv)
        # Take only region of Part from Part image.
        img_fg = cv2.bitwise_and(img,img,mask = mask)
        # Put Part in ROI and modify the main image
##        cv2.imshow("Back Ground", img_bg)
##        cv2.moveWindow("Back Ground", 40,30)
##        cv2.imshow("Part ", img_fg)
##        cv2.moveWindow("Part", 40,130)
        img_bg = cv2.add(img_bg,img_fg)
##        cv2.imshow("Extract2", img1)
##        cv2.moveWindow("Extract2", 400,30)
        cv2.imshow("cut", img_bg)
        cv2.moveWindow("cut", 40,330)
        i=cv2.waitKey(33)

        # Now crop
        
##        if largest_contourV >0:
##            M = cv2.moments(largest_contour)
##            cX = int(M["m10"] / M["m00"])
##            cY = int(M["m01"] / M["m00"])
##            Width = frame.shape[1]
##            #print(Width)
##            # draw the contour and center of the shape on the image
##            #cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)
##            # compute the bounding box for the contour, draw it on the frame,
##            # and update the text
##            
##            (x, y, w, h) = cv2.boundingRect(largest_contour)
##            #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
##            crop_img = frame[y:y+h, x:x+w]
##            if (h > 240) | ( w> 240):
##                if (h >240):
##                    scaler = 240/h
##                if (w >240):
##                    scaler = 240/w
##                print (scaler)
##                print (crop_img.shape)
##                crop_img = cv2.resize(crop_img, (0,0),fx=scaler,fy=scaler)
##                print (crop_img.shape)
##            print(h,w)
        
               
        #cv2.imwrite(filename,crop_img)

