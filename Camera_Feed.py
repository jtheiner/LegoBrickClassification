import numpy as np
import cv2
import time
import numpy as np
import imutils
import argparse
import sys
#from getkeys import key_check
from matplotlib import pyplot as plt


def process(image_1,gray):
        
	# First Image
	frameDelta = cv2.absdiff(image_1, gray)
	thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
	thresh = cv2.dilate(thresh, None, iterations=2)
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	contours = imutils.grab_contours(cnts)
	return contours

def onclick(event):
        print("test")
def press(event):
    print('press', event.key)
    sys.stdout.flush()
    if event.key == 'x':
        sys.stdout.flush()
        cam.release()
        print("key")
   
def screen_record():
    cam = cv2.VideoCapture(0)
    image_1= None
    last_time = time.time()
    
    FIGSIZE = 6.0
    SPACING = 0.1
    rows = 2
    cols = 2
    subplot=(rows,cols,1)
    plt.figure(figsize=(FIGSIZE,FIGSIZE/cols*rows))
    while(True):
        ret, frame = cam.read()
        if frame is None:
            break
        cam_frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if image_1 is None:
            image_1 = gray
            continue
            

        print('loop took {} seconds'.format(time.time()-last_time))
        last_time = time.time()
        cnts = process(image_1,gray)
        image_1= gray
        biggest_contour =0
        for c in cnts:
            print(cv2.contourArea(c))
            if cv2.contourArea(c) > biggest_contour:
                biggest_contour = cv2.contourArea(c)
                (x, y, w, h) = cv2.boundingRect(c)
                continue
        cutout_frame = np.zeros(cam_frame.shape, dtype=np.uint8)
        blank =True
        if biggest_contour>0:
            cv2.rectangle(cam_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            print('Box x={} y={} w={} h={}'.format(x,y,w,h))
            if w*h > 10000:
                cutout_frame = cam_frame[y:y+h, x:x+w]
                blank =False
            
        
        #fig, ax = plt.subplots()
        plt.subplot(*subplot)
        #cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.axis('off')
        plt.title('previous_image')
        plt.imshow(image_1)
        plt.subplot(subplot[0],subplot[1],subplot[2]+1)
        plt.axis('off')
        plt.title('Processed')
        plt.imshow(gray)
        plt.subplot(subplot[0],subplot[1],subplot[2]+2)
        plt.title('Contour')
        plt.axis('off')
        plt.imshow(cam_frame)
        plt.subplot(subplot[0],subplot[1],subplot[2]+3)
        if blank == False:
                title = 'Cut out w='+str(w)+' h='+str(h)
                plt.title(title)
                plt.axis('off')
                plt.imshow(cutout_frame)
        plt.ion()
        #plt.clf()
        plt.show()
        plt.pause(1)
        plt.tight_layout()
        plt.subplots_adjust(wspace=SPACING, hspace=SPACING)
 
        #keys = key_check()

        # p pauses game and can get annoying.
        
        #if cv2.waitKey(25) & 0xFF == ord('q'):
        #    cam.release()
        #    break

if __name__ == '__main__':       
    screen_record()




