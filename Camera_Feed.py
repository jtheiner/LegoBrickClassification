import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
from PIL import Image
import time
import numpy as np
import imutils
import argparse
import sys
import os
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, RMSprop
#from getkeys import key_check

from matplotlib import pyplot as plt


def camera_info(video_in):
    width = int(video_in.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
    length = int(video_in.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video_in.get(5))
    print(' Height {} x Width {} FPS {} FPS {}'.format(height,width,length,fps))

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
    cam.set(cv2.CAP_PROP_FPS, 100)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera_info(cam)
    while(True):
        ret, frame = cam.read()
        if frame is None:
            print("No Camera")
            break
        if ret is False:
            print("No Picture")
            break
        cam_frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if image_1 is None:
            image_1 = gray
            continue
            
        frameDelta = cv2.absdiff(image_1, gray)
       
        #cv2.imshow("Frame Delta", frameDelta)
        
        #print(np.argmax(frameDelta, axis=-1))
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
        diff =np.concatenate(thresh).sum()
        print('loop took {} seconds'.format(time.time()-last_time))
        last_time = time.time()
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
                break
        if diff >100000:
                thresh = cv2.dilate(thresh, None, iterations=2)
                cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours = imutils.grab_contours(cnts)

                image_1= gray
                biggest_contour =0
                for c in  contours:
                    #print(cv2.contourArea(c))
                    if cv2.contourArea(c) > biggest_contour:
                        biggest_contour = cv2.contourArea(c)
                        (x, y, w, h) = cv2.boundingRect(c)
                        continue
                cutout_frame = np.zeros(cam_frame.shape, dtype=np.uint8)
                blank =True
                if biggest_contour>0:
                    cv2.rectangle(cam_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    print('Box x={} y={} w={} h={}'.format(x,y,w,h))
                    if w*h > 2000:
                        cutout_frame = cam_frame[y:y+h, x:x+w]
                        blank =False
                    
                
                #fig, ax = plt.subplots()
                #plt.subplot(*subplot)
                #cid = fig.canvas.mpl_connect('button_press_event', onclick)
                #plt.axis('off')
                #plt.title('previous_image')
                #plt.imshow(image_1)
                #plt.subplot(subplot[0],subplot[1],subplot[2]+1)
                #plt.axis('off')
                #plt.title('Processed')
                #plt.imshow(gray)
                #plt.subplot(subplot[0],subplot[1],subplot[2]+2)
                #plt.title('Contour')
                #plt.axis('off')
                #plt.imshow(cam_frame)
                #plt.subplot(subplot[0],subplot[1],subplot[2]+3)
                if blank == False:
                        title = 'Cut out w='+str(w)+' h='+str(h)
                        #plt.title(title)
                        #plt.axis('off')
                        #plt.imshow(cutout_frame)
                        cv2.imshow(title, cutout_frame)
                #plt.ion()
                #plt.clf()
                #plt.show()
                #plt.pause(1)
                #plt.tight_layout()
                #plt.subplots_adjust(wspace=SPACING, hspace=SPACING)
         
                #keys = key_check()

                # p pauses game and can get annoying.
                
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cam.release()
                    break

def get_strategy():

    try: # detect TPUs
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver() # TPU detection
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
    except ValueError: # no TPU found, detect GPUs
        #strategy = tf.distribute.MirroredStrategy() # for GPU or multi-GPU machines
        strategy = tf.distribute.get_strategy() # default strategy that works on CPU and single GPU
        #strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy() # for clusters of multi-GPU machines
    print("Number of accelerators: ", strategy.num_replicas_in_sync)
    return strategy

def load_model():
    strategy = get_strategy()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model_file = os.path.join (dir_path , "Brick_Rec")
    model = tf.keras.models.load_model(model_file)
    model.summary()
    return model
   
##    if os.path.isfile(model_file):
##        
##        new_model.summary()
##    else:
##        print(model_file)
##        print("Model does not exist")
##        exit()


        
def file_list():
    for dirname, _, filenames in os.walk(dir_path):
        for filename in filenames:
            print(os.path.join(dirname, filename))
            
if __name__ == '__main__':
    FIGSIZE = 6.0
    SPACING = 0.1
    rows = 2
    cols = 2
    cam = cv2.VideoCapture(0)
    image_1= None
    last_time = time.time()
    #cam.set(cv2.CAP_PROP_FPS, 100)
    #cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    #cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    #camera_info(cam)
    ret, frame = cam.read()
    if frame is None:
        print("No Camera")
        exit(1)
    #subplot=(rows,cols,1)
    #plt.figure(figsize=(FIGSIZE,FIGSIZE/cols*rows))
    print(tf.__version__)
    path = os.path.dirname(os.path.realpath(__file__))
    dataset_path =  os.path.join(path,'DATA','LEGO-brick-images','Brick_List.csv')
    df = pd.read_csv(dataset_path, skipinitialspace=True, skip_blank_lines=True,encoding='utf-8', index_col='Brick')
    Bricks =  [( str(f)) for f in df.index]
    print(len(Bricks))
    model = load_model()
    #sou_path = os.path.join(path,'DATA','archive')
    #filenames = tf.io.gfile.glob(sou_path+'/*/*.png')
    
    ##for file_path in filenames:
    while(True):
        ret, frame = cam.read()
        if ret is False:
            print("No Picture")
            break
        cam_frame = frame
        gray = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if image_1 is None:
            image_1 = gray
            continue
            
        frameDelta = cv2.absdiff(image_1, gray)
       
        #cv2.imshow("Frame Delta", frameDelta)
        
        #print(np.argmax(frameDelta, axis=-1))
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
        diff =np.concatenate(thresh).sum()
        print('loop took {} seconds'.format(time.time()-last_time))
        last_time = time.time()
        cutout_frame = cam_frame
        cam_frame = cutout_frame
        image = cutout_frame
        key = cv2.waitKey(1) & 0xFF
        blank =True
        if key == ord("q"):
                break
        if diff >100:
                thresh = cv2.dilate(thresh, None, iterations=2)
                cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours = imutils.grab_contours(cnts)

                image_1= gray
                biggest_contour =0
                for c in  contours:
                    #print(cv2.contourArea(c))
                    if cv2.contourArea(c) > biggest_contour:
                        biggest_contour = cv2.contourArea(c)
                        (x, y, w, h) = cv2.boundingRect(c)
                        continue
                
                blank =True
                if biggest_contour>0:
                    cv2.rectangle(cam_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    print('Box x={} y={} w={} h={}'.format(x,y,w,h))
                    if w*h > 2000:
                        image = cam_frame[y:y+h, x:x+w]
                        blank =False
        if ( blank == False ):
            image_from_array = Image.fromarray(image, 'RGB')
            size_image = image_from_array.resize((224,224))
            p = np.expand_dims(size_image, 0)
            dir_path = os.path.dirname(os.path.realpath(__file__))
            img = tf.cast(p, tf.float32)
            probabilitie = model.predict(img)
            prediction = np.argmax(probabilitie, axis=-1)

            print(type(prediction))
            print(Bricks[int(prediction)])

            title = "Predict="+Bricks[prediction[0]]
            print(title)
            path_pic = os.path.join(path,'DATA','LEGO-brick-images')
            filenames_match = tf.io.gfile.glob(path_pic+'/'+Bricks[prediction[0]]+'/*.jpg')
            print(prediction)
            image2 = cv2.imread(filenames_match[0])
            
            #plt.subplot(*subplot)
            #cid = fig.canvas.mpl_connect('button_press_event', onclick)
            
            fig, ax = plt.subplots(2,2)
            plt.title('Source')
            ax[0,0].imshow(frame)
            ax[0,0].axis('off')
            #ax[0,1].title('Cut out')
            ax[0,1].axis('off')
            ax[0,1].imshow(cam_frame)
            ax[1,0].axis('off')
            #ax[1,0].title('Box')
            ax[1,0].axis('off')
            ax[1,0].imshow(image)
            ax[1,0].axis('off')
            #ax[1,1].title(title)
            ax[1,1].axis('off')
            ax[1,1].imshow(image2)
            #plt.ioff()
            #plt.close()
            plt.show()
            plt.pause(1)
                #plt.tight_layout()
                #plt.subplots_adjust(wspace=SPACING, hspace=SPACING)                           
       
    #screen_record()
    




