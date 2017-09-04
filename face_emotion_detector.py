
# coding: utf-8

# In[1]:


import glob
import random
import math
import dlib
import cv2
import numpy as np
import itertools
import sys
import os
import argparse
from pandas import Series, DataFrame
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from numpy import genfromtxt
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
import pickle
from sklearn.externals import joblib
from imutils import face_utils
import warnings


# In[2]:

warnings.filterwarnings("ignore", category=DeprecationWarning) 


# In[ ]:

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--classifier", required=True,
	help="path to facial emotion classifier")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-f", "--font_size", required=True, type=int,
	help="emotion font size")
ap.add_argument("-r", "--rec_size", required=True, type=int,
	help="face rectangle size")
args = vars(ap.parse_args())


# In[3]:

def progress(count, total, suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '#' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('|%s| %s%s ...%s\r' % (bar, percents, '%', suffix))
    sys.stdout.flush()  # As suggested by Rom Ruben


# In[4]:

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
landmark_path='shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
getLandmark = dlib.shape_predictor(landmark_path)


# In[5]:

def get_landmarks_with_eyes1(image,classifier):  
    frame = cv2.imread(image, cv2.IMREAD_GRAYSCALE) 
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_image = clahe.apply(frame)
    detections = detector(clahe_image, 1)
    if len(detections) > 0: 
        faceFound = True #If no face is detected set the data to value "error" to catch detection errors
    else: 
        faceFound = False
    landmarks_vectorised = []    
    if(faceFound):
        for k,d in enumerate(detections): #For all detected face instances individually
            shape = getLandmark(frame, d) #Draw Facial Landmarks with the predictor class
            xlist = [] #w
            ylist = [] #z
    
            for i in range(0,68): #Store X and Y coordinates in two lists
                xlist.append(float(shape.part(i).x))
                ylist.append(float(shape.part(i).y))
    
            xmean = np.mean(xlist) #Find both coordinates of centre of gravity
            ymean = np.mean(ylist)
            xcentral = [(x-xmean) for x in xlist] #Calculate distance centre <-> other points in both axes #x
            ycentral = [(y-ymean) for y in ylist] #y
        
            xeyemean = (xlist[39]+xlist[42])/2
            yeyemean = (ylist[39]+ylist[42])/2
            xeyecentral = [(x-xeyemean) for x in xlist] #a
            yeyecentral = [(y-yeyemean) for y in ylist] #b
                    
            l_xeyemean = (xlist[36]+xlist[39])/2
            l_yeyemean = (ylist[36]+ylist[39])/2
            l_xeyecentral = [(x-l_xeyemean) for x in xlist] #c
            l_yeyecentral = [(y-l_yeyemean) for y in ylist] #d
            
            r_xeyemean = (xlist[42]+xlist[45])/2
            r_yeyemean = (ylist[42]+ylist[45])/2
            r_xeyecentral = [(x-r_xeyemean) for x in xlist] #e
            r_yeyecentral = [(y-r_yeyemean) for y in ylist] #f
                 
        
            if xlist[27] == xlist[30]: #If x-coordinates of the set are the same, the angle is 0, catch to prevent 'divide by 0' error in function
                anglenose = 0
            else:
                anglenose = int(math.atan((ylist[27]-ylist[30])/(xlist[27]-xlist[30]))*180/math.pi) #point 29 is the tip of the nose, point 26 is the top of the nose brigde
    
    
            
        
            for x,y,a,b,c,d,e,f,w,z in zip(xcentral,ycentral,xeyecentral,yeyecentral,l_xeyecentral,l_yeyecentral,r_xeyecentral,r_yeyecentral,xlist,ylist):
                landmarks_vectorised.append(x) #Add the coordinates relative to the centre of gravity
                landmarks_vectorised.append(y)
        
            #Get the euclidean distance between each point and the centre point (the vector length)
                coornp = np.asarray((z,w))
            
                meannp = np.asarray((ymean,xmean))
                dist = np.linalg.norm(coornp-meannp)
                landmarks_vectorised.append(dist)
            
                meaneyenp = np.asarray((yeyemean,xeyemean))
                disteye = np.linalg.norm(coornp-meaneyenp)
                landmarks_vectorised.append(disteye)
            
                meanleyenp = np.asarray((l_yeyemean,l_xeyemean))
                distleye = np.linalg.norm(coornp-meanleyenp)
                landmarks_vectorised.append(distleye)
    
                meanreyenp = np.asarray((r_yeyemean,r_xeyemean))
                distreye = np.linalg.norm(coornp-meanreyenp)
                landmarks_vectorised.append(distreye)
    
                #Get the angle the vector describes relative to the image, corrected for the offset that the nosebrigde has when the face is not perfectly horizontal
                anglerelative = (math.atan((z-ymean)/(w-xmean+0.001))*180/math.pi) - anglenose
                landmarks_vectorised.append(anglerelative)
    
                eyeanglerelative = (math.atan((z-yeyemean)/(w-xeyemean+0.001))*180/math.pi) - anglenose
                landmarks_vectorised.append(eyeanglerelative)
    
                leyeanglerelative = (math.atan((z-l_yeyemean)/(w-l_xeyemean+0.001))*180/math.pi) - anglenose
                landmarks_vectorised.append(eyeanglerelative)
    
                reyeanglerelative = (math.atan((z-r_yeyemean)/(w-r_xeyemean+0.001))*180/math.pi) - anglenose
                landmarks_vectorised.append(eyeanglerelative)
    
                
        result = []
        emotion_result = []
        for i in range(0,len(detections)):
            test_X=np.array(landmarks_vectorised).reshape(len(detections),680)[i].ravel(order="F")
            clf1 = joblib.load(classifier)
            if clf1.predict(test_X) ==6 : emotion = "Neutral"
            elif clf1.predict(test_X) ==4 : emotion = "Sad"
            elif clf1.predict(test_X) ==3 : emotion = "Happy"
            else :  emotion = "Angry"
            emotion_result.append(emotion)

            result.append(test_X)
        return result, len(detections), emotion_result
         
    else:
        print("warnings!"+":Face was not detected at ["+image+"] file.")     


# In[8]:

def imageimport(image,font_size,rec_size,classifier):
    frame = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    detections = detector(frame,1) 
    if len(detections) > 0: 
        print("Number of faces detected: {}".format(len(detections)))
        for (i,rect) in enumerate(detections):
            progress(i+1,len(detections),suffix='Emotion detecting')
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0),rec_size)
            #show emotion
            emotion_list = get_landmarks_with_eyes1(image,classifier)[2]
            cv2.putText(frame, emotion_list[i], (x,y), cv2.FONT_HERSHEY_COMPLEX_SMALL, font_size,(255,255,255), 2)
        cv2.namedWindow("Detected", cv2.WINDOW_NORMAL)
        cv2.imshow("Detected",frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print()
    else:
        print("detection fail")


# In[ ]:

imageimport(args["image"],args["font_size"],args["rec_size"],args["classifier"])

