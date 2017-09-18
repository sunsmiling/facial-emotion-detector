
# coding: utf-8

# In[1]:

import glob,random,math,dlib,cv2,itertools,sys,os,pickle,warnings,argparse


# In[2]:

import numpy as np
from numpy import genfromtxt

from pandas import Series, DataFrame
import pandas as pd

from sklearn.externals import joblib
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


from imutils import face_utils


# In[3]:

warnings.filterwarnings("ignore", category=DeprecationWarning) 


# In[6]:

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--classifier", required=True,
	help="path to facial emotion classifier")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-f", "--font_size", required=True, type=float,
	help="emotion font size")
ap.add_argument("-w", "--font_width", required=True, type=int,
	help="emotion font width")
ap.add_argument("-r", "--rec_size", required=True, type=int,
	help="face rectangle size")
args = vars(ap.parse_args())


# In[7]:

def progress(count, total, suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '#' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('|%s| %s%s ...%s\r' % (bar, percents, '%', suffix))
    sys.stdout.flush()  # As suggested by Rom Ruben


# In[8]:

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
landmark_path='shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
getLandmark = dlib.shape_predictor(landmark_path)


# In[9]:

columns_chin=[]
for i in range(1,17):
    columns_chin.append('x'+str(i))
    columns_chin.append('y'+str(i))
    columns_chin.append('d_cent'+str(i))
    columns_chin.append('a_cent'+str(i))
columns_chin.append("x68")
columns_chin.append("y68")
columns_chin.append("d_cent68")
columns_chin.append("a_cent68")

tmp=[]
for i in range(1,69):
    tmp.append("x" + str(i))
    tmp.append("y" + str(i))
    tmp.append("d_cent" + str(i))
    tmp.append("a_cent" + str(i))
    
tmp=[]
for i in range(1,69):
    tmp.append("x" + str(i))
    tmp.append("y" + str(i))
    tmp.append("d_cent" + str(i))
    tmp.append("a_cent" + str(i))

new_colname=['x17','x18','x19','x20','x21','x22','x23','x24','x25','x26','x27','x28','x29','x30','x31','x32','x33','x34','x35',
             'x36','x37','x38','x39','x40','x41','x42','x43','x44','x45','x46','x47','x48','x49','x50','x51','x52','x53','x54',
             'x55','x56','x57','x58','x59','x60','x61','x62','x63','x64','x65','x66','x67',
             'y17','y18','y19','y20','y21','y22','y23','y24','y25','y26','y27','y28','y29','y30','y31','y32','y33','y34','y35',
             'y36','y37','y38','y39','y40','y41','y42','y43','y44','y45','y46','y47','y48','y49','y50','y51','y52','y53','y54',
             'y55','y56','y57','y58','y59','y60','y61','y62','y63','y64','y65','y66','y67',
             'd_cent17','d_cent18','d_cent19','d_cent20','d_cent21','d_cent22','d_cent23','d_cent24','d_cent25','d_cent26','d_cent27','d_cent28','d_cent29','d_cent30','d_cent31','d_cent32','d_cent33','d_cent34','d_cent35',
             'd_cent36','d_cent37','d_cent38','d_cent39','d_cent40','d_cent41','d_cent42','d_cent43','d_cent44','d_cent45','d_cent46','d_cent47','d_cent48','d_cent49','d_cent50','d_cent51','d_cent52','d_cent53','d_cent54',
             'd_cent55','d_cent56','d_cent57','d_cent58','d_cent59','d_cent60','d_cent61','d_cent62','d_cent63','d_cent64','d_cent65','d_cent66','d_cent67',
             'a_cent17','a_cent18','a_cent19','a_cent20','a_cent21','a_cent22','a_cent23','a_cent24','a_cent25','a_cent26','a_cent27','a_cent28','a_cent29','a_cent30','a_cent31','a_cent32','a_cent33','a_cent34','a_cent35',
             'a_cent36','a_cent37','a_cent38','a_cent39','a_cent40','a_cent41','a_cent42','a_cent43','a_cent44','a_cent45','a_cent46','a_cent47','a_cent48','a_cent49','a_cent50','a_cent51','a_cent52','a_cent53','a_cent54',
             'a_cent55','a_cent56','a_cent57','a_cent58','a_cent59','a_cent60','a_cent61','a_cent62','a_cent63','a_cent64','a_cent65','a_cent66','a_cent67']

my_cols = list(new_colname)


# In[18]:

def get_landmarks_integ(image,classifier):  
    frame = cv2.imread(image, cv2.IMREAD_GRAYSCALE) 
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_image = clahe.apply(frame)
    detections = detector(clahe_image, 1)
    landmarks_vectorised = []    
    if len(detections) > 0:
        for k,d in enumerate(detections): #For all detected face instances individually
            shape = getLandmark(frame, d) #Draw Facial Landmarks with the predictor class
            xlist = []
            ylist = []

            for i in range(0,68): #Store X and Y coordinates in two lists
                xlist.append(float(shape.part(i).x))
                ylist.append(float(shape.part(i).y))

            xmean = np.mean(xlist) #Find both coordinates of centre of gravity
            ymean = np.mean(ylist)
            xcentral = [(x-xmean) for x in xlist] #Calculate distance centre <-> other points in both axes #x
            ycentral = [(y-ymean) for y in ylist] #y

            if xlist[27] == xlist[30]: #If x-coordinates of the set are the same, the angle is 0, catch to prevent 'divide by 0' error in function
                anglenose = 0
            else:
                anglenose = int(math.atan((ylist[27]-ylist[30])/(xlist[27]-xlist[30]))*180/math.pi) #point 29 is the tip of the nose, point 26 is the top of the nose brigde
                    

            for x,y,w,z in zip(xcentral,ycentral,xlist,ylist):
                landmarks_vectorised.append(x) #Add the coordinates relative to the centre of gravity
                landmarks_vectorised.append(y)

                        #Get the euclidean distance between each point and the centre point (the vector length)
                coornp = np.asarray((z,w))
                meannp = np.asarray((ymean,xmean))
                dist = np.linalg.norm(coornp-meannp)
                landmarks_vectorised.append(dist)

                anglerelative = (math.atan((z-ymean)/(w-xmean+0.0001))*180/math.pi) - anglenose
                landmarks_vectorised.append(anglerelative)
    else:
        print("warnings!"+":Face was not detected at ["+image+"] file.") 
    img_num = int(np.array(landmarks_vectorised).shape[0]/272)
    detet_num = len(detections)   
    df_All = DataFrame(np.array(landmarks_vectorised).reshape(img_num,272))
    df_All.columns=tmp
    df_All_var = df_All.drop(columns_chin,axis=1)
    df_All_result = df_All_var[my_cols]
    result = []
    emotion_result = []  
    
    for i in range(0,detet_num):
        clf1 = joblib.load(classifier)
        if clf1.predict(df_All_result[i:i+1]) ==6 : emotion = "Neutral"
        elif clf1.predict(df_All_result[i:i+1]) ==4 : emotion = "Sad"
        elif clf1.predict(df_All_result[i:i+1]) ==3 : emotion = "Happy"
        else :  emotion = "Angry"
        emotion_result.append(emotion)

        result.append(df_All_result)
        
        
    return len(detections), emotion_result


def imageimport(image,font_size,font_width,rec_size,classifier):
    col_img = cv2.imread(image)
    frame = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    detections = detector(frame,1) 
    if len(detections) > 0: 
        print("Number of faces detected: {}".format(len(detections)))
        print("Facial expression: {}".format(get_landmarks_integ(image,classifier)[1]))
        for (i,rect) in enumerate(detections):
            progress(i+1,len(detections),suffix='Emotion detecting')
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(col_img, (x, y), (x + w, y + h), (0,255,0),rec_size)
            #show emotion
            emotion_list = get_landmarks_integ(image,classifier)[1]
            cv2.putText(col_img, emotion_list[i], (x,y+(font_width*5)), cv2.FONT_HERSHEY_SIMPLEX, font_size,(0,15,7), font_width)
        cv2.namedWindow("Detected", cv2.WINDOW_NORMAL)
        cv2.imshow("Detected",col_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print()
    else:
        print("detection fail")


# In[ ]:

imageimport(args["image"],args["font_size"],args["font_width"],args["rec_size"],args["classifier"])

