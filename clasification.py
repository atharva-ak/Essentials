## Atharva Kadethankar
## MS2018001
## VR Assignment Q9

import numpy as np
import cv2
import glob
import os


land_dir = "./landscape/" # Enter Directory of all images 
land_path = os.path.join(land_dir,'*.jpg')
land_files = glob.glob(land_path)
land_data = []

portrate_dir = "./portrait/" # Enter Directory of all images 
portrate_path = os.path.join(portrate_dir,'*.jpg')
portrate_files = glob.glob(portrate_path)
portrate_data = []

night_dir = "./night/" # Enter Directory of all images 
night_path = os.path.join(night_dir,'*.jpg')
night_files = glob.glob(night_path)
night_data = []

sift = cv2.xfeatures2d.SIFT_create()  # SIFT features 

land_array=[]
for land_img in land_files:
	land_img1= cv2.imread(land_img,0) 
	land_data.append(land_img1)
	land_img1=cv2.resize(land_img1,(500,500))
	kp1, des1 = sift.detectAndCompute(land_img1,None)
	landFeature= cv2.normalize(des1,des1)
	landFeature=np.mean(landFeature,axis=0)
	land_array.append(landFeature)


portrate_array=[]
for portrate_img in portrate_files:
	portrate_img1= cv2.imread(portrate_img,0) 
	portrate_data.append(portrate_img1)
	portrate_img1=cv2.resize(portrate_img1,(500,500))
	kp2, des2 = sift.detectAndCompute(portrate_img1,None)
	portrateFeature= cv2.normalize(des2,des2)
	portrateFeature=np.mean(portrateFeature,axis=0)
	portrate_array.append(portrateFeature)


night_array=[]
for night_img in night_files:
	night_img1= cv2.imread(night_img,0) 
	night_data.append(night_img1)
	night_img1=cv2.resize(night_img1,(500,500))
	kp3, des3 = sift.detectAndCompute(night_img1,None)
	night_Feature= cv2.normalize(des3,des3)
	night_Feature=np.mean(night_Feature,axis=0)
	night_array.append(night_Feature)


trainingData = land_array+portrate_array+night_array

trainingLabel = [0]*len(land_array)+[1]*len(portrate_array)+[2]*len(night_array)

trainingData = np.array(trainingData)

trainingLabel = np.array(trainingLabel)

# Test image
test = cv2.imread('test7.jpg',0)
test = cv2.resize(test,(500,500))
kp4, des4 = sift.detectAndCompute(test,None)
test_Feature = cv2.normalize(des4,des4)
test_Feature =np.mean(test_Feature,axis=0)
test_array = []
test_array.append(test_Feature)
test = np.array(test_array)

knn = cv2.ml.KNearest_create()           # KNN model
knn.train(trainingData,cv2.ml.ROW_SAMPLE,trainingLabel)
ret, results, neighbours ,dist = knn.findNearest(test,3)

if results[0]==0:
	print "Result: Landscape\n"
elif results[0]==1:
	print "Result: Portrate\n"
else:
	print "Result: Night\n" 