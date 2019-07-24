import cv2
import os
import glob
(winW, winH) = (224, 224)
# loop over the image pyramid
def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

d = 0
normal_train_dir = "/mnt/a25e8cca-e722-4bcc-ae6b-e73124ab090c/IIITB/practice/Normal/test/" # Enter Directory of all images 
normal_train_path = os.path.join(normal_train_dir,'*.jpeg')
normal_train_files = glob.glob(normal_train_path)

for normal_train_img in normal_train_files:
	image = cv2.imread(normal_train_img)
	for (x, y, window) in sliding_window(image, stepSize=224, windowSize=(winW, winH)):
		# if the window does not meet our desired window size, ignore it
		if window.shape[0] != winH or window.shape[1] != winW:
			continue

		# THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
		# MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
		# WINDOW
		# clone = image.copy()
		crop = image[y:y + winH, x:x + winW]
		# since we do not have a classifier, we'll just draw the window

		# cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
		# cv2.imshow("Window", clone)
		# imwrite(I,[Resultados1,num2str(k),'.tif']);

		# cv2.imwrite(os.path.join(path , str(i),'/.jpeg'), crop)
		# cv2.waitKey(0)
		filename = "/mnt/a25e8cca-e722-4bcc-ae6b-e73124ab090c/IIITB/practice/Normal/test224/img_%d.jpeg"%d
		cv2.imwrite(filename, crop)
		d+=1

