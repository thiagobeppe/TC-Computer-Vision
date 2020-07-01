from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os


## References -> https://www.pyimagesearch.com/2016/08/08/k-nn-classifier-for-image-classification/
def image_to_feature_vector(image, size=(32, 32)):
	return cv2.resize(image, size).flatten()

def extract_color_histogram(image, bins=(8, 8, 8)):
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
		[0, 180, 0, 256, 0, 256])
	if imutils.is_cv2():
		hist = cv2.normalize(hist)
	else:
		cv2.normalize(hist, hist)
	return hist.flatten()

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-k", "--neighbors", type=int, default=1,
	help="# of nearest neighbors for classification")
ap.add_argument("-j", "--jobs", type=int, default=-1,
	help="# of jobs for k-NN distance (-1 uses all available cores)")
args = vars(ap.parse_args())


print("[INFO] describing images...")
imagePaths = list(paths.list_images(args["dataset"]))
rawImages = []
features = []
labels = []

for (i, imagePath) in enumerate(imagePaths):
	img = cv2.imread(imagePath)
	label = imagePath.split(os.path.sep)[-1].split(".")[0]
	pixels = image_to_feature_vector(img)
	hist = extract_color_histogram(img)
	rawImages.append(pixels)
	features.append(hist)
	labels.append(label)
	if i > 0 and i % 1000 == 0:
		print("[INFO] processed {}/{}".format(i, len(imagePaths)))

rawImages = np.array(rawImages)
features = np.array(features)
labels = np.array(labels)
print("[INFO] pixels matrix: {:.2f}MB".format(
	rawImages.nbytes / (1024 * 1000.0)))
print("[INFO] features matrix: {:.2f}MB".format(
	features.nbytes / (1024 * 1000.0)))


(trainRI, testRI, trainRL, testRL) = train_test_split(
	rawImages, labels, test_size=0.25, random_state=42)
(trainFeat, testFeat, trainLabels, testLabels) = train_test_split(
	features, labels, test_size=0.25, random_state=42)


ann = cv2.ml.ANN_MLP_create()
ann.setLayerSizes(np.array([9,5,9]))
ann.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)
ann.train(trainRI, cv2.ml.ROW_SAMPLE)
print(ann)