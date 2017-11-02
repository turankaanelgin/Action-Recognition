import argparse
import numpy as np
import os
import pickle

from sklearn.svm import SVC

DATASET_DIR = "../../data"

def make_dataset(data):
	X = []
	Y = []

	for video in data:
		for frame in video["frames"]:
			# Number of total frames in train + dev set is too large (around
			# 60K), which makes SVM run very slow. Therefore, we just sample
			# 1/3 of the total number of frames (20K).
			prob = np.random.rand()
			if prob <= 1/3:
				X.append(frame)
				Y.append(video["category"])

	return X, Y

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Train SVM on bow vectors")
	parser.add_argument("--bow_file", type=str,\
		default="train_sift_bow_idf.pickle",\
		help="path to bow file")
	parser.add_argument("--C", type=float,\
		default=1)
	parser.add_argument("--output", type=str,\
		default="train_svm.pickle")

	args = parser.parse_args()
	bow_file = args.bow_file
	C = args.C
	output = args.output

	# Load and make dataset.
	data = pickle.load(open(os.path.join(DATASET_DIR, bow_file), "rb"))
	X, Y = make_dataset(data)

	# data might take a lot of memory space. Don't know if this really clears
	# up memory.
	del data

	# Train SVM and save to file.
	clf = SVC(C=C, kernel="linear", verbose=True)
	clf.fit(X, Y)
	pickle.dump(clf, open(os.path.join(DATASET_DIR, output), "wb"))
