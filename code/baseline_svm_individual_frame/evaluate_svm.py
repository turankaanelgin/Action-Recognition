import argparse
import numpy as np
import os
import pickle

from sklearn.svm import SVC

CATEGORIES = ["walking", "jogging", "running", "boxing", "handwaving", "handclapping"]
DATASET_DIR = "../../data"

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Evaluate SVM classifier")
	parser.add_argument("--svm_file", type=str,\
		default="train_svm.pickle",\
		help="path to svm file")
	parser.add_argument("--bow_file", type=str,\
		default="train_sift_bow_idf.pickle",\
		help="path to bow file")

	args = parser.parse_args()
	bow_file = args.bow_file
	svm_file = args.svm_file

	data = pickle.load(open(os.path.join(DATASET_DIR, bow_file), "rb"))

	# Load trained SVM classifier.
	clf = pickle.load(open(os.path.join(DATASET_DIR, svm_file), "rb"))

	correct = 0
	for video in data:
		# Initialize majority votes.
		majority = {}
		for category in CATEGORIES:
			majority[category] = 0
        
		# Predict every frame in this video.
		for frame in video["frames"]:
			predicted = clf.predict([frame])
			# Increase vote.
			majority[predicted[0]] += 1

		# Get the majority.
		predicted = None
		max_vote = -1
		for category in CATEGORIES:
			if majority[category] > max_vote:
				max_vote = majority[category]
				predicted = category

		# Check if majority is correct.
		if predicted == video["category"]:
			correct += 1

	print("%d/%d Correct" % (correct, len(data)))
	print("Accuracy =", correct / len(data))