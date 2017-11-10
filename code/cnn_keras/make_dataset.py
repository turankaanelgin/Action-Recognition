# This code reads in dataset and only keeps frames with humans.
# The frames' data are used to train the CNN model.
# Instead of running this code, you can download the results at:
# https://drive.google.com/open?id=1NvzQ0A5zgoCrg-iUgDsNpc2bVvNB_dkh
# And put all of them in ../../data/cnn_keras/
# List of files:
# X_train.pickle, X_dev.pickle, X_test.pickle.
# Y_train.pickle, Y_dev.pickle, Y_test.pickle.

import argparse
import cv2
import numpy as np
import os
import pickle
import sys

CATEGORIES = ["walking", "jogging", "running", "boxing", "handwaving", "handclapping"]

CATEGORY_IDX = {}
CATEGORY_IDX["boxing"] = 0
CATEGORY_IDX["handclapping"] = 1
CATEGORY_IDX["handwaving"] = 2
CATEGORY_IDX["jogging"] = 3
CATEGORY_IDX["running"] = 4
CATEGORY_IDX["walking"] = 5

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset_path", type=str,\
		default="../../data/train_set_human_frame.pickle",\
		help="path to dataset")
	parser.add_argument("--output_X_path", type=str,\
		default="../../data/cnn_rawframe/X_train.pickle",\
		help="path to output features' set X")
	parser.add_argument("--output_Y_path", type=str,\
		default="../../data/cnn_rawframe/Y_train.pickle",\
		help="path to output labels' set Y")

	args = parser.parse_args()
	dataset_path = args.dataset_path
	output_X = args.output_X_path
	output_Y = args.output_Y_path

	# Load dataset.
	print("Loading", dataset_path)
	dataset = pickle.load(open(dataset_path, "rb"))

	# Initialize features and labels list.
	X = []
	Y = []

	print("Creating instances of training examples by bundling 15 consecutive frames")
	cnt_vids = 0
	for video in dataset:
		x = []

		# Get the label.
		y = CATEGORY_IDX[video["category"]]

		# cnt keeps the number of frames we have bundled.
		cnt = 0
		for frame in video["frames"]:
			x.append(frame)
			cnt += 1

			# We bundle 15 consecutive frames into one block.
			# This block is used as an instance of training example to our CNN.
			if cnt == 15:

				x = np.array(x)
				x = np.transpose(x, (1, 2, 0))
				X.append(x)
				Y.append(y)
				x = []
				cnt = 0
		cnt_vids += 1
		print("Processed %d/%d files %d" % (cnt_vids, len(dataset), len(X)))

	X = np.array(X)
	Y = np.array(Y)

	# Save.
	print("Saving to", output_X)
	pickle.dump(X, open(output_X, "wb"))

	print("Saving to", output_Y)
	pickle.dump(Y, open(output_Y, "wb"))