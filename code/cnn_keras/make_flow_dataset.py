# Flow dataset can be downloaded at:
# https://drive.google.com/open?id=1NvzQ0A5zgoCrg-iUgDsNpc2bVvNB_dkh
# List of files:
# X_flowx_train.pickle, X_flowy_train.pickle.
# X_flowx_dev.pickle, X_flowy_dev.pickle.
# X_flowx_test.pickle, X_flowy_test.pickle.

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

farneback_params = dict(winsize = 20, iterations=1, flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN, levels=1,
                        pyr_scale=0.5, poly_n=5, poly_sigma=1.1, flow=None)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset_path", type=str,\
		default="../../data/train_set_human_frame.pickle",\
		help="path to dataset")
	parser.add_argument("--output_X_path", type=str,\
		default="../../data/cnn_rawframe/X_flowx_train.pickle",\
		help="path to output features' set X")
	parser.add_argument("--output_Y_path", type=str,\
		default="../../data/cnn_rawframe/X_flowy_train.pickle",\
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
		y = []

		# cnt keeps the number of frames we have bundled.
		cnt = 0

		for frame in video["frames"]:
			cnt += 1
			small = cv2.resize(frame, (80, 60))

			if cnt > 1:
				flows = cv2.calcOpticalFlowFarneback(prev_frame, small,\
					**farneback_params)
				xxx = np.zeros((30, 40), dtype=np.float32)
				yyy = np.zeros((30, 40), dtype=np.float32)

				for r in range(30):
					for c in range(40):
						xxx[r,c] = flows[r*2, c*2, 0]
						yyy[r,c] = flows[r*2, c*2, 1]
				x.append(xxx)
				y.append(yyy)

			prev_frame = small

			# We bundle 15 consecutive frames into one block.
			# This block is used as an instance of training example to our CNN.
			if cnt == 15:
				x = np.array(x)
				x = np.transpose(x, (1, 2, 0))
				X.append(x)

				y = np.array(y)
				y = np.transpose(y, (1, 2, 0))
				Y.append(y)

				x = []
				y = []
				cnt = 0

		cnt_vids += 1
		print("Processed %d/%d files" % (cnt_vids, len(dataset)))

	X = np.array(X)
	Y = np.array(Y)

	# Save.
	print("Saving to", output_X)
	pickle.dump(X, open(output_X, "wb"))

	print("Saving to", output_Y)
	pickle.dump(Y, open(output_Y, "wb"))