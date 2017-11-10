# Trained model cnn_3conv2d_drop05_15.h5 can be downloaded at:
# https://drive.google.com/open?id=14v9D__AlRU7JCg0s2-mF4Z-K5dL3G2Ht
#
# test_set_human_frame.pickle:
# https://drive.google.com/open?id=0B0qJbmmIVAJIWHNhd0s1TDBWaFE
import argparse
import numpy as np
import keras
from keras import layers
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Dense, Activation, ZeroPadding2D, Flatten, Conv2D, Dropout
from keras.layers import MaxPooling2D
from keras.models import Model

import pickle

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
	parser.add_argument("--model_path", type=str,\
		default="../../data/cnn_rawframe/cnn_3conv_drop05_15.h5",\
		help="path to trained model")
	parser.add_argument("--dataset_path", type=str,\
		default=".../data/test_set_human_frame.pickle",\
		help="path to test set")

	args = parser.parse_args()
	dataset_path = args.dataset_path

	dataset = pickle.load(open(dataset_path, "rb"))
	model = keras.models.load_model(args.model_path)

	correct = 0
	for video in dataset:
		x = []

		# Label.
		y = CATEGORY_IDX[video["category"]]
		
		# Init majority votes.
		majority = [0 for _ in range(6)]
		
		cnt = 0
		for frame in video["frames"]:
			# Bundling frames.
			x.append(frame)
			cnt += 1

			# If we have bundled 15 consecutive frames.
			if cnt == 15:
				# Create one instance.
				x = np.array(x)
				x = np.transpose(x, (1, 2, 0))
				x = x / 255.0
				x = x.astype(np.float32)

				# Predict.
				softmax = model.predict(np.array([x]))
				res = np.argmax(softmax)
				majority[res] += 1

				x = []
				cnt = 0

		# If classify correctly.
		if np.argmax(majority) == y:
			correct += 1

	print("Accuracy = %f" % correct / len(dataset))