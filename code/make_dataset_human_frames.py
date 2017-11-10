# http://www.nada.kth.se/cvap/actions/00sequences.txt.
# The above file specifies the frames' indices that have human.
# This code reads in the above file to know which frames have human,
# then for each video, it filters and only keeps frames with human.

import cv2
import numpy as np
import os
import pickle
import re
import sys

CATEGORIES = ["walking", "jogging", "running", "boxing", "handwaving", "handclapping"]
DATASET_DIR = "../data"

TRAIN_PEOPLE_ID = [11, 12, 13, 14, 15, 16, 17, 18]
DEV_PEOPLE_ID = [19, 20, 21, 23, 24, 25, 1, 4]
TEST_PEOPLE_ID = [22, 2, 3, 5, 6, 7, 8, 9, 10]

def parse_sequence_file():
	print("Parsing ../data/00sequences.txt")

	# Read 00sequences.txt file.
	with open('../data/00sequences.txt', 'r') as content_file:
		content = content_file.read()

	# Replace tab and newline character with space, then split file's content
	# into strings.
	content = re.sub("[\t\n]", " ", content).split()

	# Dictionary to keep ranges of frames with humans.
	# Example:
	# video "person01_boxing_d1": [(1, 95), (96, 185), (186, 245), (246, 360)].
	frames_idx = {}

	# Current video that we are parsing.
	current_filename = ""

	for s in content:
		if s == "frames":
			# Ignore this token.
			continue
		elif s.find("-") >= 0:
			# This is the token we are looking for. e.g. 1-95.
			if s[len(s) - 1] == ',':
				# Remove comma.
				s = s[:-1]

			# Split into 2 numbers => [1, 95]
			idx = s.split("-")

			# Add to dictionary.
			if not current_filename in frames_idx:
				frames_idx[current_filename] = []
			frames_idx[current_filename].append((int(idx[0]), int(idx[1])))
		else:
			# Parse next file.
			current_filename = s + "_uncomp.avi"

	return frames_idx

if __name__ == "__main__":

	frames_idx = parse_sequence_file()

	# Initialize datasets.
	train_set = []
	dev_set = []
	test_set = []

	for category in CATEGORIES:
		folderpath = os.path.join(DATASET_DIR, category)
		print("Processing folder %s" % folderpath)

		for filename in os.listdir(folderpath):
			if filename.endswith("avi"):
				# Get person_id from filename.
				person_id = int(filename.split("_")[0][6:])
				
				filepath = os.path.join(DATASET_DIR, category, filename)
				
				# Frames' indices with human.
				frame_idx = frames_idx[filename]
				
				# Read video file.
				vid = cv2.VideoCapture(filepath)
				idx_frame = 0
				frames = []

				while vid.isOpened():
					idx_frame += 1
					ret, frame = vid.read()

					# Check if end of file.
					if not ret:
						break

					# Boolean flag to check if current frame contains human.
					ok = False
					for seg in frame_idx:
						if idx_frame >= seg[0] and idx_frame <= seg[1]:
							# Current frame's index is in range, so it contains
							# human
							ok = True
							break
					
					if ok:
						# Convert to grayscale.
						frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
						frames.append(frame)
				
				file = {
					"filename": filename,
					"category": category,
					"frames": frames
				}
				# Add to corresponding dataset.
				if person_id in TRAIN_PEOPLE_ID:
					train_set.append(file)
				elif person_id in DEV_PEOPLE_ID:
					dev_set.append(file)
				else:
					test_set.append(file)

	# Save to files.
	print("Saving to file ../data/train_set_human_frame.pickle")
	pickle.dump(train_set, open("../data/train_set_human_frame.pickle", "wb"))

	print("Saving to file ../data/test_set_human_frame.pickle")
	pickle.dump(test_set, open("../data/test_set_human_frame.pickle", "wb"))

	print("Saving to file ../data/dev_set_human_frame.pickle")
	pickle.dump(dev_set, open("../data/dev_set_human_frame.pickle", "wb"))