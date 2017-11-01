import cv2
import os
import pickle
import sys
import numpy as np

CATEGORIES = ["walking", "jogging", "running", "boxing", "handwaving", \
"handclapping"]

DATASET_DIR = "/Users/karan/Documents/CMSC726/Project/data"


if __name__ == "__main__":
	# params for ShiTomasi corner detection
	feature_params = dict( maxCorners = 100, qualityLevel = 0.3, minDistance = 7, blockSize = 7)

	# Parameters for lucas kanade optical flow
	lk_params = dict(winSize = (15,15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

	# Create some random colors
	color = np.random.randint(0,255,(100,3))

	# Dictionary to store the name of all video files. e.g files["walking"]
	# stores the name of all video files in "walking" category.
	files = {}

	for category in CATEGORIES:
		files[category] = []
		folder_path = os.path.join(DATASET_DIR, category)

		# Iterate over all files in this direcory.
		for file_name in os.listdir(folder_path):
			if file_name.endswith("avi"):
				files[category].append(file_name) 


	for category in CATEGORIES:
		print("Processing category %s" % category)

		# Get all video files.
		file_names = files[category]

		# Each entry in this list corresponds to a video, with the key-value
		# pairs as follows:
		# "category": category in which this video belongs.
		# "file_name": file's name.
		# "rects": a list in which each entry corresponds to a frame in the
		# video. Each entry contains information of the bounding box enclosing 
		# the human in that frame.
		category_tracking = []

		# file_count keeps track of the number of files we have processed.
		file_count = 0

		for file_name in file_names:

			# Create a list to store the tracked positions in current video.
			position = []

			file_count += 1
			print("Starting file no. %d now." % file_count)

			file_path = os.path.join(DATASET_DIR, category, file_name)
			print("The path of this file is %s" % file_path)

			# Read video.
			vid = cv2.VideoCapture(file_path)

			frame_count = 0
			while vid.isOpened():

				# Take first frame and find corners in it
				ret, old_frame = vid.read()
				if old_frame is None:
					break
				frame_count += 1
				if not (len(old_frame.shape) == 3 or len(old_frame.shape) == 4):
					continue 
				old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

				# Function finds the most prominent corners in the image
				p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
				if p0 is None:
					break
				#print(p0.shape)

				# Create a mask image for drawing purposes
				mask = np.zeros_like(old_frame)

				while(1):
					ret, frame = vid.read()
					if frame is None:
						break
					frame_count += 1
					frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

					# Calculate optical flow -> p1 gives the next point
					p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
					#p1 = np.float32(p1)
					#print(p1.shape)

					if p1 is None:
						break

					# Select good points
					good_new = p1[st==1]
					good_old = p0[st==1]

					# Appending the tracking points in list
					position.append((good_old,good_new))

					# Draw the tracks
					for i,(new,old) in enumerate(zip(good_new,good_old)):
						a,b = new.ravel()
						c,d = old.ravel()
						mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
						frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
					img = cv2.add(frame,mask)

					cv2.imshow('frame',img)
					k = cv2.waitKey(30) & 0xff
					if k == 27:
						break

					# Now update the previous frame and previous points
					old_gray = frame_gray.copy()
					p0 = good_new.reshape(-1,1,2)
	
				category_tracking.append({
					"category": category,
					"file_name": file_name,
					"tracking": position 
					})

			cv2.destroyAllWindows()
			vid.release()

			print("Completed file number %d" % file_count)
			print("We read %d frames in this file" % frame_count)

		# Dump data to file.
		pickle.dump(category_tracking, open(os.path.join(DATASET_DIR, \
			"optical_flow_%s.pickle" % category), "wb"))