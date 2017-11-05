import cv2
import os
import pickle

# CATEGORIES = ["walking", "jogging", "running", "boxing", "handwaving", \
# 														"handclapping"]
CATEGORIES = ["walking"]

DATASET_DIR = "../data"


if __name__ == "__main__":

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

	# Setup HOG descriptor to detect human.
	hog = cv2.HOGDescriptor()
	hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

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
		category_human_rects = []
	    
	    # cnt keeps track of the number of files we have processed.
		cnt = 0

		for file_name in file_names:
	        
	        # This list contains bounding box rectangles for all frames in
	        # current video.
			video_human_rects = []

			# Read video.
			file_path = os.path.join(DATASET_DIR, category, file_name)
			vid = cv2.VideoCapture(file_path)

			while vid.isOpened():
				print('Hello')
				# Read frame.
				ret, frame = vid.read()
				# Break if we got to the end of the video.
				if not ret:
					break
	                
				# Detect human using HOG.
				(rects, weights) = hog.detectMultiScale(frame, winStride=(1,1),\
														padding=(8, 8),
														scale=1.1)
				video_human_rects.append(rects)
	                    
			category_human_rects.append({
				"category": category,
				"file_name": file_name,
				"rects": video_human_rects 
			})

			cnt += 1
			print("Done %d files" % cnt)

		# Dump data to file.
		pickle.dump(category_human_rects, open(os.path.join(DATASET_DIR, \
			"human_detected_%s.pickle" % category), "wb"))