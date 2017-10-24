import cv2
import os
import pickle
import sys

CATEGORIES = ["walking", "jogging", "running", "boxing", "handwaving", \
														"handclapping"]

DATASET_DIR = "../data"

if __name__ == "__main__":

	# This dictionary contains bounding box information of all videos in each
	# category. human_detected["walking"] contains information of all videos
	# in category "walking".
	human_detected = {}

	# Load human detected bounding box we have found from file.
	for category in CATEGORIES:
		file_path = os.path.join(DATASET_DIR, "human_detected_%s" % category)\
																+ ".pickle"
		human_detected[category] = pickle.load(open(file_path, "rb"))

	# Create HOG and SIFT to extract features.
	hog = cv2.HOGDescriptor()
	sift = cv2.xfeatures2d.SIFT_create()

	for category in CATEGORIES:
		print("Processing category %s" % category)

		# This list stores features of all videos in this category. Each entry
		# corresponds to a video.
		category_features = []

		# cnt_files keeps track of the number of videos we have processed.
		cnt_files = 0

		for video in human_detected[category]:
			# Get video's name and its file path.
			file_name = video["file_name"]
			file_path = os.path.join(DATASET_DIR, category, file_name)

			# Get the human bounding box of all frames in this video.
			human_rects = video["rects"]
        
        	# cnt_frames keeps track of the frame's index we are processing.
			cnt_frames = 0

			# List to store all features in all frames of this video. Each entry
			# corresponds to one frame.
			features = []

			# Open video.
			vid = cv2.VideoCapture(file_path)

			while vid.isOpened():
				ret, frame = vid.read()
				# Break if got to the end of video.
				if not ret:
					break
                
                # Only care about gray scale.
				frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    			# Get the human bounding box in current frame.
				rects = human_rects[cnt_frames]

				# If there is a bounding box.
				if len(rects) > 0:
					x0, y0, w, h = rects[0]
					x1 = x0 + w
					y1 = y0 + h
					human_image = frame[y0:y1, x0:x1]
					
					# Detect SIFT keypoints and compute SIFT descriptors.
					kp = sift.detect(human_image)
					desc = sift.compute(human_image, kp)

					# Keypoint data type in opencv cannot be dumps using pickle.
					# So we create a tuple to store all its data instead.
					sift_desc = []

					# Iterate over each keypoint.
					for idx in range(len(desc[0])):
						# Keypoint.
						kp = desc[0][idx]
						# Descriptor.
						kp_desc = desc[1][idx]

						# Create tuple to store keypoint's information. This
						# tuple can be dumped with pickle.
						temp = (kp.angle, kp.class_id, kp.octave, kp.pt, \
							kp.response, kp.size, kp_desc)
						# Add to list of descriptors.
						sift_desc.append(temp)

					features.append({
						"frame_idx": cnt_frames,
						"sift": sift_desc
					})
				
				cnt_frames += 1

			category_features.append({
				"category": category,
				"file_name": file_name,
				"features": features
			})

			cnt_files += 1
			print("Done %d files" % cnt_files)

		pickle.dump(category_features, open(os.path.join(DATASET_DIR,\
										"sift_%s.pickle" % category), "wb"))