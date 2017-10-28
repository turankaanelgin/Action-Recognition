import os
import pickle

CATEGORIES = ["walking", "jogging", "running", "boxing", "handwaving", "handclapping"]
DATASET_DIR = "../data"

TRAIN_PEOPLE_ID = [11, 12, 13, 14, 15, 16, 17, 18]
DEV_PEOPLE_ID = [19, 20, 21, 23, 24, 25, 1, 4]
TEST_PEOPLE_ID = [22, 2, 3, 5, 6, 7, 8, 9, 10]

if __name__ == "__main__":

	train = []
	dev = []
	test = []

	for category in CATEGORIES:
		print("Processing category %s" % category)
		category_features = pickle.load(open(os.path.join(DATASET_DIR,\
			"sift_%s.pickle" % category), "rb"))

		for video in category_features:
			person_id = int(video["filename"].split("_")[0][6:])

			if person_id in TRAIN_PEOPLE_ID:
				train.append(video)
			elif person_id in DEV_PEOPLE_ID:
				dev.append(video)
			else:
				test.append(video)

	print("Saving train/dev/test set to files")
	pickle.dump(train, open(os.path.join(DATASET_DIR, "train_sift.pickle"), "wb"))
	pickle.dump(dev, open(os.path.join(DATASET_DIR, "dev_sift.pickle"), "wb"))
	pickle.dump(test, open(os.path.join(DATASET_DIR, "test_sift.pickle"), "wb"))
