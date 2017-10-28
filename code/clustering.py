import argparse
import numpy as np
import os
import pickle

from sklearn.cluster import KMeans

CATEGORIES = ["walking", "jogging", "running", "boxing", "handwaving", "handclapping"]
DATASET_DIR = "../data"

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Run KMeans on training set")
	parser.add_argument("--clusters", type=int, default=1000,\
						help="number of clusters (default: 1000)")
	parser.add_argument("--n_samples", type=int, default=100000,\
						help="number of feature points to sample (default: 1000)")
	args = parser.parse_args()
	clusters = args.clusters
	n_samples = args.n_samples

	# List of all features.
	train_features = []

	print("Loading train_sift.pickle")
	train_data = pickle.load(open(os.path.join(DATASET_DIR, "train_sift.pickle"), "rb"))

	# Create lists of all features in training set.
	for video in train_data:
		for frame in video["frames"]:
			train_features += frame

	n_features = len(train_features)

	# Randomly sample a number of points to run clustering on.
	sampled_index = set(np.random.choice(n_features, n_samples, replace=False))
	sampled_features = []
	for i, feature in enumerate(train_features):
		if i in sampled_index:
			sampled_features.append(feature)

	print("Number of feature points to run clustering on: %d" % len(sampled_features))

	# Clustering with KMeans.
	print("Running KMeans clustering")
	kmeans = KMeans(init='k-means++', n_clusters=clusters, n_init=10, n_jobs=2, verbose=1)
	kmeans.fit(sampled_features)

	# Save trained kmeans object to file.
	pickle.dump(kmeans, open(os.path.join(DATASET_DIR, "train_kmeans_%dclusters_%dsamples.pickle" % (clusters, n_samples)), "wb"))