import cv2
from numpy import float32
import os
import pickle
from collections import Counter
from sklearn.cluster import KMeans
from sklearn import svm

CATEGORIES = ["walking", "jogging", "running", "boxing", "handwaving", "handclapping"]
DATASET_DIR = "../data"

if __name__ == "__main__":
    all_features = [] # combine features of all frames
    prefix_sums = [0] # keep the offset for each frame
    for category in CATEGORIES:
        print("Processing category %s" % category)
        category_features = pickle.load(open(os.path.join(
                                             DATASET_DIR, "opt_%s.pickle" % category), "rb"));
        for video in category_features:
            for frame in video:
                for keypoint in frame:
                    all_features += keypoint
                    prefix_sums.append(prefix_sums[-1] + len(keypoint))

    # apply kmeans
    kmeans = KMeans(n_clusters=100, random_state=0, init='k-means++').fit(all_features)
    centers = kmeans.labels_

    # create visual word histograms for each frame
    histograms = {}
    for category in CATEGORIES:
        histograms[category] = []
        category_features = pickle.load(open(os.path.join(
                        DATASET_DIR, "opt_%s.pickle" % category), "rb"));
        cnt = 0
        for video in category_features:
            histogram = Counter()
            for frame in video:
                for keypoint in frame:
                    start = prefix_sums[cnt]
                    end = prefix_sums[cnt+1]
                    cnt += 1
                    for i in range(start, end):
                        histogram[centers(i)] += 1
            histograms[category].append(histogram)

    # do one-vs-all classification with svm
    classifiers = {}
    for category in CATEGORIES:
        clf = svm.SVC()
        positives = list(histograms[category])
        negatives = []
        for other in CATEGORIES:
            if category != other:
                negatives += list(histograms[other])
        model = clf.fit(positives, negatives)
        classifiers[category] = model




