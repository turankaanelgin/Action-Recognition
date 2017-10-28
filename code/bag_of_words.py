import os
import pickle
import argparse
from collections import defaultdict
from scipy.cluster.vq import vq
from sklearn import svm
from numpy import zeros

CATEGORIES = ["walking", "jogging", "running", "boxing", "handwaving", "handclapping"]
DATASET_DIR = "../data"

def load_data(clusters, n_samples):
    print("Loading training set...")
    train_sift = pickle.load(open(os.path.join(DATASET_DIR, "train_sift.pickle"), "rb"))
    print("Loading development set...")
    dev_sift = pickle.load(open(os.path.join(DATASET_DIR, "dev_sift.pickle"), "rb"))
    print("Loading test set...")
    test_sift = pickle.load(open(os.path.join(DATASET_DIR, "test_sift.pickle"), "rb"))
    print("Loading kmeans model...")
    kmeans = pickle.load(open(os.path.join(DATASET_DIR, "train_kmeans_%dclusters_%dsamples.pickle" % \
                                           (clusters, n_samples)), "rb"))
    cluster_centers = kmeans.cluster_centers_
    return (train_sift, dev_sift, test_sift, cluster_centers)

def compute_histograms(data, cluster_centers):
    # compute visual word histograms for each category of
    # training data
    histograms = defaultdict(list)
    for video in data:
        category = video["category"]
        histogram = compute_single_histogram(video, cluster_centers)
        assert histogram.size == len(cluster_centers)
        histograms[category].append(histogram)
    return histograms

def compute_single_histogram(video, cluster_centers):
    histogram = zeros(len(cluster_centers))
    frames = video["frames"]
    cnt = 1
    for frame in frames:
        clusters = vq(frame, cluster_centers)
        cluster_indices = clusters[0]
        for cluster in cluster_indices:
            histogram[cluster] += 1
        cnt += 1
        if cnt % 10 == 0:
            print("Frame %d has been processed" % cnt)
    return histogram

def svm_classifiers(histograms):
    classifiers = {}
    for category in CATEGORIES:
        print('Creating classifier for %s' % category)
        clf = svm.SVC(probability=True)
        examples = histograms[category]
        labels = len(examples) * [1]
        for other in CATEGORIES:
            if category != other:
                negatives = histograms[other]
                examples += negatives
                labels += len(negatives) * [0]
        model = clf.fit(examples, labels)
        classifiers[category] = model
    return classifiers

def one_vs_all_classification(data, cluster_centers, classifiers):
    predicted_labels = []
    histograms = []
    for video in data:
        histogram = compute_single_histogram(video, cluster_centers)
        histograms.append(histogram)
    probs = []
    for svc in classifiers.values():
        prob = svc.predict_proba(histograms)
        probs.append(prob)

    for i in range(len(data)):
        maxProb = 0
        maxClass = ''
        for j in range(len(probs)):
            prob = probs[j][i][0]
            if prob > maxProb:
                maxProb = prob
                maxClass = CATEGORIES[j]
        predicted_labels.append(maxClass)
    print(predicted_labels)
    return predicted_labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run KMeans on training set")
    parser.add_argument("--clusters", type=int, default=1000,
                        help="number of clusters (default: 1000)")
    parser.add_argument("--n_samples", type=int, default=100000,
                        help="number of feature points to sample (default: 1000)")
    args = parser.parse_args()
    clusters = args.clusters
    n_samples = args.n_samples

    train_sift, dev_sift, test_sift, cluster_centers = load_data(clusters, n_samples)
    train_histograms = compute_histograms(train_sift, cluster_centers)
    classifiers = svm_classifiers(train_histograms)
    predicted_labels = one_vs_all_classification(train_sift, cluster_centers, classifiers)



