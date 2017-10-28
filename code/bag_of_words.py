import os
import pickle
import argparse
from collections import defaultdict
from scipy.cluster.vq import vq
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
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

def true_labels(data):
    labels = []
    for video in data:
        labels.append(video["category"])
    return labels

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
        if len(frame) == 0:
            continue
        clusters = vq(frame, cluster_centers)
        cluster_indices = clusters[0]
        for cluster in cluster_indices:
            histogram[cluster] += 1
        cnt += 1
        if cnt % 10 == 0:
            print("Frame %d has been processed" % cnt)
    return histogram

def svm_classifier(histograms, kernel, C, gamma):
    examples = []
    labels = []
    for category in CATEGORIES:
        new_examples = histograms[category].copy()
        examples += new_examples
        labels += len(new_examples) * [category]
    classifier = OneVsRestClassifier(SVC(kernel=kernel, C=C, gamma=gamma), n_jobs=2)
    classifier.fit(examples, labels)
    return classifier

def tune_svm_classifier(train_histograms, dev_data, dev_labels, cluster_centers):
    dev_histograms = []
    for video in dev_data:
        histogram = compute_single_histogram(video, cluster_centers)
        dev_histograms.append(histogram)
    best_accuracy = 0
    print('Performing validation...')
    # Polynomial and rbf kernels did not give good results
    for C in range(-4, 3):
        for gamma in range(-4, 3):
            classifier = svm_classifier(train_histograms, 'linear', 10**C, 2**gamma)
            predicted_labels = classifier.predict(dev_histograms)
            dev_accuracy = accuracy_score(dev_labels, predicted_labels)
            print('Accuracy = %f with %s kernel, C = %f, gamma = %f' % (dev_accuracy, 'linear', 10**C, 2**gamma))
            if dev_accuracy > best_accuracy:
                best_accuracy = dev_accuracy
                bestC = C
                bestGamma = gamma
    return bestC, bestGamma

def classify(data, cluster_centers, classifier):
    histograms = []
    for video in data:
        histogram = compute_single_histogram(video, cluster_centers)
        histograms.append(histogram)
    predicted_labels = classifier.predict(histograms)
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
    train_labels = true_labels(train_sift)
    dev_labels = true_labels(dev_sift)
    test_labels = true_labels(test_sift)

    train_histograms = compute_histograms(train_sift, cluster_centers)
    bestC, bestGamma = tune_svm_classifier(train_histograms, dev_sift, dev_labels, cluster_centers)

    train_dev_histograms = compute_histograms(train_sift + dev_sift, cluster_centers)
    classifier = svm_classifier(train_dev_histograms, 'linear', 10**bestC, 2**bestGamma)
    test_histograms = []
    for video in test_sift:
        histogram = compute_single_histogram(video, cluster_centers)
        test_histograms.append(histogram)
    predicted_labels = classifier.predict(test_histograms)
    test_accuracy = accuracy_score(test_labels, predicted_labels)
    print('Test accuracy = %f' % test_accuracy)