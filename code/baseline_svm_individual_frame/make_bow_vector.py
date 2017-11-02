import argparse
import numpy as np
import os
import pickle

from scipy.cluster.vq import vq

CATEGORIES = ["walking", "jogging", "running", "boxing", "handwaving", "handclapping"]
DATASET_DIR = "../../data"
DATASET = ["train_sift.pickle", "dev_sift.pickle", "test_sift.pickle"]

# Here we make a bow vector for each frame instead of for each video.
# This method takes a lot of time & memory space. Perhaps we should try to optimize this.
def make_bow(data, clusters, tfidf):
    print("Make bow vector for each individual frame")

    # Stack all feature points.
    all_features = []
    n_frames = 0
    for video in data:
        for frame in video["frames"]:
            n_frames += 1
            all_features += frame

    # Quantization: find the corresponding cluster for each feature point.
    print("Finding corresponding cluster for each feature point")
    all_features = np.array(all_features, dtype=np.float)
    visual_word_ids = vq(all_features, clusters)[0]

    # Try to clear up some memory.
    del all_features

    # Init bow vectors for all frames.
    bow = np.zeros((n_frames, clusters.shape[0]), dtype=np.float)

    # Create histograms.
    print("Making histograms")
    idx_frame = 0
    idx_feature = 0
    for video in data:
        for frame in video["frames"]:
            for keypoint in frame:
                visual_word_id = visual_word_ids[idx_feature]
                bow[idx_frame, visual_word_id] += 1
                idx_feature += 1
            idx_frame += 1

    # Check whether to use TF-IDF weighting.
    if tfidf:
        print("Applying TF-IDF weighting")
        freq = np.sum((bow > 0) * 1, axis = 0)
        idf = np.log((n_frames + 1) / (freq + 1))
        bow = bow * idf

    # Replace features in data with the bow vector we've computed.
    idx_frame = 0
    for i in range(len(data)):
        frames = []
        for frame in data[i]["frames"]:
            frames.append(bow[idx_frame])
            idx_frame += 1

        data[i]["frames"] = frames

        if (i + 1) % 50 == 0:
            print("Processed %d/%d videos" % (i + 1, len(data)))

    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make bag of words vector")
    parser.add_argument("--kmeans_file", type=str,\
        default="train_kmeans_1000clusters_100000samples.pickle",\
        help="path to kmeans file")
    parser.add_argument("--tfidf", type=int, default=1,\
        help="whether to use tfidf weighting")
    parser.add_argument("--data", type=str,\
        default="test_sift.pickle",\
        help="path to data file")
    parser.add_argument("--output", type=str,\
        default="test_sift_bow_idf.pickle",\
        help="path to output file")

    args = parser.parse_args()
    kmeans_file = args.kmeans_file
    tfidf = args.tfidf
    data_path = args.data
    output_path = args.output

    # Load clusters.
    kmeans = pickle.load(open(os.path.join(DATASET_DIR, kmeans_file), "rb"))
    clusters = kmeans.cluster_centers_

    # Load dataset.
    data = pickle.load(open(os.path.join(DATASET_DIR, data_path), "rb"))

    # Make bow vectors.
    data_bow = make_bow(data, clusters, tfidf)

    # Save.
    pickle.dump(data_bow, open(os.path.join(DATASET_DIR, output_path), "wb"))

