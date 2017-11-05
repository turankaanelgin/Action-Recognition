import pickle
import os
import cv2
from numpy import size, memmap
from joblib import Parallel, delayed

DATASET_DIR = "../data"
farneback_params = dict(winsize = 20, iterations=1, flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN, levels=1,
                        pyr_scale=0.5, poly_n=5, poly_sigma=1.1, flow=None)


def calc_optical_flow(data):
    optical_flows_x = []
    optical_flows_y = []
    cnt = 0

    for video in data:
        frames = video["frames"]
        prevFrame = frames[0]

        flows_per_video_x = []
        flows_per_video_y = []
        for frame in frames[1:]:
            flows_per_frame_x = []
            flows_per_frame_y = []
            flows = cv2.calcOpticalFlowFarneback(prevFrame, frame, **farneback_params)
            height = size(flows, 0)
            width = size(flows, 1)

            sampled_flows = []
            for row in range(height):
                for col in range(width):
                    if row % 14 == 0 and col % 14 == 0:
                        sampled_flows.append(flows[row][col].copy())
            del flows
            for flow in sampled_flows:
                flows_per_frame_x.append(flow[0])
                flows_per_frame_y.append(flow[1])
            prevFrame = frame.copy()
            flows_per_video_x.append(flows_per_frame_x)
            flows_per_video_y.append(flows_per_frame_y)
        optical_flows_x.append(flows_per_video_x)
        optical_flows_y.append(flows_per_video_y)
        cnt += 1
        print('Video %d was processed' % cnt)
    return optical_flows_x, optical_flows_y

if __name__ == "__main__":
    train = pickle.load(open(os.path.join(DATASET_DIR, "train_set.pickle"), "rb"))
    print("Training set was loaded...")
    dev = pickle.load(open(os.path.join(DATASET_DIR, "dev_set.pickle"), "rb"))
    print("Development set was loaded...")
    test = pickle.load(open(os.path.join(DATASET_DIR, "test_set.pickle"), "rb"))
    print("Test set was loaded...")

    train_flows_x, train_flows_y = calc_optical_flow(train)
    print("Training flows were calculated")
    dev_flows_x, dev_flows_y = calc_optical_flow(dev)
    print("Development flows were calculated")
    test_flows_x, test_flows_y = calc_optical_flow(test)
    print("Testing flows were calculated")

    # Save x and y separately to avoid memory error
    pickle.dump(train_flows_x, open(os.path.join(DATASET_DIR, "train_dense_flow_x.pickle"), "wb"))
    pickle.dump(train_flows_y, open(os.path.join(DATASET_DIR, "train_dense_flow_y.pickle"), "wb"))
    pickle.dump(dev_flows_x, open(os.path.join(DATASET_DIR, "dev_dense_flow_x.pickle"), "wb"))
    pickle.dump(dev_flows_y, open(os.path.join(DATASET_DIR, "dev_dense_flow_y.pickle"), "wb"))
    pickle.dump(test_flows_x, open(os.path.join(DATASET_DIR, "test_dense_flow_x.pickle"), "wb"))
    pickle.dump(test_flows_y, open(os.path.join(DATASET_DIR, "test_dense_flow_y.pickle"), "wb"))