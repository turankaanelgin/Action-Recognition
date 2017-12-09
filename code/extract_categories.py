import pickle
import os

DATASET_DIR = "../data"

def extract_categories(data):
    categories = []
    for ex in data:
        categories.append(ex["category"])
    return categories

if __name__ == "__main__":
    train = pickle.load(open(os.path.join(DATASET_DIR, "train_set_human_frame.pickle"), "rb"))
    print("Training set was loaded...")
    dev = pickle.load(open(os.path.join(DATASET_DIR, "dev_set_human_frame.pickle"), "rb"))
    print("Development set was loaded...")
    test = pickle.load(open(os.path.join(DATASET_DIR, "test_set_human_frame.pickle"), "rb"))
    print("Test set was loaded...")

    train_categories = extract_categories(train)
    dev_categories = extract_categories(dev)
    test_categories = extract_categories(test)

    pickle.dump(train_categories, open(os.path.join(DATASET_DIR, "train_categories.pickle"), "wb"))
    pickle.dump(dev_categories, open(os.path.join(DATASET_DIR, "dev_categories.pickle"), "wb"))
    pickle.dump(test_categories, open(os.path.join(DATASET_DIR, "test_categories.pickle"), "wb"))