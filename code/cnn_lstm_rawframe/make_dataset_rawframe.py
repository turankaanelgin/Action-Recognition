
import tensorflow as tf
import pickle
import os
from pandas import read_csv, DataFrame 
import numpy as np
from scipy.misc.pilutil import imresize

CATEGORIES = ["walking", "jogging", "running", "boxing", "handwaving", "handclapping"]
DATASET_DIR = "../data"

def split_data(data ,dataframe):

    Name = dataframe.columns.get_loc("Filename")
    print (dataframe.columns.values)
    start_1=  dataframe.columns.get_loc("start_1")
    end_1=  dataframe.columns.get_loc("end_1")
    start_2=  dataframe.columns.get_loc("start_2")
    end_2=  dataframe.columns.get_loc("end_2")
    start_3=  dataframe.columns.get_loc("start_3")
    end_3=  dataframe.columns.get_loc("end_3")
    start_4=  dataframe.columns.get_loc("start_4")
    end_4=  dataframe.columns.get_loc("end_4")
    FramesOFIntrest = dataframe.values
    count=0
    features = []
    labels = []
    for ex in data:

        for i in range(FramesOFIntrest.shape[0]):
            
            if(FramesOFIntrest[i,Name] in ex["filename"] ):
                
                count+=1

                if(not (np.isnan(FramesOFIntrest[i,start_1])  or np.isnan(FramesOFIntrest[i,end_1]))):
                	n = int(FramesOFIntrest[i,start_1]-1) 
                	while n<int(FramesOFIntrest[i,end_1]) :
                		if(n+15<FramesOFIntrest[i,end_1]):
                			features.append(ex["frames"][n:n+15])
                			labels.append(ex["category"])
                			n+=15
                		else:
                			n+=1
                		
                if(not (np.isnan(FramesOFIntrest[i,start_2])  or np.isnan(FramesOFIntrest[i,end_2]))):
                	n = int(FramesOFIntrest[i,start_2]-1) 
                	while n<int(FramesOFIntrest[i,end_2]) :
                		if(n+15<FramesOFIntrest[i,end_2]):
                			features.append(ex["frames"][n:n+15])
                			labels.append(ex["category"])
                			n+=15
                		else:
                			n+=1
                if(not (np.isnan(FramesOFIntrest[i,start_3])  or np.isnan(FramesOFIntrest[i,end_3]))):
                	n = int(FramesOFIntrest[i,start_3]-1) 
                	while n<int(FramesOFIntrest[i,end_3]) :

                		if(n+15<FramesOFIntrest[i,end_3]):
                			features.append(ex["frames"][n:n+15])
                			labels.append(ex["category"])
                			n+=15
                		else:
                			n+=1
                if(not (np.isnan(FramesOFIntrest[i,start_4])  or np.isnan(FramesOFIntrest[i,end_4]))):
                	n = int(FramesOFIntrest[i,start_4]-1) 
                	while n<int(FramesOFIntrest[i,end_4]) :
                		if(n+15<FramesOFIntrest[i,end_4]):
                			features.append(ex["frames"][n:n+15])
                			labels.append(ex["category"])
                			n+=15
                		else:
                			n+=1
                            #print("fl a5ar3",  n)
    print ("Dataset size" , len(features), count)
    f=0
    for feature in features:
        f+=1
        for i in range(len(feature)):
            feature[i] = imresize(feature[i], (80,60))

    return (features, labels)


def labels2indices(labels):
    indices = []
    for label in labels:
        oneHot = np.zeros((len(CATEGORIES),1))
        oneHot[CATEGORIES.index(label)]=1
        indices.append(oneHot)
    return indices

if __name__ == "__main__":
	FramesOFIntrest = os.path.join(DATASET_DIR, "FramesOFIntrest.csv")
	FramesOFIntrest_df  = read_csv(FramesOFIntrest, names=None)

	train = pickle.load(open(os.path.join(DATASET_DIR, "train_set.pickle"), "rb"))
	print("Training set was loaded...")
	train_feat, train_labels = split_data(train, FramesOFIntrest_df)
	
	dev = pickle.load(open(os.path.join(DATASET_DIR, "dev_set.pickle"), "rb"))
	print("Development set was loaded...")
	dev_feat, dev_labels = split_data(dev, FramesOFIntrest_df)
	
	test = pickle.load(open(os.path.join(DATASET_DIR, "test_set.pickle"), "rb"))
	print("Test set was loaded...")
	test_feat, test_labels = split_data(test,FramesOFIntrest_df)
	
	train_labels = labels2indices(train_labels)
	test_labels = labels2indices(test_labels)
	dev_labels =  labels2indices(dev_labels)

	train_feat = np.array(train_feat)
	train_labels = np.array(train_labels)
	dev_feat = np.array(dev_feat)
	dev_labels = np.array(dev_labels)
	test_feat = np.array(test_feat)
	test_labels = np.array(test_labels)
	print("train_feat" ,  train_feat.shape)
	print("train_labels" ,  train_labels.shape)
	print("dev_feat" ,  dev_feat.shape)
	print("dev_labels" ,  dev_labels.shape)
	print("test_feat" ,  test_feat.shape)
	print("test_labels" ,  test_labels.shape)
	# Save.
	print("Saving to", os.path.join(DATASET_DIR, "Processed_train_X_set.pickle"))


	pickle.dump(train_feat, open(os.path.join(DATASET_DIR, "Processed_train_X_set.pickle"), "wb") )
	print("Saving to", os.path.join(DATASET_DIR, "Processed_train_Y_set.pickle"))
	pickle.dump(train_labels, open(os.path.join(DATASET_DIR, "Processed_train_Y_set.pickle"), "wb"))

	pickle.dump(dev_feat, open(os.path.join(DATASET_DIR, "Processed_dev_X_set.pickle"), "wb"))
	print("Saving to", os.path.join(DATASET_DIR, "Processed_dev_Y_set.pickle"))
	pickle.dump(dev_labels, open(os.path.join(DATASET_DIR, "Processed_dev_Y_set.pickle"), "wb"))

	print("Saving to", os.path.join(DATASET_DIR, "Processed_test_X_set.pickle"))
	pickle.dump(test_feat, open(os.path.join(DATASET_DIR, "Processed_test_X_set.pickle"), "wb"))
	print("Saving to", os.path.join(DATASET_DIR, "Processed_test_Y_set.pickle"))
	pickle.dump(test_labels, open(os.path.join(DATASET_DIR, "Processed_test_Y_set.pickle"), "wb"))


