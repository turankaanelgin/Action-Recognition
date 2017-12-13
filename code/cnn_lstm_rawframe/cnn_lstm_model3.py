import tensorflow as tf
import pickle
import os
import numpy as np
from skimage.measure import block_reduce

CATEGORIES = ["walking", "jogging", "running", "boxing", "handwaving", "handclapping"]
DATASET_DIR = "../../data"

#### Parameters #####
averagingCoffient = 10
timestepCoffient = 5
maxpoolShrink = 128
n_epochs = 1000
n_layers = 2
n_neurons = 15
learning_rate = 0.001
dropout_rate = 0.1
n_outputs= len(CATEGORIES)
###################

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def getAUC (true , output):
    correct = 0
    for i in range(true.shape[0]):
        if(np.argmax(true[i]) == np.argmax(output[i])):
            correct+=1
    return float(correct)/true.shape[0]

def combine_data(data_x, data_y, categories):
    optical_flows = []
    for i in range(len(data_x)):
        video_x = data_x[i]
        video_y = data_y[i]
        flows_per_video = []
        for j in range(len(video_x)):
            flows_x = video_x[j]
            flows_y = video_y[j]
            flows = list(zip(flows_x, flows_y))
            flows_per_video.append(flows)
        optical_flows.append({'category': categories[i],
                              'frames': flows_per_video})
    return optical_flows

def labels2indices(label):
    oneHot = np.zeros((len(CATEGORIES),1))
    oneHot[CATEGORIES.index(label)]=1
    return oneHot

def concat_data(data, flow_x, flow_y):
	combined_data = []
	cnt = 0
	for i in range(len(data)):
		frame = data[i]["frames"][1:]
		flowx = flow_x[i]
		flowy = flow_y[i]
		frames_per_video = np.concatenate((frame, flowx), axis=1)
		frames_per_video = np.concatenate((frames_per_video, flowy), axis=1)
		frames_per_video = np.array(frames_per_video)
		ex = {"frames": frames_per_video, "category": data[i]["category"]}
		combined_data.append(ex)
		print("Video %d was processed" % cnt)
		cnt += 1
	return combined_data

def concat_flows(data, flow_x, flow_y):
	combined_data = []
	cnt = 0
	for i in range(len(data)):
		flowx = flow_x[i]
		flowy = flow_y[i]
		flows_per_video = np.concatenate((flowx, flowy), axis=1)
		ex = {"frames": flows_per_video, "category": data[i]["category"]}
		combined_data.append(ex)
		print("Video %d was processed" % cnt)
		cnt += 1
	return combined_data

def linearize_data(data):
	linear_data = []
	for video in data:
		linear_frames = []
		frames = video["frames"]
		for frame in frames:
			frame = np.reshape(frame, (np.size(frame), ))
			linear_frames.append(frame)
		linear_data.append({"frames": linear_frames, "category": video["category"]})
	return linear_data

def extract_labels(data):
	labels = []
	cnt = 0
	for video in data:
		for frame in video["frames"]:
			labels.append(labels2indices(video["category"]))
	return labels

def extract_samples(data):
	samples = []
	cnt = 0
	for video in data:
		for frame in video["frames"]:
			samples.append(frame)
	return samples

#Averages the frames then takes maxpool then puts them into timesteps
def preparedata(data):
	labels =[]
	frames =[]
	shrinkCof =int(216/maxpoolShrink)

	for i in range (len(data)):
		# AveragedDataframes =[]
		# j=0
		# while j+averagingCoffient < len(data[i]["frames"]):
		# 	AveragedDataframe =  data[i]["frames"][j]
		# 	for k in range (1,averagingCoffient):
		# 		AveragedDataframe+=data[i]["frames"][j+k]
		# 	AveragedDataframe=AveragedDataframe/averagingCoffient
		# 	j+=averagingCoffient
		# 	AveragedDataframe =block_reduce(AveragedDataframe, (shrinkCof, ), np.max)
		# 	# print (AveragedDataframe.shape)
		# 	AveragedDataframes.append(AveragedDataframe)
        #
		# data[i]["frames"] = AveragedDataframes

		# reducedFrames = []
		# j = 0
		# for frame in data[i]["frames"]:
		# 	if j == 0:
		# 		frame = block_reduce(frame, (shrinkCof,), np.max)
		# 		reducedFrames.append(frame)
		# 	j += 1
		# 	if j == averagingCoffient:
		# 		j = 0
		# data[i]["frames"] = reducedFrames

		n = 0
		while n+timestepCoffient < len(data[i]["frames"]):
			stepsDataframes =[]
			for k in range (timestepCoffient):
				stepsDataframes.append(data[i]["frames"][n+k])
			stepsDataframes = np.array(stepsDataframes)
			stepsDataframes = stepsDataframes.reshape(timestepCoffient,stepsDataframes.shape[1])
			n+=timestepCoffient
			frames.append(stepsDataframes)
			# print(data[i]["category"])
			label = labels2indices(data[i]["category"])
			labels.append(label)
	# 	print("")
	# print(len(frames))
	frames = np.array(frames)
	labels =  np.array(labels)
	labels = labels.reshape(labels.shape[0], labels.shape[1])
	# print (frames.shape ,  labels.shape)
	return frames ,  labels


with tf.name_scope("inputs"):
	X =  tf.placeholder(tf.float32, shape=[None, timestepCoffient,128], name="X")
	y =  tf.placeholder(tf.int32, shape=[None,n_outputs], name="Y")
	keep_prob = tf.placeholder(tf.float32)


with tf.name_scope("lstm_cell"):
	lstm_cells = [tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons)
                  for layer in range(n_layers)]


with tf.name_scope("multi_cell"):
    multi_cell = tf.contrib.rnn.OutputProjectionWrapper(
        tf.contrib.rnn.MultiRNNCell(lstm_cells),output_size=n_outputs)

with tf.name_scope("lstm_outputs"):
    outputs, states = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32)
    rnn_outputs = tf.nn.dropout(outputs, keep_prob)
    # print("lstm_outputs",outputs)


with tf.name_scope("lstm_train"):
    lstmxentropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=rnn_outputs[:,-1,:])
    lstmloss = tf.reduce_mean(lstmxentropy, name="lstmloss")
    lstmoptimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    #lstmoptimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9,
								#use_locking=False, name='Momentum', use_nesterov=False)
    lstmtraining_op = lstmoptimizer.minimize(lstmloss)
    lstmY_proba = tf.nn.softmax(rnn_outputs[:,-1,:], name="lstmY_proba")


with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()


# train = pickle.load(open(os.path.join(DATASET_DIR, "train_set.pickle"), "rb"))
# train_flow_x = pickle.load(open(os.path.join(DATASET_DIR, "train_dense_flow_x.pickle"), "rb"))
# train_flow_y = pickle.load(open(os.path.join(DATASET_DIR, "train_dense_flow_y.pickle"), "rb"))
# train = linearize_data(train)
# print("Training set was linearized...")
# train = concat_flows(train, train_flow_x, train_flow_y)
# print("Training set was concatenated...")
# trainingSamples, trainingLabels = preparedata(train)
# del train
# del train_flow_x
# del train_flow_y
# print("Training set was loaded...")
#
# dev = pickle.load(open(os.path.join(DATASET_DIR, "dev_set.pickle"), "rb"))
# dev_flow_x = pickle.load(open(os.path.join(DATASET_DIR, "dev_dense_flow_x.pickle"), "rb"))
# dev_flow_y = pickle.load(open(os.path.join(DATASET_DIR, "dev_dense_flow_y.pickle"), "rb"))
# dev = linearize_data(dev)
# print("Dev set was linearized...")
# dev = concat_flows(dev, dev_flow_x, dev_flow_y)
# print("Dev set was concatenated...")
# devSamples, devLabels = preparedata(dev)
# del dev
# del dev_flow_x
# del dev_flow_y
# print("Dev set was loaded...")
#
# test = pickle.load(open(os.path.join(DATASET_DIR, "test_set.pickle"), "rb"))
# test_flow_x = pickle.load(open(os.path.join(DATASET_DIR, "test_dense_flow_x.pickle"), "rb"))
# test_flow_y = pickle.load(open(os.path.join(DATASET_DIR, "test_dense_flow_y.pickle"), "rb"))
# test = linearize_data(test)
# print("Test set was linearized...")
# test = concat_flows(test, test_flow_x, test_flow_y)
# print("Test set was concatenated...")
# testingSamples, testingLabels = preparedata(test)
# del test
# del test_flow_x
# del test_flow_y
# print("Testing set was loaded...")

train = pickle.load(open(os.path.join(DATASET_DIR, "train_3dcnn.pickle"), "rb"))
trainingSamples, trainingLabels = preparedata(train)
del train

dev = pickle.load(open(os.path.join(DATASET_DIR, "dev_3dcnn.pickle"), "rb"))
devSamples, devLabels = preparedata(dev)
del dev

test = pickle.load(open(os.path.join(DATASET_DIR, "test_3dcnn.pickle"), "rb"))
testingSamples, testingLabels = preparedata(test)
del test

newTrainingSamples = np.vstack((trainingSamples,devSamples))
newTrainingLabels = np.vstack((trainingLabels,devLabels))
del trainingSamples
del devSamples
del trainingLabels
del devLabels
# newTrainingSamples = trainingSamples
# newTrainingLabels = trainingLabels
# print(np.size(trainingSamples, 0))

with tf.Session() as sess:
	init.run()
	for epoch in range(n_epochs):
		X_train , Y_train = unison_shuffled_copies(newTrainingSamples,newTrainingLabels)
		X_test , Y_test = testingSamples , testingLabels

		sess.run(lstmtraining_op, feed_dict={X: X_train, y: Y_train  , keep_prob:dropout_rate})
		xentropy_train = lstmloss.eval(feed_dict={X: X_train, y: Y_train ,keep_prob:1.0})
		xentropy_test = lstmloss.eval(feed_dict={X: X_test, y: Y_test,keep_prob:1.0})
		trainPred =  lstmY_proba.eval(feed_dict={X: X_train, y: Y_train ,keep_prob:1.0})
		testPred =   lstmY_proba.eval(feed_dict={X: X_test, y: Y_test,keep_prob:1.0})
		acc_train =getAUC(Y_train , trainPred)
		acc_test = getAUC(Y_test , testPred)
		print("Epoch", epoch,  "Train xentropy =" ,xentropy_train, "Train accuracy =", acc_train,
			  "Test xentropy =" ,xentropy_test, "Test accuracy =", acc_test)

	FinalTrainPred =  lstmY_proba.eval(feed_dict={X: newTrainingSamples, y: newTrainingLabels ,keep_prob:1.0})
	FinalTestPred =   lstmY_proba.eval(feed_dict={X: testingSamples, y: testingLabels,keep_prob:1.0})
	acc_train =getAUC(newTrainingLabels , FinalTrainPred)
	acc_test = getAUC(testingLabels , FinalTestPred)
	print("Epoch", epoch,   "Train accuracy =", acc_train,  "Test accuracy =", acc_test)
	saver.save(sess, open(os.path.join(DATASET_DIR, "cnn_raw_frame")))
