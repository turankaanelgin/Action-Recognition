import tensorflow as tf
import pickle
import os
import numpy as np
from skimage.measure import block_reduce

CATEGORIES = ["walking", "jogging", "running", "boxing", "handwaving", "handclapping"]
DATASET_DIR = "../../data"

#### Parameters #####
averagingCoffient = 1
timestepCoffient = 15 # TODO decrease
maxpoolShrink = 128
n_epochs =1000
n_layers =2
n_neurons=20
learning_rate=0.001
dropout_rate = 0.9 # TODO decrease
n_outputs= len(CATEGORIES)
###################

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def getAUC (true , output):
    correct = 0
    for i in range(true.shape[0]):
        #print( true[i] , output[i], "true", np.argmax(true[i])+1 ,"predicted" , np.argmax(output[i])+1)
        # print("true", np.argmax(true[i])+1 ,"predicted" , np.argmax(output[i])+1)
        if(np.argmax(true[i]) == np.argmax(output[i])):
            correct+=1
    return float(correct)/true.shape[0]


def labels2indices(label):
    oneHot = np.zeros((len(CATEGORIES),1))
    oneHot[CATEGORIES.index(label)]=1
    return oneHot

def concat_data(data, flows):
	combined_data = []
	assert len(data) == len(flows)
	for i in range(len(data)):
		frame = data[i]["frames"][1:]
		flow = flows[i]["frames"]
		comb = np.concatenate((frame, flow), axis=1)
		ex = {"frames": comb, "category": data[i]["category"]}
		combined_data.append(ex)
	return combined_data

#Averages the frames then takes maxpool then puts them into timesteps
def preparedata(data):
	labels =[]
	frames =[]
	shrinkCof =int(8192/maxpoolShrink)

	for i in range (len(data)):
		# print ("data[i][filename]" , data[i]["filename"] , "category " , data[i]["category"] ,len(data[i]["frames"]) ,data[i]["frames"][0].shape )
		AveragedDataframes =[]
		j=0
		while j+averagingCoffient < len(data[i]["frames"]):
			AveragedDataframe =  data[i]["frames"][j]
			for k in range (1,averagingCoffient):
				AveragedDataframe+=data[i]["frames"][j+k]
			AveragedDataframe=AveragedDataframe/averagingCoffient
			j+=averagingCoffient
			AveragedDataframe =block_reduce(AveragedDataframe, (shrinkCof,), np.max)
			# print (AveragedDataframe.shape)
			AveragedDataframes.append(AveragedDataframe)

		data[i]["frames"] = AveragedDataframes
		# print ("data[i][filename]" , data[i]["filename"] , "category " , data[i]["category"] ,len(data[i]["frames"]) ,data[i]["frames"][0].shape )
		n=0

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
	X =  tf.placeholder(tf.float32, shape=[None, timestepCoffient,maxpoolShrink], name="X")
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


train = pickle.load(open(os.path.join(DATASET_DIR, "train_alexnet.pickle"), "rb"))
train_flows = pickle.load(open(os.path.join(DATASET_DIR, "train_flow_alexnet.pickle"), "rb"))
train = concat_data(train, train_flows)
trainingSamples, trainingLabels = preparedata(train)
del train
del train_flows
print("Training set was loaded...")

dev = pickle.load(open(os.path.join(DATASET_DIR, "dev_alexnet.pickle"), "rb"))
dev_flows = pickle.load(open(os.path.join(DATASET_DIR, "dev_flow_alexnet.pickle"), "rb"))
dev = concat_data(dev, dev_flows)
devSamples, devLabels = preparedata(dev)
del dev
del dev_flows
print("Dev set was loaded...")

test = pickle.load(open(os.path.join(DATASET_DIR, "test_alexnet.pickle"), "rb"))
test_flows = pickle.load(open(os.path.join(DATASET_DIR, "test_flow_alexnet.pickle"), "rb"))
test = concat_data(test, test_flows)
testingSamples, testingLabels = preparedata(test)
del test
del test_flows
print("Testing set was loaded...")

newTrainingSamples = np.vstack((trainingSamples,devSamples))
newTrainingLabels = np.vstack((trainingLabels,devLabels))
del trainingSamples
del devSamples
del trainingLabels
del devLabels

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
