import tensorflow as tf
import pickle
import os
import numpy as np
from random import shuffle


def getAUC (true , output):
    correct = 0
    for i in range(true.shape[0]):
        #print( true[i] , output[i], "true", np.argmax(true[i])+1 ,"predicted" , np.argmax(output[i])+1)
        if(np.argmax(true[i]) == np.argmax(output[i])):
            correct+=1
    return float(correct)/true.shape[0]


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

CATEGORIES = ["walking", "jogging", "running", "boxing", "handwaving", "handclapping"]
DATASET_DIR = "../data"


height = 80
width = 60
n_inputs =15

conv1_fmaps = 32
conv1_ksize = 3
conv1_pad = "SAME"


pool1_ksize = [1, 2, 2, 1]
pool1_stride = [1, 2, 2, 1]

conv2_fmaps = 64
conv2_ksize = [3, 3]
conv2_pad = "SAME"


pool2_ksize = [1, 2, 2, 1]
pool2_stride = [1, 2, 2, 1]
pool2_fmaps = conv2_fmaps

n_neurons = 150
n_outputs = 6
n_layers=3
learning_rate=0.01 # TODO tune

with tf.name_scope("inputs"):
    X = tf.placeholder(tf.float32, shape=[None,height,width,n_inputs], name="X")

    y = tf.placeholder(tf.int32, shape=[None,6], name="y")


conv1 = tf.layers.conv2d(X, filters=conv1_fmaps, kernel_size=conv1_ksize,
                         padding=conv1_pad, activation=tf.nn.relu,
                         name="conv1")
print("conv1",conv1)

with tf.name_scope("pool1"):
    pool1 = tf.nn.max_pool(conv1, ksize=pool1_ksize, strides=pool1_stride, padding="VALID")
    pool1_flat = tf.reshape(pool1, shape=[-1, conv1_fmaps, 40*30])
   
print("pool1",pool1)

conv2 = tf.layers.conv2d(pool1, filters=conv2_fmaps, kernel_size=conv2_ksize,padding=conv2_pad,
                         activation=tf.nn.relu, name="conv2")
print("conv2",conv2)

with tf.name_scope("pool2"):
    pool2 = tf.nn.max_pool(conv2, ksize=pool2_ksize, strides=pool2_stride, padding="VALID")
    pool2_flat = tf.reshape(pool2, shape=[-1, pool2_fmaps, 20 * 15])
    print("pool2_flat", pool2_flat)

##################

# with tf.name_scope("dense"):
#     dense = tf.layers.dense(inputs=pool1_flat, units=n_neurons, activation=tf.nn.relu)
#     print("dense", dense)

with tf.name_scope("dropout"):
    dropout = tf.layers.dropout(inputs=pool2_flat, rate=0.5)
    print("dropout", dropout)

##################

with tf.name_scope("lstm_cell"):
    lstm_cells = [tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons)
              for layer in range(n_layers)]
    print("lstm_cells",lstm_cells)

with tf.name_scope("multi_cell"):
    multi_cell = tf.contrib.rnn.OutputProjectionWrapper(
        tf.contrib.rnn.MultiRNNCell(lstm_cells),output_size=n_outputs)
    print("multi_cell",multi_cell)

with tf.name_scope("outputs"):
    outputs, states = tf.nn.dynamic_rnn(multi_cell, dropout, dtype=tf.float32)
    print("outputs",outputs)

with tf.name_scope("logits"):
    logits=outputs[:,-1,:]
    print("logits",logits)

with tf.name_scope("train"):
    xentropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

    # AdamOptimizer -> GradientDescentOptimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope("accuracy"):
    preds = tf.nn.softmax(logits)

with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

# Select 2000 random examples from training data
train = pickle.load(open(os.path.join(DATASET_DIR, "Processed_train_X_set.pickle"), "rb"))
train_labels =pickle.load(open(os.path.join(DATASET_DIR, "Processed_train_Y_set.pickle"), "rb"))
train_set = list(zip(train, train_labels))
shuffle(train_set)
train = list(x[0] for x in train_set)
train_labels = list(x[1] for x in train_set)
train = np.array(train)
train_labels = np.array(train_labels)
train = train[0:2000]
train_labels = train_labels[0:2000]
print("Training set was loaded...")

# Select 2000 random examples from test data
test = pickle.load(open(os.path.join(DATASET_DIR, "Processed_test_X_set.pickle"), "rb"))
test_labels = pickle.load(open(os.path.join(DATASET_DIR, "Processed_test_Y_set.pickle"), "rb"))
test_set = list(zip(test, test_labels))
shuffle(test_set)
test = list(x[0] for x in test_set)
test_labels = list(x[1] for x in test_set)
test = np.array(test)
test_labels = np.array(test_labels)
test = test[0:2000]
test_labels = test_labels[0:2000]
print("Test set was loaded...")


n_epochs = 200
batch_size = 100 # might be reduced for memory issues
train=train.reshape((train.shape[0] ,train.shape[2], train.shape[3] , train.shape[1] ))
train_labels=train_labels.reshape((train_labels.shape[0] ,train_labels.shape[1] ))

test=test.reshape((test.shape[0] ,test.shape[2], test.shape[3] , test.shape[1] ))
test_labels=test_labels.reshape((test_labels.shape[0] ,test_labels.shape[1] ))

# for i in range(len(train_labels)):
#     print(train_labels[i])

print(len(train))
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        train, train_labels= unison_shuffled_copies(train, train_labels)
        for i in range(0 , len(train)-batch_size,batch_size):
            # print(epoch, i)
            X_batch =train[i:i+batch_size]

            Y_batch = train_labels[i:i+batch_size]
            
            sess.run(training_op, feed_dict={X: X_batch, y: Y_batch})

            sess.run(training_op, feed_dict={X: X_batch, y: Y_batch})
            pred_batch = preds.eval(feed_dict={X: X_batch, y: Y_batch})
            xentropy_batch = loss.eval(feed_dict={X: X_batch, y: Y_batch})
            acc_batch = getAUC(Y_batch , pred_batch)
            print(i, "Batch accuracy:", acc_batch, "xentropy_batch" ,  xentropy_batch)

        
        pred_train = preds.eval(feed_dict={X: train, y: train_labels})
        acc_train = getAUC(train_labels , pred_train)
        pred_test = preds.eval(feed_dict={X: test, y: test_labels})
        acc_test = getAUC(test_labels , pred_test)

        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
    saver.save(sess, open(os.path.join(DATASET_DIR, "cnn_raw_frame")))