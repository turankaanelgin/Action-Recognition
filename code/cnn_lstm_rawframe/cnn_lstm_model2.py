import tensorflow as tf
import pickle
import os
from pandas import read_csv, DataFrame 
import numpy as np
CATEGORIES = ["walking", "jogging", "running", "boxing", "handwaving", "handclapping"]
DATASET_DIR = "../../data"

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def  getAUC (true , output):
    correct = 0
    for i in range(true.shape[0]):
        #print( true[i] , output[i], "true", np.argmax(true[i])+1 ,"predicted" , np.argmax(output[i])+1)
        # print("true", np.argmax(true[i])+1 ,"predicted" , np.argmax(output[i])+1)
        if(np.argmax(true[i]) ==    np.argmax(output[i])):
            correct+=1
    return float(correct)/true.shape[0]

def expandLabes(npArray , num):
    newNpArray = np.zeros((npArray.shape[0]*num,npArray.shape[1]))
    n=0
    for i in range (npArray.shape[0]):
        newNpArray[n:n+num,:] = npArray[i,:]
        n+=num
    return newNpArray



height = 80
width = 60
channels = 1
n_inputs = height * width

conv1_fmaps = 64
conv1_ksize = 5
conv1_stride = 1
conv1_pad = "SAME"

conv2_fmaps = 64
conv2_ksize = 3
conv2_stride = 1
conv2_pad = "SAME"
conv2_dropout_rate = 0.25

pool3_fmaps = conv2_fmaps

n_fc1 = 128
fc1_dropout_rate = 0.5

n_outputs = 6
n_steps=15
n_layers =2
n_neurons=250

learning_rate=0.01
with tf.name_scope("inputs"):
    X = tf.placeholder(tf.float32, shape=[None, height, width], name="X")
    X_reshaped = tf.reshape(X, shape=[-1, height, width, channels])
    y = tf.placeholder(tf.int32, shape=[None,n_outputs], name="y")
    LSTM_X =  tf.placeholder(tf.float32, shape=[None, n_steps,n_fc1], name="X")
    LSTM_Y =  tf.placeholder(tf.int32, shape=[None,n_outputs], name="LSTM_Y")

    training = tf.placeholder_with_default(False, shape=[], name='training')


conv1 = tf.layers.conv2d(X_reshaped, filters=conv1_fmaps, kernel_size=conv1_ksize,
                         strides=conv1_stride, padding=conv1_pad,
                         activation=tf.nn.relu, name="conv1")
print(conv1)
conv2 = tf.layers.conv2d(conv1, filters=conv2_fmaps, kernel_size=conv2_ksize,
                         strides=conv2_stride, padding=conv2_pad,
                         activation=tf.nn.relu, name="conv2")
print(conv2)
with tf.name_scope("pool3"):
    pool3 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
    pool3_flat = tf.reshape(pool3, shape=[-1, pool3_fmaps * 40 * 30])
    pool3_flat_drop = tf.layers.dropout(pool3_flat, conv2_dropout_rate, training=training)
print(pool3_flat_drop)

with tf.name_scope("fc1"):
    fc1 = tf.layers.dense(pool3_flat_drop, n_fc1, activation=tf.nn.relu, name="fc1")
    fc1_drop = tf.layers.dropout(fc1, fc1_dropout_rate, training=training)
print(fc1_drop)

with tf.name_scope("cnn_output"):
    logits = tf.layers.dense(fc1, n_outputs, name="output")
    print (logits)
    Y_proba = tf.nn.softmax(logits, name="Y_proba")

with tf.name_scope("train"):
    xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)


with tf.name_scope("lstm_cell"):
    lstm_cells = [tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons)
              for layer in range(n_layers)]


with tf.name_scope("multi_cell"):
    multi_cell = tf.contrib.rnn.OutputProjectionWrapper(
        tf.contrib.rnn.MultiRNNCell(lstm_cells),output_size=n_outputs)

with tf.name_scope("lstm_outputs"):
    outputs, states = tf.nn.dynamic_rnn(multi_cell, LSTM_X, dtype=tf.float32)
    print("lstm_outputs",outputs)


with tf.name_scope("lstm_train"):
    lstmxentropy = tf.nn.softmax_cross_entropy_with_logits(labels=LSTM_Y, logits=outputs[:,-1,:])
    lstmloss = tf.reduce_mean(lstmxentropy, name="lstmloss")
    lstmoptimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    lstmtraining_op = lstmoptimizer.minimize(lstmloss)
    lstmY_proba = tf.nn.softmax(outputs[:,-1,:], name="lstmY_proba")

with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()


train = pickle.load(open(os.path.join(DATASET_DIR, "Processed_train_X_set.pickle"), "rb"))
train_labels =pickle.load(open(os.path.join(DATASET_DIR, "Processed_train_Y_set.pickle"), "rb"))
print("Training set was loaded...")

test = pickle.load(open(os.path.join(DATASET_DIR, "Processed_test_X_set.pickle"), "rb"))
test_labels = pickle.load(open(os.path.join(DATASET_DIR, "Processed_test_Y_set.pickle"), "rb"))
print("Test set was loaded...")


n_epochs = 200
print (train.shape)
print (train_labels.shape)
train  ,  train_labels= unison_shuffled_copies(train, train_labels)
train=train.reshape((train.shape[0]*train.shape[1], train.shape[2] , train.shape[3]))
train_labels=train_labels.reshape((train_labels.shape[0] ,train_labels.shape[1] ))

expanddedtrain_labels = expandLabes(train_labels,15)



test=test.reshape((test.shape[0]*  test.shape[1],test.shape[2], test.shape[3] ))
test_labels=test_labels.reshape((test_labels.shape[0] ,test_labels.shape[1] ))
expanddedtest_labels = expandLabes(test_labels,15)



with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        j=0
        for i in range(0 , len(train)-1500,1500):
            # print(epoch, i)
            X_batch =train[i:i+1500] 

            Y_batch = expanddedtrain_labels[i:i+1500] 

            sess.run(training_op, feed_dict={X: X_batch, y: Y_batch})
            pred_batch = Y_proba.eval(feed_dict={X: X_batch, y: Y_batch})
            xentropy_batch = loss.eval(feed_dict={X: X_batch, y: Y_batch})
            acc_batch = getAUC(Y_batch , pred_batch)
            print(i, "Batch accuracy:", acc_batch, "xentropy_batch" ,  xentropy_batch)
            
            last_layer = fc1.eval(feed_dict={X: X_batch, y: Y_batch})
            # print(last_layer.shape)
            lstm_batch = last_layer.reshape((int(last_layer.shape[0]/15),15,last_layer.shape[1]))
            # print(lstm_batch.shape)  
            Y_lstm = train_labels[j:j+100]
            j+=100
            # print(Y_lstm.shape)
            sess.run(lstmtraining_op, feed_dict={LSTM_X: lstm_batch, LSTM_Y: Y_lstm})
            lstmpred_batch = lstmY_proba.eval(feed_dict={LSTM_X: lstm_batch, LSTM_Y: Y_lstm})
            lstmxentropy_batch = lstmloss.eval(feed_dict={LSTM_X: lstm_batch, LSTM_Y: Y_lstm})


            acc_batch = getAUC(Y_lstm , lstmpred_batch)
            print(i, "lstm Batch accuracy:", acc_batch, " lstm xentropy_batch" ,  lstmxentropy_batch)



        
        last_cnnTrain = fc1.eval(feed_dict={X: train, y: expanddedtrain_labels})
        lstm_train = last_cnnTrain.reshape((int(last_cnnTrain.shape[0]/15),15,last_cnnTrain.shape[1]))
        lstmpred = lstmY_proba.eval(feed_dict={LSTM_X: lstm_train, LSTM_Y: train_labels})
        acc_train = getAUC(train_labels , lstmpred)
      
        last_cnnTest = fc1.eval(feed_dict={X: test, y: expanddedtest_labels})
        lstm_test = last_cnnTest.reshape((int(last_cnnTest.shape[0]/15),15,last_cnnTest.shape[1]))
        lstmpredtest = lstmY_proba.eval(feed_dict={LSTM_X: lstm_test, LSTM_Y: test_labels})
        acc_test = getAUC(test_labels , lstmpredtest)

        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
        
    saver.save(sess, open(os.path.join(DATASET_DIR, "cnn_raw_frame")))