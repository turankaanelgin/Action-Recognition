# Trained model cnn_3conv2d_drop05_15.h5 can be downloaded at:
# https://drive.google.com/open?id=14v9D__AlRU7JCg0s2-mF4Z-K5dL3G2Ht

import numpy as np
import keras
from keras import layers
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Dense, Activation, ZeroPadding2D, Flatten, Conv2D, Dropout
from keras.layers import MaxPooling2D
from keras.models import Model

import pickle

def load_dataset(dataset_name):
	X = pickle.load(open("../../data/cnn_keras/X_" + dataset_name + ".pickle", "rb"))
	Y = pickle.load(open("../../data/cnn_keras/Y_" + dataset_name + ".pickle", "rb"))
	return X, Y

def convert_to_one_hot(Y, n_classes):
	one_hot = np.zeros((Y.shape[0], n_classes), dtype=np.uint8)
	one_hot[range(Y.shape[0]), Y] = 1
	return one_hot

def cnn_model(input_shape):
    
    X_input = Input(input_shape)
    
    X = ZeroPadding2D((2, 2))(X_input)
    X = Conv2D(64, (5, 5), strides=(1, 1))(X)
    X = Activation("relu")(X)
    X = MaxPooling2D((2, 2))(X)
    
    X = ZeroPadding2D((1, 1))(X)
    X = Conv2D(128, (3, 3), strides=(1, 1))(X)
    X = Activation("relu")(X)
    X = MaxPooling2D((2, 2))(X)
    
    X = ZeroPadding2D((1, 1))(X)
    X = Conv2D(256, (3, 3), strides=(1, 1))(X)
    X = Activation("relu")(X)
    X = MaxPooling2D((2, 2))(X)
    X = Dropout(0.5)(X)

    X = Flatten()(X)
    X = Dense(256, activation="relu")(X)
    X = Dense(6, activation="softmax")(X)

    model = Model(inputs=X_input, outputs=X)
    
    return model

if __name__ == "__main__":
	# Load original dataset.
	X_train_orig, Y_train_orig = load_dataset("train")

	# Normalize dataset and create one-hot encoding for labels.
	X_train = (X_train_orig / 255.0).astype(np.float32)
	Y_train = (convert_to_one_hot(Y_train_orig, 6)).astype(np.float32)

	print ("number of training examples = " + str(X_train.shape[0]))
	print ("X_train shape: " + str(X_train.shape))
	print ("Y_train shape: " + str(Y_train.shape))

	model = cnn_model((120, 160, 15))
	model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])

	# Create checkpoint to save model after each epoch.
	filepath = "../../data/cnn_keras/cnn_3conv_drop04_{epoch:02d}.h5"
	checkpoint = ModelCheckpoint(filepath, verbose=1, period=1)

	# Train.
	model.fit(x=X_train, y=Y_train, epochs=10, batch_size=32,\
		callbacks=[checkpoint], validation_data=(X_dev, Y_dev))
	