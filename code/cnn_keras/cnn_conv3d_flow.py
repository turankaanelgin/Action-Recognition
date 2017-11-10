# Trained model cnn_conv3d_flow_10.h5 can be downloaded at:
# https://drive.google.com/open?id=14v9D__AlRU7JCg0s2-mF4Z-K5dL3G2Ht
# Test accuracy: 88.4%.
import argparse
import numpy as np
import keras
from keras import layers
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Dense, Activation, ZeroPadding2D, Flatten, Conv2D, Dropout, Conv3D, Concatenate
from keras.layers import MaxPooling2D, MaxPooling3D
from keras.models import Model

import pickle

def load_dataset():
	print("Loading dataset")

	X_train = pickle.load(open("../../data/cnn_keras/X_train_resize80x60.pickle", "rb"))
	X_flowx_train = pickle.load(open("../../data/cnn_keras/X_flowx_train.pickle", "rb"))
	X_flowy_train = pickle.load(open("../../data/cnn_keras/X_flowy_train.pickle", "rb"))
	Y_train = pickle.load(open("../../data/cnn_keras/Y_train.pickle", "rb"))

	X_dev = pickle.load(open("../../data/cnn_keras/X_dev_resize80x60.pickle", "rb"))
	X_flowx_dev = pickle.load(open("../../data/cnn_keras/X_flowx_dev.pickle", "rb"))
	X_flowy_dev = pickle.load(open("../../data/cnn_keras/X_flowy_dev.pickle", "rb"))
	Y_dev = pickle.load(open("../../data/cnn_keras/Y_dev.pickle", "rb"))

	return X_train, X_flowx_train, X_flowy_train,\
		Y_train, X_dev, X_flowx_dev, X_flowy_dev, Y_dev

def convert_to_one_hot(Y, n_classes):
	one_hot = np.zeros((Y.shape[0], n_classes), dtype=np.uint8)
	one_hot[range(Y.shape[0]), Y] = 1
	return one_hot

# Architecture of CNN model.
def cnn_model(raw_input_shape, flow_input_shape):

	X_raw_input = Input(raw_input_shape)
	X_flowx_input = Input(flow_input_shape)
	X_flowy_input = Input(flow_input_shape)

	# Raw frame.
	X_raw = Conv3D(32, (3, 3, 3), strides=(1, 1, 1))(X_raw_input)
	X_raw = Activation("relu")(X_raw)
	X_raw = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(X_raw)
	X_raw = Dropout(0.5)(X_raw)
	X_raw = Conv3D(64, (3, 3, 3), strides=(1, 1, 1))(X_raw)
	X_raw = Activation("relu")(X_raw)
	X_raw = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(X_raw)
	X_raw = Dropout(0.5)(X_raw)
	X_raw = Flatten()(X_raw)

	# Optical flow x.
	X_flowx = Conv3D(32, (3, 3, 3), strides=(1, 1, 1))(X_flowx_input)
	X_flowx = Activation("relu")(X_flowx)
	X_flowx = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(X_flowx)
	X_flowx = Dropout(0.5)(X_flowx)
	X_flowx = Conv3D(64, (3, 3, 3), strides=(1, 1, 1))(X_flowx)
	X_flowx = Activation("relu")(X_flowx)
	X_flowx = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(X_flowx)
	X_flowx = Dropout(0.5)(X_flowx)
	X_flowx = Flatten()(X_flowx)

	# Optical flow y.
	X_flowy = Conv3D(32, (3, 3, 3), strides=(1, 1, 1))(X_flowy_input)
	X_flowy = Activation("relu")(X_flowy)
	X_flowy = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(X_flowy)
	X_flowy = Dropout(0.5)(X_flowy)
	X_flowy = Conv3D(64, (3, 3, 3), strides=(1, 1, 1))(X_flowy)
	X_flowy = Activation("relu")(X_flowy)
	X_flowy = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(X_flowy)
	X_flowy = Dropout(0.5)(X_flowy)
	X_flowy = Flatten()(X_flowy)

	# Merge.
	X = Concatenate()([X_raw, X_flowx, X_flowy])

	X = Dense(150, activation="relu")(X)
	X = Dropout(0.5)(X)
	X = Dense(6, activation="softmax")(X)

	model = Model(inputs=[X_raw_input, X_flowx_input, X_flowy_input], outputs=X)

	return model

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--initial_epoch", type=int, default=0,\
		help="the initial number of epoch to start training (0 if traning from scratch)")
	parser.add_argument("--num_epochs", type=int, default=1,\
		help="the number of epochs to train")
	parser.add_argument("--model_path", type=str, default="",\
		help=("path to trained model file so that we can continue training"
			"(if initial epoch is zero, this can be empty)")

	# If training from scratch, and you want to train for 3 epochs, then call:
	# python cnn_conv3d_flow.py --initial_epoch=0 --num_epochs=3
	#
	# If you have trained 3 epochs, and the trained model file is cnn_conv3d_flow_03.h5,
	# and you want to continue training for 2 more epochs, then call:
	# python cnn_conv3d_flow.py --initial_epoch=3 --num_epochs=2 --model_path=cnn_conv3d_flow_03.h5

	args = parser.parse_args()
	initial_epoch = args.initial_epoch
	num_epochs = args.num_epochs
	model_path = args.model
	print(initial_epoch, model_path)

	# Load dataset.
	X_train, X_flowx_train, X_flowy_train,\
		Y_train, X_dev, X_flowx_dev, X_flowy_dev, Y_dev = load_dataset()
	
	# Normalize dataset and create one-hot encoding for labels.
	X_train = (X_train / 255.0).astype(np.float32)
	Y_train = (convert_to_one_hot(Y_train, 6)).astype(np.float32)
	X_dev = (X_dev / 255.0).astype(np.float32)
	Y_dev = (convert_to_one_hot(Y_dev, 6)).astype(np.float32) 

	print ("X_train shape: " + str(X_train.shape))
	print ("Y_train shape: " + str(Y_train.shape))
	print ("X_flowx_train shape: " + str(X_flowx_train.shape))
	print ("X_flowy_train shape: " + str(X_flowy_train.shape))

	if initial_epoch == 0:
		model = cnn_model((60, 80, 15, 1), (30, 40, 14, 1))
		model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["accuracy"])
	else:
		model = keras.models.load_model(model_path) 

	# Create checkpoint to save model after each epoch.
	filepath = "../../data/cnn_keras/cnn_conv3d_flow_{epoch:02d}.h5"
	checkpoint = ModelCheckpoint(filepath, verbose=1, period=1)

	# Train.
	print("Training")
	model.fit(x=[X_train[:,:,:,:,np.newaxis], X_flowx_train[:,:,:,:,np.newaxis],\
		X_flowy_train[:,:,:,:,np.newaxis]], y=Y_train, epochs=initial_epoch+num_epochs,\
		batch_size=32, callbacks=[checkpoint], validation_data=([X_dev[:,:,:,:,np.newaxis],\
		X_flowx_dev[:,:,:,:,np.newaxis], X_flowy_dev[:,:,:,:,np.newaxis]], Y_dev),\
		verbose=1, initial_epoch=initial_epoch)
