# MSA Phase 2 Project - AI CIFAR-10 Program
# Author: King Hang Tam

import keras
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# from keras.models import Sequential, load_model
# from keras.layers import Input, Dense, Conv2D, LSTM, Bidirectional
# from keras.layers import BatchNormalization, Activation, Flatten, TimeDistributed, Reshape
# from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix

import os

# ===== Settings =====

# Filepaths to the locations for input and saved data
CIFAR_10_dir = "./cifar-10-batches-py/"
CIFAR_10_path = CIFAR_10_dir + "data_batch_1"
dataset_save_dir = "./cifar-10-npy/"
dataset_save_path = dataset_save_dir + "cifar-10-batch-1.npy"

# ====================

def setup():
	'''Set up folder structure for the program'''
	assert os.path.exists(CIFAR_10_dir) or os.path.exists(dataset_save_dir), "No input data folder/files found"
	
	# Create any missing folders for loading and saving data
	if not os.path.exists(dataset_save_dir):
		os.mkdir(dataset_save_dir)

def unpickle(file):		# Function taken from: https://www.cs.toronto.edu/~kriz/cifar.html
	import pickle
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict

def load_from_file(CIFAR_10_path=CIFAR_10_path, save_path=dataset_save_path):
	'''Load the dataset from file'''
	if not os.path.exists(save_path):
		dataset_dict = unpickle(CIFAR_10_path)
		
		data = dataset_dict[b"data"]
		ground_truth = dataset_dict[b"labels"]
		ground_truth = np.array(ground_truth)
		
		dataset = np.c_[data, ground_truth]
		np.save(save_path, dataset)
	else:
		dataset = np.load(save_path)
		
		data = dataset[:,:-1]
		ground_truth = dataset[:,-1]
	
	return data, ground_truth

def load_data():
	''''Load train and test datasets'''
	
	# for i in range(1,6):
	for i in range(1,2):
		fpath = CIFAR_10_dir + "data_batch_%d"%i
		save_path = dataset_save_dir + "cifar-10-batch-%d.npy"%i
		x_train, y_train = load_from_file(CIFAR_10_path=fpath, save_path=save_path)
	
	fpath = CIFAR_10_dir + "test_batch"
	save_path = dataset_save_dir + "cifar-10-batch-test.npy"
	x_test, y_test = load_from_file(CIFAR_10_path=fpath, save_path=save_path)
	
	label_names = unpickle(CIFAR_10_dir + "batches.meta")[b"label_names"]
	label_names = [n.decode('utf-8') for n in label_names]
	# print(label_names)
	
	return x_train, y_train, x_test, y_test, label_names
	
def adv_training_tut(x_train, y_train, x_test, y_test):

	'''Flattened features NN version'''
	
	# model = keras.models.Sequential()
	# model.add(keras.layers.Flatten(input_shape=(3072.)))
	# model.add(keras.layers.Dense(64, activation='relu'))
	# model.add(keras.layers.Dense(32, activation='relu'))
	# model.add(keras.layers.Dense(10, activation='softmax'))    # 10 output classes, converts to probabilities summing to 1
	# model.summary()
	
	# model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
	
	# training = model.fit(x_train, y_train, epochs=1, validation_split=0.33)
	
	# training.history.keys()

	# loss = training.history['loss']
	# val_loss = training.history['val_loss']
	# ax = pd.DataFrame(loss).plot()
	# ax = pd.DataFrame(val_loss).plot(ax=ax)
	# ax.legend(['loss', 'val_loss'])
	# plt.show()
	
	# plt.imshow(x_test[0], cmap='gray')
	# plt.show()
	
	# pred = model.predict(x_test[0:1])
	
	# pred = model.predict(x_test)
	# y_pred = np.argmax(pred, axis=1)
	# y_pred.shape
	
	# confusion_matrix(y_test, y_pred)
	# sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, ax=ax)
	
	
	'''CNN version'''
	
	# # Add extra pixels dimension (1 in this case; 3 if rgb)
	# x_train = x_train.reshape((-1, 3072, 1))  # Shape is inferred if set as -1 (for one dim max)
	# x_test = x_test.reshape((-1, 3072, 1))
	
	# Convolution NN model

	model = keras.models.Sequential()
	model.add(keras.layers.Conv2D(16, kernel_size=(3,3), activation='relu', input_shape=(32, 32, 3)))
	model.add(keras.layers.Conv2D(8, kernel_size=(3,3), activation='relu'))
	model.add(keras.layers.Flatten())
	model.add(keras.layers.Dense(10, activation='softmax'))    # 10 output classes, converts to probabilities summing to 1

	model.summary()
	
	model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
	training = model.fit(x_train, y_train, epochs=10, validation_split=0.33)
	training.history.keys()
	
	loss = training.history['loss']
	val_loss = training.history['val_loss']
	ax = pd.DataFrame(loss).plot()
	ax = pd.DataFrame(val_loss).plot(ax=ax)
	ax.legend(['loss', 'val_loss'])
	# plt.show()
	
	y_pred = model.predict(x_test, verbose=1)
	print(np.argmax(y_pred))
	
	print(y_pred.shape)
	y_pred = np.argmax(y_pred, axis=1)
	print(y_pred.shape)
	
	plt.clf()
	sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)
	plt.show()
	
	
def main():
	print("- Program running -")
	
	# Set up folder structure as needed
	setup()
	
	# Load dataset
	x_train, y_train, x_test, y_test, label_names = load_data()
	
	
	#=========================================
	
	'''Display Images...................'''
	
	# num_to_plot = 6
	# for i in range(num_to_plot):
		# plt.subplot(1,num_to_plot,i+1)
		# plt.imshow(x_train[i], cmap='gray')
	# plt.show()
	
	# plt.hist(x_train[0])
	# plt.show()
	
	#=========================================
	
	
	# Reshape the dataset into a shape that tensorflow expects: (samples, rows, cols, channels)
	# Could also use (samples, channels, rows, cols)
	# Note that the original data is stored as a flattened array in row-major order
	
	print(x_train.shape, '->')
	x_train = x_train.reshape(-1, 3, 32, 32)
	print(x_train.shape, '->')
	x_train = np.moveaxis(x_train, 1, -1)
	print(x_train.shape)
	
	x_test = x_test.reshape(-1, 3, 32, 32)
	x_test = np.moveaxis(x_test, 1, -1)
	
	print("\n=======\nNew shape:")
	print(x_train.shape)
	
	# Normalise data into the range [0,1]
	x_train = x_train / np.max(x_train)
	x_test = x_test / np.max(x_test)
	
	adv_training_tut(x_train, y_train, x_test, y_test)
	
	print("- Program ended -")


if __name__ == '__main__':
	main()
