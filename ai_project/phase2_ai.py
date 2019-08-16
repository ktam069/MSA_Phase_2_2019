# MSA Phase 2 Project - AI CIFAR-10 Program
# Author: King Hang Tam

import keras
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from keras.models import Sequential, load_model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, GlobalAveragePooling2D
# from keras.layers import Input, BatchNormalization, Activation
# from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix

import os
import glob
from datetime import datetime

# ===== Settings =====

# Used (the latest) peviously saved model
USE_LOADED_MODEL = False

# Filepaths to the locations for input and saved data
CIFAR_10_dir = "./cifar-10-batches-py/"
CIFAR_10_path = CIFAR_10_dir + "data_batch_1"
dataset_save_dir = "./cifar-10-npy/"
dataset_save_path = dataset_save_dir + "cifar-10-batch-1.npy"
model_save_dir = "./saved_models/"

# ====================

def setup():
	'''Set up folder structure for the program'''
	assert os.path.exists(CIFAR_10_dir) or os.path.exists(dataset_save_dir), "No input data folder/files found"
	
	# Create any missing folders for loading and saving data
	if not os.path.exists(dataset_save_dir):
		os.mkdir(dataset_save_dir)
	if not os.path.exists(model_save_dir):
		os.mkdir(model_save_dir)

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

def load_data_batch(batch_no=1):
	''''Load train and test datasets'''
	
	fpath = CIFAR_10_dir + "data_batch_%d"%batch_no
	save_path = dataset_save_dir + "cifar-10-batch-%d.npy"%batch_no
	x_train, y_train = load_from_file(CIFAR_10_path=fpath, save_path=save_path)

	fpath = CIFAR_10_dir + "test_batch"
	save_path = dataset_save_dir + "cifar-10-batch-test.npy"
	x_test, y_test = load_from_file(CIFAR_10_path=fpath, save_path=save_path)
	
	label_names = unpickle(CIFAR_10_dir + "batches.meta")[b"label_names"]
	label_names = [n.decode('utf-8') for n in label_names]
	# print(label_names)
	
	return x_train, y_train, x_test, y_test, label_names

def load_data():
	''''Load train and test datasets'''
	
	x_train = None
	y_train = None
	
	for i in range(1,6):
		fpath = CIFAR_10_dir + "data_batch_%d"%i
		save_path = dataset_save_dir + "cifar-10-batch-%d.npy"%i
		x_data, y_data = load_from_file(CIFAR_10_path=fpath, save_path=save_path)
		
		if i == 1:
			x_train = x_data
			y_train = y_data
		else:
			x_train = np.append(x_train, x_data, axis=0)
			y_train = np.append(y_train, y_data, axis=0)
	
	fpath = CIFAR_10_dir + "test_batch"
	save_path = dataset_save_dir + "cifar-10-batch-test.npy"
	x_test, y_test = load_from_file(CIFAR_10_path=fpath, save_path=save_path)
	
	label_names = unpickle(CIFAR_10_dir + "batches.meta")[b"label_names"]
	label_names = [n.decode('utf-8') for n in label_names]
	# print(label_names)
	
	return x_train, y_train, x_test, y_test, label_names
	
def create_compiled_model():
	'''Convolution NN model'''

	model = Sequential()
	
	if False:
		# Convolution layers with max pooling
		
		model.add(Conv2D(16, kernel_size=(3,3), activation='relu', padding='same', input_shape=(32, 32, 3)))
		model.add(MaxPooling2D(pool_size=(2,2), strides=2))
		model.add(Conv2D(32, kernel_size=(3,3), activation='relu', padding='same'))
		model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))
		# model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))
		model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Conv2D(128, kernel_size=(3,3), activation='relu', padding='same'))
		# model.add(Conv2D(128, kernel_size=(3,3), activation='relu', padding='same'))
		model.add(MaxPooling2D(pool_size=(2,2)))
		# model.add(Dropout(0.25))
		
		model.add(Flatten())
		
		# Three fully connected layers (including the output layer)
		# model.add(Dense(128, activation='relu'))
		# model.add(Dense(256, activation='relu'))
		# model.add(Dropout(0.25))
		model.add(Dense(10, activation='softmax'))    # 10 output classes, as probabilities
	elif False:
		# Convolution layers with max pooling
		
		model.add(Conv2D(16, kernel_size=(3,3), activation='relu', padding='same', input_shape=(32, 32, 3)))
		model.add(Conv2D(16, kernel_size=(3,3), activation='relu', padding='same'))
		model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Conv2D(32, kernel_size=(3,3), activation='relu', padding='same'))
		model.add(Conv2D(32, kernel_size=(3,3), activation='relu', padding='same'))
		model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))
		model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))
		model.add(MaxPooling2D(pool_size=(2,2)))
		# model.add(Dropout(0.25))
		
		model.add(Flatten())
		
		# Three fully connected layers (including the output layer)
		# model.add(Dense(128, activation='relu'))
		# model.add(Dense(256, activation='relu'))
		# model.add(Dropout(0.25))
		model.add(Dense(10, activation='softmax'))    # 10 output classes, as probabilities
	elif True:
		# Model based on the paper, https://arxiv.org/pdf/1412.6806.pdf
		
		model.add(Conv2D(96, kernel_size=(3,3), activation='relu', padding='same', input_shape=(32, 32, 3)))
		model.add(Conv2D(96, kernel_size=(3,3), activation='relu', padding='same'))
		model.add(MaxPooling2D(pool_size=(2,2), strides=2))
		model.add(Conv2D(192, kernel_size=(3,3), activation='relu', padding='same'))
		model.add(Conv2D(192, kernel_size=(3,3), activation='relu', padding='same'))
		model.add(MaxPooling2D(pool_size=(2,2), strides=2))
		# model.add(Conv2D(192, kernel_size=(3,3), activation='relu', padding='same'))
		# model.add(Conv2D(192, kernel_size=(1,1), activation='relu', padding='same'))
		# model.add(Conv2D(10, kernel_size=(1,1), activation='relu', padding='same'))
		# model.add(MaxPooling2D(pool_size=(2,2), strides=2))
		# model.add(Dropout(0.25))
		
		model.add(Flatten())
		# model.add(GlobalAveragePooling2D())
		
		# Three fully connected layers (including the output layer)
		model.add(Dense(128, activation='relu'))
		# model.add(Dense(256, activation='relu'))
		# model.add(Dropout(0.25))
		model.add(Dense(10, activation='softmax'))    # 10 output classes, as probabilities
	
	else:
		# Convolution layers with max pooling
		model.add(Conv2D(96, kernel_size=(3,3), activation='relu', padding='same', input_shape=(32, 32, 3)))
		# model.add(MaxPooling2D(pool_size=(2,2), strides=2))
		# model.add(Conv2D(96, kernel_size=(3,3), activation='relu', padding='same'))
		# model.add(MaxPooling2D(pool_size=(2,2), strides=2))
		model.add(Conv2D(192, kernel_size=(3,3), activation='relu', padding='same'))
		# model.add(Conv2D(192, kernel_size=(3,3), activation='relu', padding='same'))
		# model.add(MaxPooling2D(pool_size=(2,2), strides=2))
		# model.add(Dropout(0.25))
		
		model.add(Flatten())
		
		# Three fully connected layers (including the output layer)
		# model.add(Dense(128, activation='relu'))
		# model.add(Dense(256, activation='relu'))
		# model.add(Dropout(0.25))
		model.add(Dense(10, activation='softmax'))    # 10 output classes, as probabilities
	
	model.summary()
	
	# (Sparse for integer encoded y_train, as opposed to one-hot encoded.)
	model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
	
	return model
	
def plot_training_result(training):
	# print(training.history.keys())
	
	loss = training.history['loss']
	val_loss = training.history['val_loss']
	ax = pd.DataFrame(loss).plot()
	ax = pd.DataFrame(val_loss).plot(ax=ax)
	ax.legend(['loss', 'val_loss'])
	plt.show()

def load_newest_model():
	'''Load latest saved model'''
	
	# Get the newest available model
	all_model_paths = glob.glob(model_save_dir+'*')
	newest_model_path = max(all_model_paths, key=os.path.getctime)

	# Load the latest model
	model = load_model(newest_model_path)
	print("\n" + "="*60 + "\n")
	print("Using model loaded from:", newest_model_path)
	print("\nLoaded model summary:")
	print(model.summary())
	
	return model

def save_trained_model(model, filename="saved_model"):
	'''Save an existing (trained) model'''
	
	t = datetime.now().strftime("%d_%m_%H%M%S")
	model_save_path = model_save_dir + (filename + "_%s.h5"%t)
	model.save(model_save_path)
	
def eval_model(model, x_test, y_test):
	print("\nEvaluating on test data...")
		
	loss = model.evaluate(x_test, y_test)
	
	for i in len(model.metrics_names):
		print(model.metrics_names[i]+":", loss[i])
	
def display_confusion_matrix(y_test, y_pred):
	print("\nEvaluating confusion matrix...")
	
	plt.clf()
	sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)
	plt.show()
	
def run_CNN(x_train, y_train, x_test, y_test):
	if USE_LOADED_MODEL:
		model = load_newest_model()
	else:
		# Convolution NN model
		model = create_compiled_model()

		# TODO: hyperparameter tuning (simply using the validation result currently)
		training = model.fit(x_train, y_train, batch_size=1000, epochs=20, validation_split=0.2)
		
		# Save the model so it can be loaded if desired (rather than having to re-train)
		save_trained_model(model)
		
		# Plot curves of training loss and validation loss
		plot_training_result(training)
	
	# Make predictions on test set
	y_pred = model.predict(x_test, verbose=1)
	
	# Get the class (i.e. index) with the highest probability
	y_pred = np.argmax(y_pred, axis=1)
	
	display_confusion_matrix(y_test, y_pred)
	
	eval_model(model, x_test, y_test)
	
def main():
	print("- Program running -")
	
	# Set up folder structure as needed
	setup()
	
	# Load dataset
	# x_train, y_train, x_test, y_test, label_names = load_data_batch()
	x_train, y_train, x_test, y_test, label_names = load_data()
	
	# Reshape the dataset into a shape that tensorflow expects: (samples, rows, cols, channels)
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
	
	run_CNN(x_train, y_train, x_test, y_test)
	
	print("- Program ended -")


if __name__ == '__main__':
	main()
