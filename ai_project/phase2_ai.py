# MSA Phase 2 Project - AI CIFAR-10 Program
# Author: King Hang Tam

import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# from keras.models import Sequential, load_model
# from keras.layers import Input, Dense, Conv2D, LSTM, Bidirectional
# from keras.layers import BatchNormalization, Activation, Flatten, TimeDistributed, Reshape
# from keras.callbacks import ModelCheckpoint

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
	
def main():
	print("- Program running -")
	
	# Set up folder structure as needed
	setup()
	
	# Load dataset
	x_train, y_train, x_test, y_test, label_names = load_data()
	
	print("- Program ended -")


if __name__ == '__main__':
	main()
