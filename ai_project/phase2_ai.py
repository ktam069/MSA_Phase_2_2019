import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential, load_model
from keras.layers import Input, Dense, Conv2D, LSTM, Bidirectional
from keras.layers import BatchNormalization, Activation, Flatten, TimeDistributed, Reshape
from keras.callbacks import ModelCheckpoint

import os
# import pandas as pd

CIFAR_10_dir = "./cifar-10-batches-py/"
CIFAR_10_path = CIFAR_10_dir + "data_batch_1"
dataset_save_dir = "./cifar-10-npy/"
dataset_save_path = dataset_save_dir + "cifar-10-batch-1.npy"


def setup():
	assert os.path.exists(CIFAR_10_dir) or os.path.exists(dataset_save_dir), "No input data files found"
	
	# Create any missing folders for loading and saving data
	if not os.path.exists(dataset_save_dir):
		os.mkdir(dataset_save_dir)

def unpickle(file):		# Function taken from: https://www.cs.toronto.edu/~kriz/cifar.html
	import pickle
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict

def load_dataset(save_path=dataset_save_path, CIFAR_10_path=CIFAR_10_path):
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
	
def main():
	# Set up folder structure as needed
	setup()
	
	# for i in range(1,6):
	for i in range(1,2):
		fpath = CIFAR_10_dir + "data_batch_%d"%i
		save_path = dataset_save_dir + "cifar-10-batch-%d.npy"%i
		x_train, y_train = load_dataset(save_path=save_path, CIFAR_10_path=fpath)
		
		print(type(x_train))
		print(x_train.shape)
		print(y_train.shape)


if __name__ == '__main__':
	main()
