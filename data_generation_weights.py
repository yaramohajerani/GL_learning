#!/usr/bin/env python
u"""
Yara Mohajerani (Last Update 04/2020)

data generator class for feeding data into keras model.
Modified from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
"""
import numpy as np
import rasterio
import keras

class DataGenerator(keras.utils.Sequence):
	'Generates data for Keras'
	def __init__(self, list_IDs, batch_size=10, dim=(512,512), 
			n_channels=2, shuffle=True, ratio=727):
		'Initialization'
		self.dim = dim
		self.batch_size = batch_size
		self.n_channels = n_channels
		self.list_IDs = list_IDs
		self.shuffle = shuffle
		self.ratio = ratio
		self.on_epoch_end()

	def __len__(self):
		'Denotes the number of batches per epoch'
		return int(np.floor(len(self.list_IDs) / self.batch_size))

	def __getitem__(self, index):
		'Generate one batch of data'
		# Generate indexes of the batch
		indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

		# Find list of IDs
		list_IDs_temp = [self.list_IDs[k] for k in indexes]

		# Generate data
		X, y, w = self.__data_generation(list_IDs_temp)

		return X, y, w

	def on_epoch_end(self):
		'Updates indexes after each epoch'
		self.indexes = np.arange(len(self.list_IDs))
		if self.shuffle == True:
			np.random.shuffle(self.indexes)

	def __data_generation(self, list_IDs_temp):
		'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
		# Initialization
		X = np.empty((self.batch_size, *self.dim, self.n_channels))
		y = np.empty((self.batch_size, self.dim[0]*self.dim[1], 1))
		w = np.empty((self.batch_size, self.dim[0]*self.dim[1]))
		# Generate data
		for i, ID in enumerate(list_IDs_temp):
			#-- read image
			X[i,] = np.load(ID)
			#-- read labels
			y[i,] = np.load(ID.replace('coco','delineation'))
			#-- get flattened weights
			w[i,] = np.squeeze(y[i,])*self.ratio

		return X,y,w