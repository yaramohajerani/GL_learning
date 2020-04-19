#!/usr/bin/env python
u"""
Yara Mohajerani 04/2020

Read geotiff files, preprocess, and convert to
npy arrays for keras data generator
"""
import os
import sys
import getopt
import numpy as np
import rasterio
from skimage.morphology import binary_dilation

#-- main function
def main():
	#-- Read the system arguments listed after the program
	long_options=['DIR=','N_TEST=','AUGMENT','DILATE']
	optlist,arglist = getopt.getopt(sys.argv[1:],'D:N:AD',long_options)

	#-- Set default settings
	ddir = os.path.join(os.path.expanduser('~'),'Google Drive File Stream',
		'Shared drives','GROUNDING_LINE_TEAM_DRIVE','ML_Yara','geocoded_v1')
	augment = False
	aug_str = ''
	n_img = 1 #default 1 image without augmentation
	dilate = False
	dilate_str = ''
	n_test = 50
	for opt, arg in optlist:
		if opt in ("-D","--DIR"):
			ddir = os.path.expanduser(arg)
			#-- remove the last dash if present
			if ddir.endswith('/'):
				ddir = ddir[:-1]
		elif opt in ("-A","--AUGMENT"):
			augment = True
			aug_str = '_aug'
			n_img = 4
		elif opt in ("-D","--dilate"):
			dilate = True
			dilate_str = '_dilated'
		elif opt in ("-N","--N_TEST"):
			n_test = int(arg)

	#-- Get list of files
	img_dir = os.path.join(ddir,'cocotile_withoutnull_v1')
	fileList = os.listdir(img_dir)
	img_list = [f for f in fileList if (f.endswith('.tif') and f.startswith('coco'))]
	
	#-- output directories
	out_train = os.path.join(ddir,'train{0}{1}_n{2}.dir'.format(aug_str,dilate_str,n_test))
	out_test = os.path.join(ddir,'test_n{0}.dir'.format(n_test))

	#-- make directories if they don't exist
	if not os.path.exists(out_train):
		os.mkdir(out_train)
	if not os.path.exists(out_test):
		os.mkdir(out_test)

	#-- read first file to get dimensions
	raster = rasterio.open(os.path.join(ddir,'cocotile_withoutnull_v1',img_list[0]))
	h = raster.height
	w = raster.width
	ch = raster.count
	if ch > 1:
		sys.exit('More than one channel in input dataset. Exiting.')
	#-- loop through files, read, and preprocess
	#-- save each file to output directory so we don't have to read them all at once
	n_train = len(img_list) - n_test
	for f in img_list[:n_train]:
		#-- IMAGE
		if (not os.path.exists(os.path.join(out_train,f.replace('.tif','.npy')))):
			img = np.ones((n_img,w,h,2))
			#-- read image
			raster = rasterio.open(os.path.join(ddir,'cocotile_withoutnull_v1',f))
			img[0,:,:,0] = raster.read(1).real
			img[0,:,:,1] = raster.read(1).imag
			if augment:
				for c in range(2):
					img[1,:,:,c] = np.fliplr(img[0,:,:,c])
					img[2,:,:,c] = np.flipud(img[0,:,:,c])
					img[3,:,:,c] = np.fliplr(np.flipud(img[0,:,:,c]))
			#-- save arrays to file
			for i in range(n_img):
				if i == 0:
					suffix = ''
				else:
					suffix = '_aug%i'%i
				#-- save image array
				np.save(os.path.join(out_train,f.replace('.tif','%s.npy'%suffix)),img[i])
		#-- TRAINING LABELS
		if (not os.path.exists(os.path.join(out_train,\
				f.replace('coco','delineation').replace('.tif','.npy')))):
			lbl = np.ones((n_img,w,h,1),dtype=np.int8)
			#-- read label
			raster = rasterio.open(os.path.join(ddir,'delineationtile_withoutnull_v1',
				f.replace('coco','delineation')))
			if dilate:
				lbl[0,:,:,0] = binary_dilation(np.int8(raster.read(1)/255.))
				if augment:
					lbl[1,:,:,0] = binary_dilation(np.fliplr(lbl[0,:,:,0]))
					lbl[2,:,:,0] = binary_dilation(np.flipud(lbl[0,:,:,0]))
					lbl[3,:,:,0] = binary_dilation(np.fliplr(np.flipud(lbl[0,:,:,0])))
			else:
				lbl[0,:,:,0] = np.int8(raster.read(1)/255.)
				if augment:
					lbl[1,:,:,0] = np.fliplr(lbl[0,:,:,0])
					lbl[2,:,:,0] = np.flipud(lbl[0,:,:,0])
					lbl[3,:,:,0] = np.fliplr(np.flipud(lbl[0,:,:,0]))
			#-- save arrays to file
			for i in range(n_img):
				if i == 0:
					suffix = ''
				else:
					suffix = '_aug%i'%i
				#-- save label array
				np.save(os.path.join(out_train,\
					f.replace('coco','delineation').replace('.tif','%s.npy'%suffix)),lbl[i])

	#-- read, process, and write TEST data
	for f in img_list[n_train:]:
		#-- initialize
		img = np.ones((w,h,2))
		lbl = np.ones((w,h,1),dtype=np.int8)
		#-- read image
		raster = rasterio.open(os.path.join(ddir,'cocotile_withoutnull_v1',f))
		img[:,:,0] = raster.read(1).real
		img[:,:,1] = raster.read(1).imag
	
		#-- read label
		raster = rasterio.open(os.path.join(ddir,'delineationtile_withoutnull_v1',
			f.replace('coco','delineation')))
		lbl[:,:,0] = np.int8(raster.read(1))
			
		#-- save image array
		np.save(os.path.join(out_test,f.replace('.tif','.npy')),img)
		#-- save label array
		np.save(os.path.join(out_test,\
			f.replace('coco','delineation').replace('.tif','.npy')),lbl)

#-- run main program
if __name__ == '__main__':
	main()
