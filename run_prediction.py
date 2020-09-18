#!/usr/bin/env python
u"""
run_prediction.py
by Yara Mohajerani (07/2020)

Run already trained network on specifed data.
"""
#-- Import Modules
import os
import sys
import imp
import getopt
import numpy as np
import rasterio
from osgeo import gdal,osr
import keras
import timeit
from keras import backend as K
from keras.preprocessing import image
from tensorflow.python.client import device_lib

#-- main function
def main():
	# print(K.tensorflow_backend._get_available_gpus())
	print(device_lib.list_local_devices())
	#-- Read the system arguments listed after the program
	long_options=['DIR=','DOWN=','INIT=','DROPOUT=','RATIO=','MOD=','NUM=','START=','MODEL_DIR=','RUN_ALL']
	optlist,arglist = getopt.getopt(sys.argv[1:],'D:W:I:O:R:M:N:S:L:A',long_options)

	#-- Set default settings
	ddir = os.path.join(os.path.expanduser('~'),'Google Drive File Stream',\
		'Shared drives','GROUNDING_LINE_TEAM_DRIVE','ML_Yara','S1_Pope-Smith-Kohler',\
		'UNUSED','coco_PSK-UNUSED_with_null')
	model_dir = os.path.join(os.path.expanduser('~'),'GL_learning')
	ndown = 4 # number of 'down' steps
	ninit = 32 #number of channels to start with
	dropout_frac = 0.2 # dropout fraction
	ratio = 727 # penalization ratio for GL and non-GL points based on smaller dataaset
	mod_lbl = 'atrous'
	num = 500
	cc = 0
	run_all = False
	for opt, arg in optlist:
		if opt in ("-D","--DIR"):
			ddir = os.path.expanduser(arg)
		elif opt in ("-L","--MODEL_DIR"):
			model_dir = os.path.expanduser(arg)
		elif opt in ("-W","--DOWN"):
			ndown = int(arg)
		elif opt in ("-I","--INIT"):
			ninit = int(arg)
		elif opt in ("-O","--DROPOUT"):
			dropout_frac = float(arg)
		elif opt in ("-R","--RATIO"):
			ratio = float(arg)
		elif opt in ("-M","--MOD"):
			mod_lbl = arg
		elif opt in ("-N","--NUM"):
			num = int(arg)
		elif opt in ("-S","--START"):
			cc = int(arg)
		elif opt in ("-A","--RUN_ALL"):
			run_all = True
			cc = 0

	#-- set up model name
	if mod_lbl == 'unet':
		mod_str = '{0}_{1}init_{2}down_drop{3:.1f}_customLossR{4}'.\
			format(mod_lbl,ninit,ndown,dropout_frac,ratio)
	elif mod_lbl == 'atrous':
		mod_str = '{0}_{1}init_drop{2:.1f}_customLossR{3}'.\
			format(mod_lbl,ninit,dropout_frac,ratio)
	else:
		print(mod_str)
		sys.exit('model label not matching.')

	#-- Get list of images
	fileList = os.listdir(ddir)
	# file_list = sorted([f for f in fileList if ( (f.endswith('DIR00.tif') or f.endswith('DIR11.tif')) and f.startswith('coco') )])
	file_list = sorted([f for f in fileList if (f.endswith('.tif') and f.startswith('coco'))])
	N = len(file_list)
	print(N)

	#-- read first file to get dimensions
	raster = rasterio.open(os.path.join(ddir,file_list[0]))
	h = raster.height
	wi = raster.width
	ch = raster.count

	print(h,wi,ch)

	#-- set channel to 2 because there are actually real and imaginary components
	ch = 2

	#-- Import model
	mod_module = imp.load_source('nn_model',os.path.join(model_dir,'nn_model.py'))
	#-- set up model
	if mod_lbl == 'unet':
		print('loading unet model')
		model = mod_module.unet_model_double_dropout(height=h,width=wi,\
			channels=ch,n_init=ninit,n_layers=ndown,drop=dropout_frac)
	elif mod_lbl == 'atrous':
		print("loading atrous model")
		model = mod_module.nn_model_atrous_double_dropout(height=h,\
			width=wi,channels=ch,n_filts=ninit,drop=dropout_frac)
	else:
		print('Model label not correct.')

	#-- define custom loss function
	def customLoss(yTrue,yPred):
		return -1*K.mean(ratio*(yTrue*K.log(yPred+1e-32)) + ((1. - yTrue)*K.log(1-yPred+1e-32)))

	#-- compile imported model
	model.compile(loss=customLoss,optimizer='adam',
				metrics=['accuracy'])

	#-- checkpoint file
	chk_file = os.path.join(model_dir,'{0}_weights.h5'.format(mod_str))
	print(chk_file)
	#-- if file exists, read model from file
	if os.path.isfile(chk_file):
		print('Check point exists; loading model from file.')
		#-- load weights
		model.load_weights(chk_file)
	else:
		sys.exit('Model does not previously exist.')

	#-------------------------------
	#-- Run model on data
	#-------------------------------
	start_time = timeit.default_timer()
	#-- make output directory
	out_dir = os.path.join(ddir,'{0}.dir'.format(mod_str))
	if (not os.path.isdir(out_dir)):
		os.mkdir(out_dir)
	#-- if not running all, set max number to 'num'. otherwise N is total
	#-- number of files
	if not run_all:
		N = num + cc
	print('Running total: ', num)
	print('start: ', cc)
	print('N: ', N)
	while (cc < N):
		#-- read "num" files at a time
		print(cc)
		#-- Read data all at once
		imgs = np.ones((num,h,wi,ch))
		trans = [None]*num
		for i,f in enumerate(file_list[cc:cc+num]):
			#-- read image
			raster = rasterio.open(os.path.join(ddir,f))
			try:
				imgs[i,:,:,0] = raster.read(1).real
				imgs[i,:,:,1] = raster.read(1).imag
			except:
				print('Skipping %s'%f)
				imgs[i,:,:,0] = None
				imgs[i,:,:,1] = None
			#-- get transformation matrix
			trans[i] = raster.transform
			raster.close()

		out_crs = raster.crs.to_epsg()
		out_imgs = model.predict(imgs, batch_size=1, verbose=1)
		out_imgs = out_imgs.reshape(out_imgs.shape[0],h,wi,out_imgs.shape[2])
		
		#-- save output images
		for i,f in enumerate(file_list[cc:cc+num]):
			#-- get pixel size
			x_orig,y_orig = rasterio.transform.xy(trans[i], 0, 0)
			x2,y2 = rasterio.transform.xy(trans[i], 0, 1)
			x3,y3 = rasterio.transform.xy(trans[i], 1, 0)
			dx = np.abs(x2 - x_orig)
			dy = np.abs(y3 - y_orig)

			#-- get transformation for output
			#-- output as geotiff
			driver = gdal.GetDriverByName("GTiff")
			#-- set up the dataset with compression options (1 is for band 1)
			OPTS = ['COMPRESS=LZW'] #['COMPRESS=NONE'] #['COMPRESS=PACKBITS']
			ds = driver.Create(os.path.join(out_dir,os.path.basename(f).replace('coco','pred')),\
				h, wi, 1, gdal.GDT_Float32, OPTS)
			#-- top left x, w-e pixel resolution, rotation
			#-- top left y, rotation, n-s pixel resolution
			ds.SetGeoTransform([x_orig, dx, 0, y_orig, 0, -dy])
			#-- set the reference info
			srs = osr.SpatialReference()
			srs.ImportFromEPSG(out_crs)
			#-- export
			ds.SetProjection( srs.ExportToWkt() )
			#-- write to geotiff array
			ds.GetRasterBand(1).WriteArray(out_imgs[i].reshape(h,wi))
			ds.FlushCache()
			ds = None
			#-- also save as image
			# im = image.array_to_img(out_imgs[i]) 
			# im.save(os.path.join(out_dir,os.path.basename(f).replace('coco','pred').replace('tif','png')))
		#-- increment counter
		cc += num
	#-- print total time
	end_time = timeit.default_timer()
	print('Time Elapsed: ', end_time - start_time)  
#-- run main program
if __name__ == '__main__':
	main()
