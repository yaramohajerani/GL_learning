#!/usr/bin/env python
u"""
test_model.py
Yara Mohajerani (Last update 07/2020)

Write History
	06/2020	add user inputs
			save output as npy files
	05/2020 Written
"""
#-- Import Modules
import os
import sys
import getopt
import imp
import numpy as np
import matplotlib.pyplot as plt 
import keras
from keras import backend as K
from keras.preprocessing import image

ninit = 16 #number of channels to start with
dropout_frac = 0.2 # dropout fraction

#-- main function
def main():
	#-- Read the system arguments listed after the program
	long_options=['MOD=','INIT=','DROPOUT=','NTEST=','RATIO=','FORMAT=']
	optlist,arglist = getopt.getopt(sys.argv[1:],'M:I:D:N:R:F:',long_options)

	#-- Set default settings
	mod_lbl = 'atrous'
	ninit = 32 #number of channels to start with
	dropout_frac = 0.2 # dropout fraction
	n_test = 500
	ratio = 727 # penalization ratio for GL and non-GL points based on smaller dataaset
	out_form = 'npy'
	for opt, arg in optlist:
		if opt in ("-M","--MOD"):
			mod_lbl = arg
		elif opt in ("-I","--INIT"):
			ninit = int(arg)
		elif opt in ("-D","--DROPOUT"):
			dropout_frac = int(arg)
		elif opt in ("-N","--NTEST"):
			n_test = int(arg)
		elif opt in ("-R","--RATIO"):
			ratio = int(arg)
		elif opt in ("-F","--FORMAT"):
			out_form = arg

	#-- make model string
	if mod_lbl == 'unet':
		mod_str = '{0}_{1}init_{2}down_drop{3:.1f}_customLossR{4}'.format(mod_lbl,ninit,ndown,
															dropout_frac,ratio)
	elif mod_lbl in ['atrous','atrous_noPool']:
		mod_str = '{0}_{1}init_drop{2:.1f}_customLossR{3}'.format(mod_lbl,ninit,dropout_frac,ratio)
	else:
		print('model label not matching.')
	print(mod_str)

	#-- Directory setup
	gdrive = os.path.expanduser('~/Google Drive File Stream')
	colabdir = os.path.join(gdrive,'My Drive','Colab Notebooks')
	mod_dir = os.path.join(gdrive,'My Drive','GL_Learning')
	output_dir = os.path.join(os.path.expanduser('~'),'GL_learning_data','geocoded_v1')
	ddir = os.path.join(gdrive,'Shared drives','GROUNDING_LINE_TEAM_DRIVE',\
		'ML_Yara','geocoded_v1')
	subdir = {}
	subdir['Train'] = os.path.join(ddir,'train_n%i.dir'%n_test)
	subdir['Test'] = os.path.join(ddir,'test_n%i.dir'%n_test)

	#-- Get list of images
	file_list = {}
	fileList = os.listdir(subdir['Train'])
	file_list['Train'] = [f for f in fileList if (f.endswith('.npy') and f.startswith('coco'))]
	fileList = os.listdir(subdir['Test'])
	file_list['Test'] = [f for f in fileList if (f.endswith('.npy') and f.startswith('coco'))]

	#-- get full path of files
	ID_list = {}
	N = {}
	for t in ['Train','Test']:
		ID_list[t] = [os.path.join(subdir[t],f) for f in file_list[t]]
		N[t] = len(ID_list[t])

	#-- read 1 file to get dimensions
	im = np.load(ID_list['Train'][0])
	h,wi,ch = im.shape
	print(h,wi,ch)

	#-- Import model
	mod_module = imp.load_source('nn_model',os.path.join(colabdir,'nn_model.py'))
	#-- set up model
	if mod_lbl == 'unet':
		print('loading unet model')
		model = mod_module.unet_model_double_dropout(height=h,width=wi,channels=ch, 
													n_init=ninit,n_layers=ndown,
													drop=dropout_frac)
	elif mod_lbl == 'atrous':
		print("loading atrous model")
		model = mod_module.nn_model_atrous_double_dropout(height=h,width=wi,
															channels=ch,
															n_filts=ninit,
															drop=dropout_frac)
	elif mod_lbl == 'atrous_noPool':
		print("loading atrous_noPool model")
		model = mod_module.nn_model_atrous_noPool(height=h,width=wi,
													channels=ch,
													n_filts=ninit,
													drop=dropout_frac)
	else:
		sys.exit('Model label not correct.')

	#-- define custom loss function
	def customLoss(yTrue,yPred):
		return -1*K.mean(ratio*(yTrue*K.log(yPred+1e-32)) + ((1. - yTrue)*K.log(1-yPred+1e-32)))
	
	#-- compile imported model
	model.compile(loss=customLoss,optimizer='adam',
					metrics=['accuracy'])

	#-- checkpoint file
	chk_file = os.path.join(mod_dir,'{0}_weights.h5'.format(mod_str))
	print(chk_file)
	#-- if file exists, read model from file
	if os.path.isfile(chk_file):
		print('Check point exists; loading model from file.')
		#-- load weights
		model.load_weights(chk_file)
	else:
		sys.exit('Model does not exist.')

	#-------------------------------
	#-- Run on train and test data
	#-------------------------------
	for t in ['Train','Test']:
		print(t)
		#-- make output directory
		out_dir = os.path.join(output_dir,'{0}_predictions.dir'.format(t),\
			'{0}.dir'.format(mod_str))
		if (not os.path.isdir(out_dir)):
			os.mkdir(out_dir)
		#-- read 500 files at a time (memory bottleneck)
		cc = 0
		while (cc < N[t]):
			print(cc)
			#-- Read data all at once
			test_imgs = np.ones((500,h,wi,ch))
			for i,f in enumerate(ID_list[t][cc:cc+500]):
				test_imgs[i,] = np.load(f)
			out_imgs = model.predict(test_imgs, batch_size=1, verbose=1)
			out_imgs = out_imgs.reshape(out_imgs.shape[0],h,wi,out_imgs.shape[2])
			#-- save output images
			for i,f in enumerate(ID_list[t][cc:cc+500]):
				if out_form == 'png':
					im = image.array_to_img(out_imgs[i]) 
					im.save(os.path.join(out_dir,os.path.basename(f).replace('coco','pred').replace('npy','png')))
				elif out_form == 'npy':
					np.save(os.path.join(out_dir,os.path.basename(f).replace('coco','pred')),out_imgs[i])
				else:
					sys.exit('Output format not recognized.')
			#-- increment counter
			cc += 500

#-- run main program
if __name__ == '__main__':
	main()