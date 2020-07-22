#!/usr/bin/env python
u"""
plot_activation_maps.py
Yara Mohajerani (07/2020)
"""

import os
import imp
import sys
import numpy as np
import matplotlib.pyplot as plt 
import keras
from keras import backend as K
from keras.preprocessing import image
from keras.utils import plot_model

#-- Set up configurations / parameters
retrain = True # retrain previously existing model
ndown = 4 # number of 'down' steps
ninit = 32 #number of channels to start with
dropout_frac = 0.2 # dropout fraction
ratio = 727 # penalization ratio for GL and non-GL points based on smaller dataaset
mod_lbl = 'atrous' #'unet'
if mod_lbl == 'unet':
  mod_str = '{0}_{1}init_{2}down_drop{3:.1f}_customLossR{4}'.format(mod_lbl,ninit,ndown,
                                                        dropout_frac,ratio)
elif mod_lbl == 'atrous':
  mod_str = '{0}_{1}init_drop{2:.1f}_customLossR{3}'.format(mod_lbl,ninit,dropout_frac,ratio)
else:
  print('model label not matching.')
print(mod_str)

#-- Directory setup
gdrive =  os.path.expanduser('~/Google Drive File Stream')
colabdir = os.path.join(gdrive,'My Drive','Colab Notebooks')
ddir = os.path.join(gdrive,'Shared drives','GROUNDING_LINE_TEAM_DRIVE','ML_Yara','geocoded_v1')
test_dir = os.path.join(ddir,'test_n%i.dir'%n_test)
output_dir = os.path.expanduser('~/GL_learning_data/')

#-- Get list of images
fileList = os.listdir(test_dir)

file_ind = fileList.index('coco_gl_069_181124-181130-181206-181212_013745-024816-013920-024991_T110456_T110456_x1536_y0512_DIR11.npy')
im = np.load(os.path.join(test_dir,test_list[file_ind]))
h,wi,ch = im.shape

#-- Import model
mod_module = imp.load_source('unet_model',os.path.join(colabdir,'unet_model.py'))
#-- set up model
if mod_lbl == 'unet':
  print('loading unet model')
  model = mod_module.unet_model_double_dropout(height=h,width=wi,channels=ch, 
                                        n_init=ninit,n_layers=ndown,
                                        drop=dropout_frac)
elif mod_lbl == 'atrous':
  print("loading atrous model")
  model = mod_module.unet_model_atrous_double_dropout(height=h,width=wi,
                                                channels=ch,
                                                n_filts=ninit,
                                                drop=dropout_frac)
else:
  print('Model label not correct.')


#-- define custom loss function
def customLoss(yTrue,yPred):
  return -1*K.mean(ratio*(yTrue*K.log(yPred+1e-32)) + ((1. - yTrue)*K.log(1-yPred+1e-32)))


#-- compile imported model
model.compile(loss=customLoss,optimizer='adam',
              metrics=['accuracy'])


#-- checkpoint file
chk_file = os.path.join(output_dir,'{0}_weights.h5'.format(mod_str))

#-- if file exists, read model from file
if os.path.isfile(chk_file):
  print('Check point exists; loading model from file.')
  #-- load weights
  model.load_weights(chk_file)
else:
  sys.exit('Model does not previously exist.')

layer_outputs = [layer.output for layer in model.layers[:22]] 
#-- extract output of first 22 layers
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(img_tensor) 


plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')
