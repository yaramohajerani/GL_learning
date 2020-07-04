#!/anaconda2/bin/python2.7
u"""
nn_model.py
by Yara Mohajerani (Last Update 03/2020)

Construct a dynamic u-net model with a variable
number of layers for glacier calving front detection.

Update History
	03/2020	Add atrous U-net-like model
    01/2019 Fix batch normalization axis input
    09/2018 Add multiple functions to test different versions
            Don't compile (compile in main script to allow for 
            different weighting experiments)
            Add multiple functions with different architectures
            Add new option for batch normalization instead of dropout
    04/2018 Written
"""
from keras import backend as K
import keras.layers as kl
import keras.models as km
import copy
import sys
import keras
import tensorflow as tf
from keras.layers.normalization import BatchNormalization
import os
import imp

#-----------------------------------------------------------------------------------
#-- model with no pooling or upsampling
#-----------------------------------------------------------------------------------
def nn_model_atrous_noPool(height=0,width=0,channels=1,n_filts=32,drop=0):
	#-- inner function for convolutional units
	def conv_unit(x,nn):
		#-- convolution layer
		c = kl.Conv2D(nn,3,activation='elu',padding='same')(x)
		if drop != 0:
			c = kl.Dropout(drop)(c)
		c = kl.Conv2D(nn,3,activation='elu',padding='same')(c)
		return(c)

	#-- define input
	inputs = kl.Input((height,width,channels))

	#-- call depthwise separable conv unit
	c1 = conv_unit(inputs,n_filts)
	
	#-- convolutional block
	c2 = conv_unit(c1,n_filts*2)

	#-- perform 3 parrallel atrous convolutions
	a = {}
	for i in [1,3,5]:
		a[i] = kl.Conv2D(n_filts*2,3,activation='elu',\
			dilation_rate=i,padding='same')(c2)
	
	#-- concatanate dilated convs
	c3 = kl.Concatenate(axis=3)([a[i] for i in a.keys()])
	
	#-- convolution
	c4 = kl.Conv2D(n_filts*2,3,activation='elu',padding='same')(c3)

	#-- concatenate with c2
	c5 = kl.Concatenate(axis=3)([c4,c2])

	#-- convlutional block
	c6 = conv_unit(c5,n_filts)

	#-- do one final sigmoid convolution into just 1 final channel (None,h,w,1)
	c7 = kl.Conv2D(1,1,activation='sigmoid')(c6)
	#-- reshape into a flattened output to match sample weights
	c8 = kl.Reshape((height*width,1,))(c7)

	#-- make model
	model = km.Model(inputs=inputs,outputs=c8)

	#-- return model
	return model


#-----------------------------------------------------------------------------------
#-- double the size of each convolution layer 
#-----------------------------------------------------------------------------------
def nn_model_atrous_double_dropout(height=0,width=0,channels=1,n_filts=32,drop=0):
	#-- inner function for convolutional units
	def conv_unit(x,nn):
		#-- convolution layer
		c = kl.Conv2D(nn,3,activation='elu',padding='same')(x)
		if drop != 0:
			c = kl.Dropout(drop)(c)
		c = kl.Conv2D(nn,3,activation='elu',padding='same')(c)
		return(c)

    #-- define input
	inputs = kl.Input((height,width,channels))
	#-- call convolutional block
	c1 = conv_unit(inputs,n_filts) #(h,w)
	#-- 2x2 pooling
	p1 = kl.MaxPooling2D(pool_size=(2,2))(c1) #(h/2,w/2)
	#-- second convolutional block
	c2 = conv_unit(p1,n_filts*2) #(h/2,w/2)
	#-- 2x2 pooling
	p2 = kl.MaxPooling2D(pool_size=(2,2))(c2) #(h/4,w/4)
	#-- third convolutional block
	c3 = conv_unit(p2,n_filts*2) #(h/4,w/4)
	#-- 2x2 pooling
	p3 = kl.MaxPooling2D(pool_size=(2,2))(c3) #(h/8,w/8)
	#-- fourth convolutional block
	c4 = conv_unit(p3,n_filts*4) #(h/8,w/8)
	#-- now perform parallel atrous convolutions
	a = {}
	for i in [1,2,3,4,5]:
		a[i] = kl.SeparableConv2D(n_filts*4,3,activation='elu',\
			dilation_rate=(i,i), depth_multiplier=1,padding='same')(c4)
	#-- concatanate dilated convs
	c5 = kl.Concatenate(axis=3)([a[i] for i in a.keys()]) #(h/8,w/8)

	#-- upsample (h/4,w/4)
	c6 = kl.UpSampling2D(size=(2,2))(c5)
	c7 = kl.Concatenate(axis=3)([c6,c3])
	#-- convolutional block
	c8 = conv_unit(c7,n_filts*2)

	#-- upsample (h/2,w/2)
	c9 = kl.UpSampling2D(size=(2,2))(c8)
	c10 = kl.Concatenate(axis=3)([c9,c2])
	#-- convolutional block
	c11 = conv_unit(c10,n_filts)

	#-- upsample (h,w)
	c12 = kl.UpSampling2D(size=(2,2))(c11)
	c13 = kl.Concatenate(axis=3)([c12,c1])
	#-- convolutional block
	c14 = conv_unit(c13,n_filts)

	#-- convlution across the last 'n_filts' filters into 3 channels
	c15 = kl.Conv2D(3,3,activation='elu',padding='same')(c14)
	#-- do one final sigmoid convolution into just 1 final channel (None,h,w,1)
	c16 = kl.Conv2D(1,1,activation='sigmoid')(c15)
	#-- reshape into a flattened output to match sample weights
	c17 = kl.Reshape((height*width,1,))(c16)

	#-- make model
	model = km.Model(inputs=inputs,outputs=c17)

	#-- return model
	return model

#---------------------------------------------------------------------------------------
#-- linearly scale the size of each convolution layer (i.e. initial*i for the ith layer)
#---------------------------------------------------------------------------------------
def unet_model_linear_dropout(height=0,width=0,channels=1,n_init=12,n_layers=2,drop=0):

    #-- define input
    inputs = kl.Input((height,width,channels))

    c = {}
    p = {}
    count = 0
    #-- define input
    p[0] = inputs
    for i in range(1,n_layers+1):
        #-- convolution layer
        c[i] = kl.Conv2D(n_init*i,3,activation='relu',padding='same')(p[i-1])
        if drop != 0:
            c[i] = kl.Dropout(drop)(c[i])
        c[i] = kl.Conv2D(n_init*i,3,activation='relu',padding='same')(c[i])

        #-- pool, 2x2 blockcs
        #-- don't do pooling for the last down layer
        if i != n_layers:
            p[i] = kl.MaxPooling2D(pool_size=(2,2))(c[i])
        count += 1

    #---------------------------------------------
    #-- now go back up to reconsturct the image
    #---------------------------------------------
    upsampled_c = {}
    up = {}
    print('Max Number of Convlution Filters: ',n_init*i)
    while count>1:
        #-- concatenate the 1st convolution layer with an upsampled 2nd layer
        #-- where the missing elements in the 2nd layer are padded with 0
        #-- concatenating along the color channels
        upsampled_c[i] = kl.UpSampling2D(size=(2,2))(c[i])
        up[i] = kl.concatenate([upsampled_c[i],c[count-1]],axis=3)
        #-- now do a convolution with the merged upsampled layer
        i += 1
        c[i] = kl.Conv2D(n_init*(count-1),3,activation='relu',padding='same')(up[i-1])
        if drop != 0:
            c[i] = kl.Dropout(drop)(c[i])
        c[i] = kl.Conv2D(n_init*(count-1),3,activation='relu',padding='same')(c[i])
        #-- counter decreases as we go back up
        count -= 1

    print('Number of Convlution Filters at the end of up segment: ',n_init*count)
    #-- convlution across the last n_init filters into 3 channels
    i += 1
    c[i] = kl.Conv2D(3,3,activation='relu',padding='same')(c[i-1])
    #-- do one final sigmoid convolution into just 1 final channel (None,h,w,1)
    i += 1
    c[i] = kl.Conv2D(1,1,activation='sigmoid')(c[i-1])
    #-- reshape into a flattened output to match sample weights
    i += 1
    c[i] = kl.Reshape((height*width,1,))(c[i-1])

    print('output shape: ', c[i].shape)
    print('Total Number of layers: ',i)

    #-- make model
    model = km.Model(inputs=inputs,outputs=c[i])

    #-- return model
    return model



#-----------------------------------------------------------------------------------
#-- double the size of each convolution layer 
#-----------------------------------------------------------------------------------
def unet_model_double_dropout(height=0,width=0,channels=1,n_init=12,n_layers=2,drop=0):

    #-- define input
    inputs = kl.Input((height,width,channels))

    c = {}
    p = {}
    count = 0
    #-- define input
    p[0] = inputs
    n_filts = copy.copy(n_init)
    for i in range(1,n_layers+1):
        #-- convolution layer
        c[i] = kl.Conv2D(n_filts,3,activation='relu',padding='same')(p[i-1])
        if drop != 0:
            c[i] = kl.Dropout(drop)(c[i])
        c[i] = kl.Conv2D(n_filts,3,activation='relu',padding='same')(c[i])

        #-- pool, 2x2 blockcs
        #-- don't do pooling for the last down layer
        #-- also don't double the filter numbers
        if i != n_layers:
            p[i] = kl.MaxPooling2D(pool_size=(2,2))(c[i])
            n_filts *= 2
        count += 1

    #---------------------------------------------
    #-- now go back up to reconsturct the image
    #---------------------------------------------
    upsampled_c = {}
    up = {}
    print('Max Number of Convlution Filters: ',n_filts)
    while count>1:
        n_filts = int(n_filts/2)
        #-- concatenate the 1st convolution layer with an upsampled 2nd layer
        #-- where the missing elements in the 2nd layer are padded with 0
        #-- concatenating along the color channels
        upsampled_c[i] = kl.UpSampling2D(size=(2,2))(c[i])

        # upsampled_c[i] = BilinearUpsampling.BilinearUpsampling(upsampling=(2,2))(c[count])
        up[i] = kl.Concatenate(axis=3)([upsampled_c[i],c[count-1]])

        #-- now do a convlution with the merged upsampled layer
        i += 1
        c[i] = kl.Conv2D(n_filts,3,activation='relu',padding='same')(up[i-1])
        if drop != 0:
            c[i] = kl.Dropout(drop)(c[i])

        c[i] = kl.Conv2D(n_filts,3,activation='relu',padding='same')(c[i])
        #-- counter decreases as we go back up
        count -= 1

    print('Number of Convlution Filters at the end of up segment: ',n_filts)
	#-- convlution across the last n_init filters into 3 channels
    i += 1
    c[i] = kl.Conv2D(3,3,activation='relu',padding='same')(c[i-1])
    #-- do one final sigmoid convolution into just 1 final channel (None,h,w,1)
    i += 1
    c[i] = kl.Conv2D(1,1,activation='sigmoid')(c[i-1])
    #-- reshape into a flattened output to match sample weights
    i += 1
    c[i] = kl.Reshape((height*width,1,))(c[i-1])

    print('output shape: ', c[i].shape)
    print('Total Number of layers: ',i)


    #-- make model
    model = km.Model(inputs=inputs,outputs=c[i])

    #-- return model
    return model




#-----------------------------------------------------------------------------------
#-- batch normalization instread of dropout for "linear" architecture
#-----------------------------------------------------------------------------------
def unet_model_linear_normalized(height=0,width=0,channels=1,n_init=12,n_layers=2):

    #-- define input
    inputs = kl.Input((height,width,channels))

    c = {}
    p = {}
    count = 0
    #-- define input
    p[0] = inputs
    for i in range(1,n_layers+1):
        #-- convolution layer
        c[i] = BatchNormalization(axis=-1)(kl.Conv2D(n_init*i,3,activation='relu',padding='same')(p[i-1]))
        c[i] = BatchNormalization(axis=-1)(kl.Conv2D(n_init*i,3,activation='relu',padding='same')(c[i]))

        #-- pool, 2x2 blockcs
        #-- don't do pooling for the last down layer
        if i != n_layers:
            p[i] = kl.MaxPooling2D(pool_size=(2,2))(c[i])
        count += 1

    #---------------------------------------------
    #-- now go back up to reconsturct the image
    #---------------------------------------------
    upsampled_c = {}
    up = {}
    print('Max Number of Convlution Filters: ',n_init*i)
    while count>1:
        #-- concatenate the 1st convolution layer with an upsampled 2nd layer
        #-- where the missing elements in the 2nd layer are padded with 0
        #-- concatenating along the color channels
        upsampled_c[i] = kl.UpSampling2D(size=(2,2))(c[i])
        up[i] = kl.concatenate([upsampled_c[i],c[count-1]],axis=3)
        #-- now do a convolution with the merged upsampled layer
        i += 1
        c[i] = BatchNormalization(axis=-1)(kl.Conv2D(n_init*(count-1),3,activation='relu',padding='same')(up[i-1]))
        c[i] = BatchNormalization(axis=-1)(kl.Conv2D(n_init*(count-1),3,activation='relu',padding='same')(c[i]))
        #-- counter decreases as we go back up
        count -= 1

    print('Number of Convlution Filters at the end of up segment: ',n_init*count)
    #-- convlution across the last n_init filters into 3 channels
    i += 1
    c[i] = BatchNormalization(axis=-1)(kl.Conv2D(3,3,activation='relu',padding='same')(c[i-1]))
    #-- do one final sigmoid convolution into just 1 final channel (None,h,w,1)
    i += 1
    c[i] = BatchNormalization(axis=-1)(kl.Conv2D(1,1,activation='sigmoid')(c[i-1]))
    #-- reshape into a flattened output to match sample weights
    i += 1
    c[i] = kl.Reshape((height*width,1,))(c[i-1])

    print('output shape: ', c[i].shape)
    print('Total Number of layers: ',i)

    #-- make model
    model = km.Model(inputs=inputs,outputs=c[i])

    #-- return model
    return model




#-----------------------------------------------------------------------------------
#-- batch normalization instread of dropout for "double" architecture
#-----------------------------------------------------------------------------------
def unet_model_double_normalized(height=0,width=0,channels=1,n_init=12,n_layers=2):
    #-- define input
    inputs = kl.Input((height,width,channels))

    c = {}
    p = {}
    count = 0
    #-- define input
    p[0] = inputs
    n_filts = copy.copy(n_init)
    for i in range(1,n_layers+1):
        #-- convolution layer
        c[i] = BatchNormalization(axis=-1)(kl.Conv2D(n_filts,3,activation='relu',padding='same')(p[i-1]))
        c[i] = BatchNormalization(axis=-1)(kl.Conv2D(n_filts,3,activation='relu',padding='same')(c[i]))

        #-- pool, 2x2 blockcs
        #-- don't do pooling for the last down layer
        #-- also don't double the filter numbers
        if i != n_layers:
            p[i] = kl.MaxPooling2D(pool_size=(2,2))(c[i])
            n_filts *= 2
        count += 1

    #---------------------------------------------
    #-- now go back up to reconsturct the image
    #---------------------------------------------
    upsampled_c = {}
    up = {}
    print('Max Number of Convlution Filters: ',n_filts)
    while count>1:
        n_filts = int(n_filts/2)
        #-- concatenate the 1st convolution layer with an upsampled 2nd layer
        #-- where the missing elements in the 2nd layer are padded with 0
        #-- concatenating along the color channels
        upsampled_c[i] = kl.UpSampling2D(size=(2,2))(c[i])
        up[i] = kl.concatenate([upsampled_c[i],c[count-1]],axis=3)
        #-- now do a convlution with the merged upsampled layer
        i += 1
        c[i] = BatchNormalization(axis=-1)(kl.Conv2D(n_filts,3,activation='relu',padding='same')(up[i-1]))
        c[i] = BatchNormalization(axis=-1)(kl.Conv2D(n_filts,3,activation='relu',padding='same')(c[i]))
        #-- counter decreases as we go back up
        count -= 1

    print('Number of Convlution Filters at the end of up segment: ',n_filts)
     #-- convlution across the last n_init filters into 3 channels
    i += 1
    c[i] = BatchNormalization(axis=-1)(kl.Conv2D(3,3,activation='relu',padding='same')(c[i-1]))
    #-- do one final sigmoid convolution into just 1 final channel (None,h,w,1)
    i += 1
    c[i] = BatchNormalization(axis=-1)(kl.Conv2D(1,1,activation='sigmoid')(c[i-1]))
    #-- reshape into a flattened output to match sample weights
    i += 1
    c[i] = kl.Reshape((height*width,1,))(c[i-1])

    print('output shape: ', c[i].shape)
    print('Total Number of layers: ',i)


    #-- make model
    model = km.Model(inputs=inputs,outputs=c[i])

    #-- return model
    return model
