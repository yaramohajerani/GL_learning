#!/usr/bin/env python
u"""
convert_shapefile.py
Yara Mohajerani

Read output predictions and labels and convert to lines in shapefile
"""
import os
import sys
import rasterio
import numpy as np
import fiona
import imageio
import skimage
import getopt
import shapefile
import scipy.ndimage as ndimage
from skimage.graph import route_through_array
from skimage.morphology import thin, skeletonize

gdrive = os.path.join(os.path.expanduser('~'),'Google Drive File Stream',
	'Shared drives','GROUNDING_LINE_TEAM_DRIVE','ML_Yara','geocoded_v1')
gdrive_out = output_dir = os.path.join(os.path.expanduser('~'),'Google Drive File Stream',
	'My Drive','GL_Learning')
outdir = os.path.expanduser('~/GL_learning_data/geocoded_v1')
#-- main function
def main():
	#-- Read the system arguments listed after the program
	long_options=['DIR=']
	optlist,arglist = getopt.getopt(sys.argv[1:],'D:',long_options)

	#-- Set default settings
	subdir = 'atrous_32init_drop0.2_customLossR727.dir'
	for opt, arg in optlist:
		if opt in ("-D","--DIR"):
			subdir = arg

	#-- Get list of geotiff label files
	lbl_dir = os.path.join(gdrive,'delineationtile_withoutnull_v1')
	fileList = os.listdir(lbl_dir)
	lbl_list = [f for f in fileList if (f.endswith('.tif') and f.startswith('delineation'))]
	
	#-- Get list of prediction files
	pred_dir = os.path.join(gdrive_out,'Test_predictions.dir',subdir)
	fileList = os.listdir(pred_dir)
	pred_list = [f for f in fileList if (f.endswith('.png') and f.startswith('pred'))]

	#-- output directory
	output_dir = os.path.join(outdir,'Test_predictions.dir',subdir,'shapefiles.dir')
	#-- make directories if they don't exist
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)

	#-- threshold for getting contours and centerlines
	eps = 0.3

	#-- loop through prediction files
	#-- get contours and save each as a line in shapefile
	#-- also save training label as line
	for f in pred_list:
		#-- read prediction file
		im = imageio.imread(os.path.join(gdrive_out,'Test_predictions.dir',subdir,f)).astype(float)/255.
		#-- Read corresponding label from tiff file
		#-- first find the index of the corresponding file
		file_ind = lbl_list.index(f.replace('pred','delineation').replace('.png','.tif'))
		raster = rasterio.open(os.path.join(gdrive,'delineationtile_withoutnull_v1',lbl_list[file_ind]))
		#-- get transformation matrix
		trans = raster.transform

		#-- read label line and divide to segments by getting contours
		lbl = raster.read(1)
		lbl_contours = skimage.measure.find_contours(lbl, 1.)
		#-- initialize list of contour linestrings
		lbl_cnts = [None]*len(lbl_contours)
		#-- convert to coordinates
		for n, contour in enumerate(lbl_contours):
			#-- convert to coordinates
			x,y = rasterio.transform.xy(trans, contour[:,0], contour[:,1])
			lbl_cnts[n] = [list(a) for a in zip(x,y)]
		
		#-- get contours of prediction
		#-- close contour ends to make polygons
		im[np.nonzero(im[:,0] > eps),0] = eps
		im[np.nonzero(im[:,-1] > eps),-1] = eps
		im[0,np.nonzero(im[0,:] > eps)] = eps
		im[-1,np.nonzero(im[-1,:] > eps)] = eps
		contours = skimage.measure.find_contours(im, eps)
		#-- initialize list of contour linestrings
		cnts = [None]*len(contours)
		centers = [None]*len(contours)
		#-- convert to coordinates
		for n,contour in enumerate(contours):
			#-- convert to coordinates
			x,y = rasterio.transform.xy(trans, contour[:,0], contour[:,1])
			cnts[n] = [list(a) for a in zip(x,y)]
			#-- get centerlines
			#-- initialize centerline plot
			im2 = np.zeros(im.shape, dtype=int)
			#-- draw line through contour
			im2[np.round(contour[:, 0]).astype('int'), np.round(contour[:, 1]).astype('int')] = 1
			im2 = thin(ndimage.binary_fill_holes(im2))
			# xtmp,ytmp = np.nonzero(im2)
			# startPoint = (int(xtmp[0]),int(ytmp[0]))
			# endPoint = (int(xtmp[-1]),int(ytmp[-1]))
			nl = len(contour[:,0])
			startPoint = (int(round(contour[0,0])),int(round(contour[0,1])))
			endPoint = (int(round(contour[int(nl/2),0])),int(round(contour[int(nl/2),1])))
			inds, ws = route_through_array(1-im2, (startPoint[0], startPoint[1]),\
				(endPoint[0], endPoint[1]), geometric=True,fully_connected=True)
			#-- wrap up list of tuples
			ii,jj = zip(*inds)
			xc,yc = rasterio.transform.xy(trans, ii, jj)
			centers[n] = [list(a) for a in zip(xc,yc)]
		
		#-- save all linestrings to file
		outfile = os.path.join(output_dir,f.replace('pred','post').replace('.png','.shp'))
		w = shapefile.Writer(outfile)
		w.field('ID', 'C')
		#-- loop over contours and write them
		for n in range(len(cnts)):
			w.line([cnts[n]])
			w.record(n)
		#-- loop over contour centerlines
		for n in range(len(centers)):
			w.line([centers[n]])
			w.record('cntr%i'%n)
		#-- loop over label contours and write them
		for n in range(len(lbl_cnts)):
			w.line([lbl_cnts[n]])
			w.record('lbl%i'%n)
		w.close()
		# create the .prj file
		prj = open(outfile.replace('.shp','.prj'), "w")
		prj.write(raster.crs.to_wkt())
		prj.close()

#-- run main program
if __name__ == '__main__':
	main()
