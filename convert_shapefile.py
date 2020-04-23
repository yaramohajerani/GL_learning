#!/usr/bin/env python
u"""
convert_shapefile.py
Yara Mohajerani

Read output predictions and labels and convert to lines in shapefile
"""
import os
import sys
import numpy as np
import rasterio
import fiona
import imageio
import skimage
import getopt
import shapefile

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
		#-- get coordinates of label
		ii = np.nonzero(raster.read(1))
		x,y = trans*ii
		#-- convert to line
		lbl_line = [list(a) for a in zip(x,y)]
		#-- get contours of prediction
		contours = skimage.measure.find_contours(im, 0.3)
		#-- initialize list of contour linestrings
		cnts = [None]*len(contours)
		#-- convert to coordinates
		for n, contour in enumerate(contours):
			#-- convert to coordinates
			x,y = rasterio.transform.xy(trans, contour[:,0], contour[:,1])
			cnts[n] = [list(a) for a in zip(x,y)]
		#-- save all linestrings to file
		outfile = os.path.join(output_dir,f.replace('pred','post').replace('.png','.shp'))
		w = shapefile.Writer(outfile)
		w.field('ID', 'C')
		#-- loop over contours and write them
		for n in range(len(cnts)):
			w.line([cnts[n]])
			w.record(n)
		#-- also write label line
		w.line([lbl_line])
		w.record('label')	
		w.close()
		# create the .prj file
		prj = open(outfile.replace('shp','.prj'), "w")
		prj.write(raster.crs.to_wkt())
		prj.close()

#-- run main program
if __name__ == '__main__':
	main()
