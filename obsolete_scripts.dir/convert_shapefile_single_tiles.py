#!/usr/bin/env python
u"""
convert_shapefile_single_tiles.py
Yara Mohajerani

Read output predictions before they're stitched together
 and the corresponding labels and convert to shapefile lines
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
from collections import Counter
from shapely.geometry import Polygon
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
	pred_list = {}
	output_dir = {}
	for t in ['Train']:#,'Test']:
		pred_dir = os.path.join(gdrive_out,'%s_predictions.dir'%t,subdir)
		fileList = os.listdir(pred_dir)
		# pred_list[t] = [f for f in fileList if (f.endswith('.png') and f.startswith('pred'))]
		pred_list[t] = ['pred_gl_054_180328-180403-180403-180409_010230-021301-021301-010405_T102357_T102440_x1590_y1024_DIR01.png']
		#-- output directory
		output_dir[t] = os.path.join(outdir,'%s_predictions.dir'%t,subdir,'shapefiles.dir')
		#-- make directories if they don't exist
		if not os.path.exists(output_dir[t]):
			os.mkdir(output_dir[t])

	#-- threshold for getting contours and centerlines
	eps = 0.3

	#-- loop through prediction files
	#-- get contours and save each as a line in shapefile
	#-- also save training label as line
	for t in ['Train']:#,'Test']:
		for f in pred_list[t]:
			#-- read prediction file
			im = imageio.imread(os.path.join(gdrive_out,'%s_predictions.dir'%t,subdir,f)).astype(float)/255.
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
			#-- make contours into closed polyons to find pinning points
			pols = [None]*len(contours)
			for n,contour in enumerate(contours):
				pols[n] = Polygon(zip(contour[:,0],contour[:,1]))
			#-- intialize matrix of polygon containment
			cmat = np.zeros((len(pols),len(pols)),dtype=bool)
			#-- initialize list of loops, inner and outer contours
			loops = []
			outer = []
			inner = []
			for i in range(len(pols)):
				for j in range(len(pols)):
					if (i != j) and pols[i].contains(pols[j]):
						#-- if the outer contour is significantly longer than the
						#-- inner contour, then it's not a pinning point but a loop
						#-- in the GL (use factor of 10 difference). In that case, get 
						#-- the inner loop instead
						if len(contours[i][:,0]) > 10*len(contours[j][:,0]):
							#-- the outer contour is a loop
							loops.append(i)
							#-- inner contour is considered pinning point
							inner.append(j)
						else:
							cmat[i,j] = True
			#-- However, note that if one outer control has more than 1 inner contour,
			#-- then it's not a pinning point and it's actually just noise.
			#-- In that case, ignore the inner contours. We add a new array for 'noise' points
			#-- to be ignored
			noise = []
			#-- get indices of rows with more than 1 True column in cmat
			for i in range(len(cmat)):
				if np.count_nonzero(cmat[i,:]) > 1:
					noise_idx, = np.nonzero(cmat[i,:])
					#-- concentante to noise list
					noise += list(noise_idx)
					#-- turn the matrix elements back off
					for j in noise_idx:
						cmat[i,j] = False 
			#-- remove repeating elements
			noise = list(set(noise))

			#-- go through overlapping elements and get nonoverlapping area to convert to 'donuts'
			#-- NOTE we will get the the contour corresponding to the inner ring
			for i in range(len(pols)):
				for j in range(len(pols)):
					if cmat[i,j] and (i not in noise) and (j not in noise):
						#-- save indices of inner and outer rings
						outer.append(i)
						if j not in loops:
							inner.append(j)
			#-- initialize list of contour linestrings
			cnts = [None]*(len(contours)-len(noise))
			centers = [None]*(len(contours)-len(outer)-len(noise))
			#-- counters for centerlines and contours
			cc = 0 # contouer counter
			n = 0  # center line counter
			#-- convert to coordinates
			for idx,contour in enumerate(contours):
				if idx not in noise:
					#-- convert to coordinates
					x,y = rasterio.transform.xy(trans, contour[:,0], contour[:,1])
					cnts[cc] = [list(a) for a in zip(x,y)]
					#-- get centerline onl if this is not an outer ring
					if idx not in outer:
						#-- if this is an inner ring, then the centerline is the same as the contour
						if idx in inner:
							centers[n] = cnts[cc].copy()
						#-- if neither inner nor outer (i.e. not pinnin point), get route through thinned line
						else:
							#-- get centerlines
							#-- initialize centerline plot
							im2 = np.zeros(im.shape, dtype=int)
							#-- draw line through contour
							im2[np.round(contour[:, 0]).astype('int'), np.round(contour[:, 1]).astype('int')] = 1
							im2 = thin(ndimage.binary_fill_holes(im2))
							nl = len(contour[:,0])
							startPoint = (int(round(contour[0,0])),int(round(contour[0,1])))
							endPoint = (int(round(contour[int(nl/2),0])),int(round(contour[int(nl/2),1])))
							inds, ws = route_through_array(1-im2, (startPoint[0], startPoint[1]),\
								(endPoint[0], endPoint[1]), geometric=True,fully_connected=True)
							#-- wrap up list of tuples
							ii,jj = zip(*inds)
							xc,yc = rasterio.transform.xy(trans, ii, jj)
							try:
								centers[n] = [list(a) for a in zip(xc,yc)]
							except:
								print(f)
								print(inner)
								print(outer)
								print('n: %i, cc: %i'%(n,cc))
								print(cmat)
								print('len centers: %i, len outer: %i, len contours: %i, len noise: %i'%(
									len(centers),len(outer),len(contours),len(noise)))
								sys.exit('index out of bounds.')
						#-- increment centerline counter
						n += 1
					#-- increment contour counter
					cc += 1
			
			#-- save all linestrings to file
			outfile = os.path.join(output_dir[t],f.replace('pred','post').replace('.png','.shp'))
			w = shapefile.Writer(outfile)
			w.field('ID', 'C')
			#-- loop over contours and write them
			for n in range(len(cnts)):
				w.line([cnts[n]])
				w.record('%i_%s'%(n,t))
			#-- loop over contour centerlines
			for n in range(len(centers)):
				w.line([centers[n]])
				w.record('cntr%i_%s'%(n,t))
			#-- loop over label contours and write them
			for n in range(len(lbl_cnts)):
				w.line([lbl_cnts[n]])
				w.record('lbl%i_%s'%(n,t))
			w.close()
			# create the .prj file
			prj = open(outfile.replace('.shp','.prj'), "w")
			prj.write(raster.crs.to_wkt())
			prj.close()

#-- run main program
if __name__ == '__main__':
	main()
