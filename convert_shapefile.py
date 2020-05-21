#!/usr/bin/env python
u"""
convert_shapefile.py
Yara Mohajerani (Last update 05/2020)

Read output predictions and convert to shapefile lines
"""
import os
import sys
import rasterio
import numpy as np
import skimage
import getopt
import shapefile
import scipy.ndimage as ndimage
from shapely.geometry import Polygon,LineString
from skimage.graph import route_through_array
from skimage.morphology import thin,skeletonize

#-- directory setup
ddir = os.path.expanduser('~/GL_learning_data/geocoded_v1')

#-- main function
def main():
	#-- Read the system arguments listed after the program
	long_options=['DIR=','FILTER=']
	optlist,arglist = getopt.getopt(sys.argv[1:],'D:F:',long_options)

	#-- Set default settings
	subdir = 'atrous_32init_drop0.2_customLossR727.dir'
	filter = 0.
	flt_str = ''
	for opt, arg in optlist:
		if opt in ("-D","--DIR"):
			subdir = arg
		elif opt in ("-F","--FILTER"):
			if arg not in ['NONE','none','None','N','n',0]:
				filter = float(arg)
				flt_str = '_%.1fkm'%(filter/1000)
	
	#-- Get list of files
	pred_dir = os.path.join(ddir,'stitched.dir',subdir)
	fileList = os.listdir(pred_dir)
	pred_list = [f for f in fileList if (f.endswith('.tif') and ('mask' not in f))]
	# pred_list = ['gl_069_181218-181224-181224-181230_014095-025166-025166-014270_T110614_T110655.tif']
	# pred_list = ['gl_007_180518-180524-180530-180605_021954-011058-022129-011233_T050854_T050855.tif']
	print('# of files: ', len(pred_list))
	#-- output directory
	output_dir = os.path.join(pred_dir,'shapefiles.dir')
	#-- make directories if they don't exist
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)

	#-- threshold for getting contours and centerlines
	eps = 0.3

	#-- loop through prediction files
	#-- get contours and save each as a line in shapefile
	#-- also save training label as line
	for f in pred_list:
		#-- read file
		raster = rasterio.open(os.path.join(pred_dir,f),'r')
		im = raster.read(1)
		#-- get transformation matrix
		trans = raster.transform

		#-- also read the corresponding mask file
		mask_file = os.path.join(pred_dir,f.replace('.tif','_mask.tif'))
		print(mask_file)
		mask_raster = rasterio.open(mask_file,'r')
		mask = mask_raster.read(1)
		mask_raster.close()

		#-- get contours of prediction
		#-- close contour ends to make polygons
		im[np.nonzero(im[:,0] > eps),0] = eps
		im[np.nonzero(im[:,-1] > eps),-1] = eps
		im[0,np.nonzero(im[0,:] > eps)] = eps
		im[-1,np.nonzero(im[-1,:] > eps)] = eps
		contours = skimage.measure.find_contours(im, eps)
		#-- make contours into closed polyons to find pinning points
		pols = [None]*len(contours)
		pol_type = [None]*len(contours)
		for n,contour in enumerate(contours):
			pols[n] = Polygon(zip(contour[:,0],contour[:,1]))
			#-- get elements of mask the contour is on
			submask = mask[np.round(contour[:, 0]).astype('int'), np.round(contour[:, 1]).astype('int')]
			#-- if more than half of the elements are from test tile, count contour as test type
			if np.count_nonzero(submask) > submask.size/2.:
				pol_type[n] = 'Test'
			else:
				pol_type[n] = 'Train'
			
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
		#-- In that case, ignore the inner contours. We add a new array for 
		#-- 'noise' points to be ignored.
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
		
		#-- also apply noise filter and append to noise list
		x = {}
		y = {}
		tmp_noise = []
		for idx,contour in enumerate(contours):
			#-- don't go over noise to cut uncessary iterations
			if idx not in noise:
				#-- convert to coordinates
				x[idx],y[idx] = rasterio.transform.xy(trans, contour[:,0], contour[:,1])
				#-- apply filter
				if (len(x[idx]) < 2 or LineString(zip(x[idx],y[idx])).length <= filter):
					tmp_noise.append(idx)

		#-- combine and remove repeating elements
		noise = list(set(noise+tmp_noise))

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
		cnts = [None]*len(contours)
		centers = [None]*(len(contours)-len(outer)-len(noise))
		#-- counters for centerlines and contours
		cc = 0 # contour counter
		n = 0  # center line counter
		pc = 1 # pinning point counter
		lc = 1 # line counter
		er_type = [None]*len(cnts)
		cn_type = [None]*len(centers)
		er_class = [None]*len(cnts)
		cn_class = [None]*len(centers)
		er_lbl = [None]*len(cnts)
		cn_lbl = [None]*len(centers)
		#-- convert to coordinates
		for idx,contour in enumerate(contours):
			cnts[cc] = [list(a) for a in zip(x[idx],y[idx])]
			er_type[cc] = pol_type[idx]
			if idx in noise:
				er_class[cc] = 'Noise'				
			elif idx in outer:
				er_class[cc] = 'Outer Contour'
			else:
				#-- In these cases there is a grounding line to be counted.
				#-- either pinning point or line
				cn_type[n] = pol_type[idx]
				#-- if this is an inner ring, then the centerline is the same as the contour
				if idx in inner:
					centers[n] = cnts[cc].copy()
					cn_class[n] = 'Pinning Point'
					er_class[cc] = 'Inner Contour'
					#-- set label
					cn_lbl[n] = 'pin%i'%pc
					pc += 1 #- incremenet pinning point counter
				#-- if neither inner nor outer (i.e. not pinnin point), get route through thinned line
				else:
					cn_class[n] = 'Grounding Line'
					er_class[cc] = 'GL Uncertainty'
					#-- get centerlines
					#-- initialize centerline plot
					im2 = np.zeros(im.shape, dtype=int)
					#-- draw line through contour
					im2[np.round(contour[:, 0]).astype('int'), np.round(contour[:, 1]).astype('int')] = 1
					im2 = thin(ndimage.binary_fill_holes(im2))
					nl = len(contour[:,0])

					#-- test start/end point every 10 pixels and find the longest distance to cover the
					#- whole line segment
					#-- increments of 10 for half-curve nl/2, so # of incremements is nl/20
					dist = np.zeros(int(nl/20)+1)
					for d in range(len(dist)):
						#-- for each index dc, get the point on the opposite side by adding
						#-- half the distance nl/2
						dist[d] = np.sqrt((contour[10*d,0] - contour[10*d+int(nl/2),0])**2 + \
							(contour[10*d,1] - contour[10*d+int(nl/2),1])**2)
					#-- find the index for the longest distance
					dist_ind = np.argmax(dist)
					startPoint = (int(round(contour[10*dist_ind,0])),int(round(contour[10*dist_ind,1])))
					endPoint = (int(round(contour[10*dist_ind+int(nl/2),0])),int(round(contour[10*dist_ind+int(nl/2),1])))
					inds, ws = route_through_array(1-im2, (startPoint[0], startPoint[1]),\
						(endPoint[0], endPoint[1]), geometric=True,fully_connected=True)
					#-- wrap up list of tuples
					ii,jj = zip(*inds)
					xc,yc = rasterio.transform.xy(trans, ii, jj)
					try:
						centers[n] = [list(a) for a in zip(xc,yc)]
					except:
						print('%s\nlen centers: %i, len outer: %i, len contours: %i, len noise: %i'%(
							f,len(centers),len(outer),len(contours),len(noise)))
						sys.exit('index out of bounds.')

					#-- set label
					cn_lbl[n] = 'line%i'%lc
					er_lbl[cc] = 'err%i'%lc
					lc += 1 #- incremenet line counter
				#-- increment centerline counter
				n += 1
			#-- increment contour counter
			cc += 1
		
		#-- save all linestrings to file
		#-- make separate files for centerlines and errors
		# 1) GL file
		gl_file = os.path.join(output_dir,f.replace('.tif','%s.shp'%flt_str))
		w = shapefile.Writer(gl_file)
		w.field('ID', 'C')
		w.field('Type','C')
		w.field('Class','C')
		#-- loop over contour centerlines
		for n in range(len(centers)):
			w.line([centers[n]])
			w.record(cn_lbl[n], cn_type[n], cn_class[n])
		w.close()
		# create the .prj file
		prj = open(gl_file.replace('.shp','.prj'), "w")
		prj.write(raster.crs.to_wkt())
		prj.close()

		# 2) Err File
		er_file = os.path.join(output_dir,f.replace('.tif','%s_ERR.shp'%flt_str))
		w = shapefile.Writer(er_file)
		w.field('ID', 'C')
		w.field('Type','C')
		w.field('Class','C')
		#-- loop over contours and write them
		for n in range(len(cnts)):
			w.line([cnts[n]])
			w.record(er_lbl[n] , er_type[n], er_class[n])
		w.close()
		# create the .prj file
		prj = open(er_file.replace('.shp','.prj'), "w")
		prj.write(raster.crs.to_wkt())
		prj.close()

		#-- close input file
		raster.close()

#-- run main program
if __name__ == '__main__':
	main()
