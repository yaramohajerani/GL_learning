#!/usr/bin/env python
u"""
convert_shapefile_centerline.py
Yara Mohajerani (Last update 06/2020)

Read output predictions and convert to shapefile lines
This script uses the centerline module 
"""
import os
import sys
import rasterio
import numpy as np
import skimage
import getopt
import shapefile
import scipy.ndimage as ndimage
from shapely.geometry import Polygon,LineString,Point
from centerline.geometry import Centerline
from shapely.ops import linemerge

#-- directory setup
ddir = os.path.expanduser('~/GL_learning_data/geocoded_v1')

#-- main function
def main():
	#-- Read the system arguments listed after the program
	long_options=['DIR=','FILTER=','CLOBBER']
	optlist,arglist = getopt.getopt(sys.argv[1:],'D:F:C',long_options)

	#-- Set default settings
	subdir = 'atrous_32init_drop0.2_customLossR727.dir'
	FILTER = 0.
	flt_str = ''
	clobber = False
	for opt, arg in optlist:
		if opt in ("-D","--DIR"):
			subdir = arg
		elif opt in ("-F","--FILTER"):
			if arg not in ['NONE','none','None','N','n',0]:
				FILTER = float(arg)
				flt_str = '_%.1fkm'%(FILTER/1000)
		elif opt in ("-C","--CLOBBER"):
			clobber = True
			
	
	#-- Get list of files
	pred_dir = os.path.join(ddir,'stitched.dir',subdir)
	# fileList = os.listdir(pred_dir)
	# pred_list = [f for f in fileList if (f.endswith('.tif') and ('mask' not in f))]
	#-- output directory
	output_dir = os.path.join(pred_dir,'shapefiles.dir')
	#-- make directories if they don't exist
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)
	"""
	#-- if CLOBBBER is False, we are not overwriting old files, so remove exisiting files from list
	if not clobber:
		print('Removing exisitng files.')
		existingList = os.listdir(output_dir)
		existing = [f for f in existingList if (f.endswith('.shp') and ('ERR' not in f) and f.startswith('gl_'))]
		rem_list = []
		for p in pred_list:
			if p.replace('.tif','%s.shp'%flt_str) in existing:
				#-- save index for removing at the end
				rem_list.append(p)
		for p in rem_list:
			print('Ignoring %s.'%p)
			pred_list.remove(p)
	"""
	pred_list = ['gl_069_181218-181224-181224-181230_014095-025166-025166-014270_T110614_T110655.tif']
	# pred_list = ['gl_007_180518-180524-180530-180605_021954-011058-022129-011233_T050854_T050855.tif']
	print('# of files: ', len(pred_list))

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
		#-- also apply noise filter and append to noise list
		x = {}
		y = {}
		noise = []
		pols = [None]*len(contours)
		pol_type = [None]*len(contours)
		for n,contour in enumerate(contours):
			#-- convert to coordinates
			x[n],y[n] = rasterio.transform.xy(trans, contour[:,0], contour[:,1])

			pols[n] = Polygon(zip(x[n],y[n]))
			#-- get elements of mask the contour is on
			submask = mask[np.round(contour[:, 0]).astype('int'), np.round(contour[:, 1]).astype('int')]
			#-- if more than half of the elements are from test tile, count contour as test type
			if np.count_nonzero(submask) > submask.size/2.:
				pol_type[n] = 'Test'
			else:
				pol_type[n] = 'Train'
		
		#-- Loop through all the polygons and taking any overlapping areas out
		#-- of the enclosing polygon and ignore the inside polygon
		ignore_list = []
		for i in range(len(pols)):
			for j in range(len(pols)):
				if (i != j) and pols[i].contains(pols[j]):
					pols[i] = pols[i].difference(pols[j])
					ignore_list.append(j)

		#-- loop through and apply noise filter
		for n in range(len(contours)):
			#-- apply filter
			if (n not in ignore_list) and (len(x[n]) < 2 or LineString(zip(x[n],y[n])).length <= FILTER):
				noise.append(n)

		#-- loop through remaining polygons and determine which ones are 
		#-- pinning points based on the width and length of the bounding box
		pin_list = []
		box_ll = [None]*len(contours)
		box_ww = [None]*len(contours)
		for n in range(len(contours)):
			box_ll[n] = pols[n].length
			box_ww[n] = pols[n].area/box_ll[n]
			if (n not in noise) and (n not in ignore_list):
				#-- make bounding box
				# box = pols[n].minimum_rotated_rectangle
				# bx,by = box.exterior.coords.xy
				# #-- get the dimensions of the sides of the box
				# edge_length = (Point(bx[0],by[0]).distance(Point(bx[1],by[1])), Point(bx[1],by[1]).distance(Point(bx[2],by[2])))
				#-- length is the larger dimension
				# box_ll = max(edge_length)
				# #-- width is the smaller dimension
				# box_ww = min(edge_length)
				#-- if the with is larger than 1/4 of the length, it's a pinning point
				if box_ww[n] > box_ll[n]/25:
					pin_list.append(n)

		#-- find overlap between ignore list nad noise list
		print('overlap: ',list(set(noise) & set(ignore_list)))

		#-- initialize list of contour linestrings
		er = [None]*len(contours)
		cn = [] #[None]*(len(contours)-len(ignore_list)-len(noise))
		n = 0  # total center line counter
		pc = 1 # pinning point counter
		lc = 1 # line counter
		er_type = [None]*len(er)
		cn_type = [] #[None]*len(cn)
		er_class = [None]*len(er)
		cn_class = [] #[None]*len(cn)
		er_lbl = [None]*len(er)
		cn_lbl = [] #[None]*len(cn)
		#-- loop through polygons, get centerlines, and save
		for idx,p in enumerate(pols):
			er[idx] = [list(a) for a in zip(x[idx],y[idx])]
			er_type[idx] = pol_type[idx]
			if idx in noise:
				print('idx %i, n%i, noise.'%(idx,n))
				er_class[idx] = 'Noise'				
			elif idx in ignore_list:
				print('idx %i, n%i, ignore list.'%(idx,n))
				er_class[idx] = 'Inner Contour'
			else:
				if idx in pin_list:
					#-- pinning point. Just get perimeter of polygon
					xc,yc = pols[idx].exterior.coords.xy
					cn.append([[list(a) for a in zip(xc,yc)]])
					cn_class.append(['Pinning Point'])
					cn_type.append([pol_type[idx]])
					#-- set label
					cn_lbl.append(['pin%i'%pc])
					pc += 1 #- incremenet pinning point counter
				else:
					#-- get centerlines
					attributes = {"id": idx, "name": "polygon", "valid": True}
					#-- loop over interpolation distances until we can get a single line
					dis = pols[idx].length/100
					try:
						cl = Centerline(p,interpolation_distance=dis, **attributes)
					except:
						print('not enough ridges. Skip')
						continue
					else:
						# print('idx %i, n%i, contour.'%(idx,n))
						# print(dis,len(cl))
						"""
						while cl.geom_type == 'MultiLineString':
							try:
								dis *= 2
								cl = Centerline(p,interpolation_distance=dis, **attributes)
							except:
								print('cant increase distance. Merge line.')
								cl = Centerline(p,interpolation_distance=dis/2, **attributes)
								cl = linemerge(cl)
						"""
						print(cl.geom_type)
						
						#-- merge all the lines
						merged_lines = linemerge(cl)
						if merged_lines.geom_type == 'LineString':
							#-- save coordinates of linestring
							xc,yc = merged_lines.coords.xy
							cn.append([[list(a) for a in zip(xc,yc)]])
							cn_class.append(['Grounding Line'])
							cn_lbl.append(['line%i'%lc])
							cn_type.append([pol_type[idx]])
							er_class[idx] = 'GL Uncertainty'
							#-- set label
							er_lbl[idx] = 'err%i'%lc
							lc += 1 #- incremenet line counter
						else:
							nml = len(merged_lines)
							#-- for lines with many bifurcations, the average segment is 
							#-- about 300m, so if # of segments is length/300 or more, ignore.
							if nml < pols[idx].length/300:
								coord_list = []
								for nn in range(nml):
									xc,yc = merged_lines[nn].coords.xy
									coord_list.append([list(a) for a in zip(xc,yc)])
								cn.append(coord_list)
								cn_class.append(['Grounding Line']*nml)
								cn_lbl.append(['line%i'%lc]*nml)
								cn_type.append([pol_type[idx]]*nml)
								er_class[idx] = 'GL Uncertainty'
								"""
								#-- get longest line and plot
								merged_lines = linemerge(cl)
								line_ind = np.argmax([m.length for m in merged_lines])
								xc,yc = merged_lines[line_ind].coords.xy
								# xc,yc = cl.coords.xy
								cn[n] = [list(a) for a in zip(xc,yc)]
								"""
								#-- set label
								er_lbl[idx] = 'err%i'%lc
								lc += 1 #- incremenet line counter
				# #-- increment centerline counter
				# n += 1
		
		#-- save all linestrings to file
		#-- make separate files for centerlines and errors
		# 1) GL file
		gl_file = os.path.join(output_dir,f.replace('.tif','%s_TEST.shp'%flt_str))
		w = shapefile.Writer(gl_file)
		w.field('ID', 'C')
		w.field('Type','C')
		w.field('Class','C')
		#-- loop over contour centerlines
		for n in range(len(cn)):
			for nn in range(len(cn[n])):
				w.line([cn[n][nn]])
				w.record(cn_lbl[n][nn], cn_type[n][nn], cn_class[n][nn])
		w.close()
		# create the .prj file
		prj = open(gl_file.replace('.shp','.prj'), "w")
		prj.write(raster.crs.to_wkt())
		prj.close()

		# 2) Err File
		er_file = os.path.join(output_dir,f.replace('.tif','%s_TEST_ERR.shp'%flt_str))
		w = shapefile.Writer(er_file)
		w.field('ID', 'C')
		w.field('Type','C')
		w.field('Class','C')
		w.field('Length','C')
		w.field('Width','C')
		#-- loop over contours and write them
		for n in range(len(er)):
			w.line([er[n]])
			w.record(er_lbl[n] , er_type[n], er_class[n], box_ll[n], box_ww[n])
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
