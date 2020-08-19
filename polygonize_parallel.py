u"""
convert_shapefile_thinning.py
Yara Mohajerani (Last update 08/2020)

Read output predictions and convert to shapefile lines
"""
import os
import sys
import rasterio
import numpy as np
import getopt
import shapefile
from skimage.measure import find_contours
from shapely.geometry import Polygon,LineString,Point


#-- main function
def main():
	#-- Read the system arguments listed after the program
	long_options=['DIR=','FILTER=','BASE_DIR=','NUM=','START=','noMASK']
	optlist,arglist = getopt.getopt(sys.argv[1:],'D:F:B:N:S:M',long_options)

	#-- Set default settings
	subdir = os.path.join('geocoded_v1'\
		,'stitched.dir','atrous_32init_drop0.2_customLossR727.dir')
	FILTER = 0.
	base_dir = '/DFS-L/DATA/gl_ml/'
	num = 500
	start_ind = 0 
	for opt, arg in optlist:
		if opt in ("-D","--DIR"):
			subdir = arg
		elif opt in ("-F","--FILTER"):
			if arg not in ['NONE','none','None','N','n',0]:
				FILTER = float(arg)
		elif opt in ("B","--BASE_DIR"):
			base_dir = os.path.expanduser(arg)
		elif opt in ("B","--BASE_DIR"):
			base_dir = os.path.expanduser(arg)
		elif opt in ("N","--NUM"):
			num = int(arg)
		elif opt in ("S","--START"):
			start_ind = int(arg)
		elif opt in ("M","--noMASK"):
			make_mask = False
	flt_str = '_%.1fkm'%(FILTER/1000)

	#-- make sure out directory doesn't end with '\' so we can get parent directory
	if base_dir.endswith('/'):
		base_dir = base_dir[:-1]
	indir = os.path.join(base_dir,subdir)

	#-- Get list of files
	fileList = os.listdir(indir)
	pred_list = sorted([f for f in fileList if (f.endswith('.tif') and ('mask' not in f))])
	#-- LOCAL output directory
	local_output_dir = os.path.join(indir,'shapefiles.dir')
	#-- make output directory if it doesn't exist
	if not os.path.exists(local_output_dir):
		os.mkdir(local_output_dir)
	
	pred_list = pred_list[start_ind:start_ind+num]
	print('# of files: ', len(pred_list))
	
	#-- threshold for getting contours and centerlines
	eps = 0.3

	#-- loop through prediction files
	#-- get contours and save each as a line in shapefile format
	for f in pred_list:
		#-- read file
		raster = rasterio.open(os.path.join(indir,f),'r')
		im = raster.read(1)
		#-- get transformation matrix
		trans = raster.transform

		if make_mask:
			#-- also read the corresponding mask file
			mask_file = os.path.join(indir,f.replace('.tif','_mask.tif'))
			mask_raster = rasterio.open(mask_file,'r')
			mask = mask_raster.read(1)
			mask_raster.close()

		#-- get contours of prediction
		#-- close contour ends to make polygons
		im[np.nonzero(im[:,0] > eps),0] = eps
		im[np.nonzero(im[:,-1] > eps),-1] = eps
		im[0,np.nonzero(im[0,:] > eps)] = eps
		im[-1,np.nonzero(im[-1,:] > eps)] = eps
		contours = find_contours(im, eps)
		#-- make contours into closed polyons to find pinning points
		#-- also apply noise filter and append to noise list
		x = {}
		y = {}
		noise = []
		none_list = []
		pols = [None]*len(contours)
		pol_type = [None]*len(contours)
		for n,contour in enumerate(contours):
			#-- convert to coordinates
			x[n],y[n] = rasterio.transform.xy(trans, contour[:,0], contour[:,1])
			if len(x[n]) < 3:
				pols[n] = None
				none_list.append(n)
			else:
				pols[n] = Polygon(zip(x[n],y[n]))
			if make_mask:
				#-- get elements of mask the contour is on
				submask = mask[np.round(contour[:, 0]).astype('int'), np.round(contour[:, 1]).astype('int')]
				#-- if more than half of the elements are from test tile, count contour as test type
				if np.count_nonzero(submask) > submask.size/2.:
					pol_type[n] = 'Test'
				else:
					pol_type[n] = 'Train'
			else:
				pol_type[n] = 'Test'

		#-- loop through remaining polygons and determine which ones are 
		#-- pinning points based on the width and length of the bounding box
		pin_list = []
		box_ll = [None]*len(contours)
		box_ww = [None]*len(contours)
		for n in range(len(pols)):
			if n not in none_list:
				box_ll[n] = pols[n].length
				box_ww[n] = pols[n].area/box_ll[n]
				#-- if the with is larger than 1/25 of the length, it's a pinning point
				if box_ww[n] > box_ll[n]/25:
					pin_list.append(n)

		#-- Loop through all the polygons and take any overlapping areas out
		#-- of the enclosing polygon and ignore the inside polygon
		ignore_list = []
		for i in range(len(pols)):
			for j in range(len(pols)):
				if (i != j) and (i not in none_list) and (j not in none_list) and pols[i].contains(pols[j]):
					# pols[i] = pols[i].difference(pols[j])
					if (i in pin_list) and (j in pin_list):
						#-- if it's a pinning point, ignore outer loop
						ignore_list.append(i)
					else:
						#-- if not, add inner loop to ignore list
						ignore_list.append(j)

		#-- get rid of duplicates in ignore list
		ignore_list = list(set(ignore_list))

		#-- loop through and apply noise filter
		for n in range(len(contours)):
			#-- apply filter
			if (n not in none_list) and (n not in ignore_list) and (len(x[n]) < 2 or LineString(zip(x[n],y[n])).length <= FILTER):
				noise.append(n)

		#-- find overlap between ignore list nad noise list
		if len(list(set(noise) & set(ignore_list))) != 0:
			sys.exit('Overlap not empty: ', list(set(noise) & set(ignore_list)))
		#-- find overlap between ignore list nad none list
		if len(list(set(none_list) & set(ignore_list))) != 0:
			sys.exit('Overlap not empty: ', list(set(none_list) & set(ignore_list)))


		#-- initialize list of contour linestrings
		er = [None]*len(contours)
		n = 0  # total center line counter
		er_type = [None]*len(er)
		er_class = [None]*len(er)
		er_lbl = [None]*len(er)
		pc = 1 # pinning point counter
		lc = 1 # line counter
		#-- loop through polygons and save to separate files
		for idx,p in enumerate(pols):
			er[idx] = [list(a) for a in zip(x[idx],y[idx])]
			er_type[idx] = pol_type[idx]
			if (idx in noise) or (idx in none_list):
				er_class[idx] = 'Noise'			
			elif idx in ignore_list:
				er_class[idx] = 'Ignored Contour'
			else:
				if idx in pin_list:
					er_class[idx] = 'Pinning Contour'
					er_lbl[idx] = 'pin_err%i'%pc
					pc += 1 #- incremenet pinning point counter
				else:
					er_class[idx] = 'GL Uncertainty'
					#-- set label
					er_lbl[idx] = 'err%i'%lc
					lc += 1 #- incremenet line counter
				
		
		#-- save all contours to file
		er_file = os.path.join(local_output_dir,f.replace('.tif','%s_ERR.shp'%flt_str))
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
