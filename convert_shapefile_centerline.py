u"""
convert_shapefile_thinning.py
Yara Mohajerani (Last update 07/2020)

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
from label_centerlines import get_centerline


#-- main function
def main():
	#-- Read the system arguments listed after the program
	long_options=['INPUT=','FILTER=']
	optlist,arglist = getopt.getopt(sys.argv[1:],'I:F',long_options)

	#-- Set default settings
	INPUT = os.path.join(os.path.expanduser('~'),'GL_learning_data','geocoded_v1',\
		'stitched.dir','atrous_32init_drop0.2_customLossR727.dir',\
		'gl_007_180518-180524-180530-180605_021954-011058-022129-011233_T050854_T050855.tif')
	FILTER = 0.
	flt_str = ''
	for opt, arg in optlist:
		if opt in ("-I","--INPUT"):
			INPUT = arg
		elif opt in ("-F","--FILTER"):
			if arg not in ['NONE','none','None','N','n',0]:
				FILTER = float(arg)
				flt_str = '_%.1fkm'%(FILTER/1000)
			
	#-- threshold for getting contours and centerlines
	eps = 0.3

	#-- read file
	raster = rasterio.open(INPUT,'r')
	im = raster.read(1)
	#-- get transformation matrix
	trans = raster.transform

	#-- also read the corresponding mask file
	mask_file = INPUT.replace('.tif','_mask.tif')
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
	contours = find_contours(im, eps)
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
	for n in range(len(pols)):
		box_ll[n] = pols[n].length
		box_ww[n] = pols[n].area/box_ll[n]
		if (n not in noise):
			#-- if the with is larger than 1/25 of the length, it's a pinning point
			if box_ww[n] > box_ll[n]/25:
				pin_list.append(n)

	#-- Loop through all the polygons and take any overlapping areas out
	#-- of the enclosing polygon and ignore the inside polygon
	ignore_list = []
	for i in range(len(pols)):
		for j in range(len(pols)):
			if (i != j) and pols[i].contains(pols[j]):
				# pols[i] = pols[i].difference(pols[j])
				if (i in pin_list) and (j in pin_list):
					#-- if it's a pinning point, ignore outer loop
					ignore_list.append(i)
				else:
					#-- if not, add inner loop to ignore list
					ignore_list.append(j)

	#-- find overlap between ignore list nad noise list
	if len(list(set(noise) & set(ignore_list))) != 0:
		sys.exit('Overlap not empty: ', list(set(noise) & set(ignore_list)))

	#-- initialize list of contour linestrings
	er = [None]*len(contours)
	cn = []
	n = 0  # total center line counter
	pc = 1 # pinning point counter
	lc = 1 # line counter
	er_type = [None]*len(er)
	cn_type = []
	er_class = [None]*len(er)
	cn_class = []
	er_lbl = [None]*len(er)
	cn_lbl = []
	#-- loop through polygons, get centerlines, and save
	for idx,p in enumerate(pols):
		er[idx] = [list(a) for a in zip(x[idx],y[idx])]
		er_type[idx] = pol_type[idx]
		if idx in noise:
			er_class[idx] = 'Noise'				
		elif idx in ignore_list:
			er_class[idx] = 'Ignored Contour'
		else:
			if idx in pin_list:
				#-- pinning point. Just get perimeter of polygon
				xc,yc = pols[idx].exterior.coords.xy
				cn.append([list(a) for a in zip(xc,yc)])
				cn_class.append('Pinning Point')
				cn_type.append(pol_type[idx])
				#-- set label
				cn_lbl.append('pin%i'%pc)
				pc += 1 #- incremenet pinning point counter
			else:
				dis = pols[idx].length/10
				mx = pols[idx].length/80
				merged_lines = get_centerline(p,segmentize_maxlen=dis,max_points=mx)
				#-- save coordinates of linestring
				xc,yc = merged_lines.coords.xy
				cn.append([list(a) for a in zip(xc,yc)])
				cn_class.append('Grounding Line')
				cn_lbl.append('line%i'%lc)
				cn_type.append(pol_type[idx])
				er_class[idx] = 'GL Uncertainty'
				#-- set label
				er_lbl[idx] = 'err%i'%lc
				lc += 1 #- incremenet line counter

	#-- save all linestrings to file
	output_dir = os.path.join(os.path.dirname(INPUT),'shapefiles.dir')
	if (not os.path.isdir(output_dir)):
			os.mkdir(output_dir)
	filename = os.path.basename(INPUT)
	#-- make separate files for centerlines and errors
	# 1) GL file
	gl_file = os.path.join(output_dir,filename.replace('.tif','%s.shp'%flt_str))
	w = shapefile.Writer(gl_file)
	w.field('ID', 'C')
	w.field('Type','C')
	w.field('Class','C')
	#-- loop over contour centerlines
	for n in range(len(cn)):
		w.line([cn[n]])
		w.record(cn_lbl[n], cn_type[n], cn_class[n])
	w.close()
	# create the .prj file
	prj = open(gl_file.replace('.shp','.prj'), "w")
	prj.write(raster.crs.to_wkt())
	prj.close()

	# 2) Err File
	er_file = os.path.join(output_dir,filename.replace('.tif','%s_ERR.shp'%flt_str))
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
