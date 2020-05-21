#!/usr/bin/env python
u"""
Yara Mohajerani (05/2020)

In order to keep track of which line segments are train and which are test,
we use the ML line are the reference. But not all lines have a corresponding 
hand-drawn lines. So for each test line, we look for any intersecting handdrawn line.
If such a line exists, we choose the line with the shorter length and do a closest-pixel
pairwise distance mean.
"""
import os
import sys
import fiona
import getopt
import numpy as np
from shapely.geometry import LineString

#-- directory setup
ddir = os.path.expanduser('~/GL_learning_data/geocoded_v1')
gdrive = os.path.join(os.path.expanduser('~'),'Google Drive File Stream',
	'Shared drives','GROUNDING_LINE_TEAM_DRIVE','ML_Yara')

#-- main function
def main():
	#-- Read the system arguments listed after the program
	long_options=['DIR=','FILTER=']
	optlist,arglist = getopt.getopt(sys.argv[1:],'D:F:',long_options)

	#-- Set default settings
	subdir = 'atrous_32init_drop0.2_customLossR727.dir'
	FILTER = 0.
	flt_str = ''
	for opt, arg in optlist:
		if opt in ("-D","--DIR"):
			subdir = arg
		elif opt in ("-F","--FILTER"):
			if arg not in ['NONE','none','None','N','n',0]:
				FILTER = float(arg)
				flt_str = '_%.1fkm'%(FILTER/1000)

	#-- Get list of postprocessed files
	pred_dir = os.path.join(ddir,'stitched.dir',subdir,'shapefiles.dir')
	fileList = os.listdir(pred_dir)
	pred_list = [f for f in fileList if (f.endswith('%s.shp'%flt_str) and (f.startswith('gl_')))]

	#-- also get list of handdrawn lines
	lbl_dir = os.path.join(gdrive,'SOURCE_SHP')
	fileList = os.listdir(lbl_dir)
	lbl_list = [f for f in fileList if (f.endswith('.shp') and (f.startswith('gl_')))]

	#-- out file for saving average error for each file
	outtxt = open(os.path.join(pred_dir,'error_summary.txt'),'w')
	outtxt.write('Average Error (m) \t File\n')
	#-- initialize array for distances
	distances = np.zeros(len(pred_list))
	#-- go through files and get pairwise distances
	for count,f in enumerate(pred_list):
		#-- read ML line
		fid1 = fiona.open(os.path.join(pred_dir,f),'r')
		

		#-- loop over ML lines and save all test lines
		ml_lines = []
		for j in range(len(fid1)):
			g = fid1.next()
			if (('err' not in g['properties']['ID']) and (g['properties']['Type']=='Test')\
				and (g['properties']['Class'] in ['Pinning Point','Grounding Line'])):
				ml_lines.append(LineString(g['geometry']['coordinates']))
		fid1.close()

		#-- find corresponding label file
		f_ind = lbl_list.index(f.replace('%s.shp'%flt_str,'.shp'))
		#-- read label file
		fid2 = fiona.open(os.path.join(lbl_dir,f.replace('%s.shp'%flt_str,'.shp')),'r')
		#-- loop over the hand-written lines and save all coordinates
		hd_lines = []
		for j in range(len(fid2)):
			g2 = fid2.next()
			if not g2['geometry'] is None:
				hd_lines.append(LineString(g2['geometry']['coordinates']))
		fid2.close()

		#-- initialize array of all pairwise distances
		dist = []
		#-- Now loop over each test line and find intersecting handdrawn lines
		for l1 in ml_lines:
			for l2 in hd_lines:
				if l1.intersects(l2) or l1.touches(l2):
					#-- lines intersect. Now Find the shorter line to use as reference
					if l1.length <= l2.length:
						#-- use ML line as reference
						x1,y1 = l1.coords.xy
						x2,y2 = l2.coords.xy
					else:
						#-- use manual line as reference (set as x1,y1)
						x1,y1 = l2.coords.xy
						x2,y2 = l1.coords.xy
					#-- go along x1,y1 and find closest points on x2,y2
					for i in range(len(x1)):
						#-- get list of distances
						d = np.sqrt((np.array(x2)-x1[i])**2 + (np.array(y2)-y1[i])**2)
						#-- save shortest distances
						dist.append(np.min(d))
		distances[count] = np.mean(dist)
		outtxt.write('%.2f \t%s\n'%(distances[count],f))

	#-- also save the overal average
	outtxt.write('%.2f \tMEAN\n'%(np.nanmean(distances)))
	outtxt.close()

#-- run main program
if __name__ == '__main__':
	main()
