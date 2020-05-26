#!/usr/bin/env python
u"""
Yara Mohajerani (05/2020)

In order to keep track of which line segments are train and which are test,
we use the ML line are the reference. But not all lines have a corresponding 
hand-drawn lines. So for each test line, we look for any intersecting handdrawn line.
If such a line exists, we choose the line with the shorter length and do a closest-pixel
pairwise distance mean.


ALSO ADD MAX/MIN DIST FOR EVERY FILE
2 FILES WITH AND WITHOUT PINNING POINTS

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
	long_options=['DIR=','FILTER=','PINNING']
	optlist,arglist = getopt.getopt(sys.argv[1:],'D:F:P',long_options)

	#-- Set default settings
	subdir = 'atrous_32init_drop0.2_customLossR727.dir'
	FILTER = 0.
	flt_str = ''
	pinning = False #-- default don't include pinning points
	pin_str = '_noPinning'
	for opt, arg in optlist:
		if opt in ("-D","--DIR"):
			subdir = arg
		elif opt in ("-F","--FILTER"):
			if arg not in ['NONE','none','None','N','n',0]:
				FILTER = float(arg)
				flt_str = '_%.1fkm'%(FILTER/1000)
		elif opt in ("-P","--PINNING"):
			pinning = True
			pin_str = ''

	#-- Get list of postprocessed files
	pred_dir = os.path.join(ddir,'stitched.dir',subdir,'shapefiles.dir')
	fileList = os.listdir(pred_dir)
	pred_list = [f for f in fileList if (f.endswith('%s.shp'%flt_str) and (f.startswith('gl_')))]
	# pred_list = ['gl_007_180816-180822-180822-180828_012283-023354-023354-012458_T050839_T050924_6.0km.shp']
	#-- also get list of handdrawn lines
	lbl_dir = os.path.join(gdrive,'SOURCE_SHP')
	fileList = os.listdir(lbl_dir)
	lbl_list = [f for f in fileList if (f.endswith('.shp') and (f.startswith('gl_')))]

	#-- out file for saving average error for each file
	outtxt = open(os.path.join(pred_dir,'error_summary%s%s.txt'%(pin_str,flt_str)),'w')
	outtxt.write('Average\tError(m)\tMIN(m)\tMAX(m) \t File\n')
	#-- initialize array for distances, minimums, and maxmimums
	distances = np.zeros(len(pred_list))
	minims = np.zeros(len(pred_list))
	maxims = np.zeros(len(pred_list))
	#-- go through files and get pairwise distances
	for count,f in enumerate(pred_list):
		#-- read ML line
		fid1 = fiona.open(os.path.join(pred_dir,f),'r')
		

		#-- loop over ML lines and save all test lines
		ml_lines = []
		for j in range(len(fid1)):
			g = fid1.next()
			if (('err' not in g['properties']['ID']) and (g['properties']['Type']=='Test')):
				if ((pinning) and (g['properties']['Class'] in ['Pinning Point','Grounding Line'])) or\
					((not pinning) and (g['properties']['Class'] in ['Grounding Line'])):
					ml_lines.append(LineString(g['geometry']['coordinates']))
		fid1.close()

		#-- read label file
		fid2 = fiona.open(os.path.join(lbl_dir,f.replace('%s.shp'%flt_str,'.shp')),'r')
		#-- loop over the hand-written lines and save all coordinates
		hd_lines = []
		for j in range(len(fid2)):
			g2 = fid2.next()
			if not g2['geometry'] is None:
				hd_lines.append(LineString(g2['geometry']['coordinates']))
		fid2.close()

		#-- initialize array of all pairwise distances and used indices
		dist = []
		#-- Now loop over each test line and find intersecting handdrawn lines
		for l1 in ml_lines:
			for l2 in hd_lines:
				if l1.intersects(l2): #or l1.touches(l2):
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
					d = np.empty((len(x1),len(x2)),dtype=float)
					ind_list = np.empty(len(x1),dtype=int)
					for i in range(len(x1)):
						#-- get list of distances
						d[i,:] = np.sqrt((np.array(x2)-x1[i])**2 + (np.array(y2)-y1[i])**2)
						#-- get index of shortest distanace
						ind_list[i] = np.argmin(d[i,:])
					#-- Now check check if multiple points of the reference line point to the same
					#-- (x2,y2) point
					#-- first get list of unique indices
					unique_list = list(set(ind_list))
					#-- sort in ascending order
					unique_list.sort()

					#-- get how many times each unique index is repeated
					u_count = np.zeros(len(unique_list),dtype=int)
					#-- loop through unique indices and find all corresponding indices
					for k,u in enumerate(unique_list):
						u_count[k] = np.count_nonzero(ind_list == u)

					#-- for repeating indices that are side-by-side (for example many 4s and many 5s),
					#-- the line is out of bounds of the other line, and the far-away points are 
					#-- alternating between a few points on the refernec line. Make them all the same index
					remove_list = []
					for k in range(len(unique_list)):
						if u_count[k] > 1:
							#-- compare with element after
							if (unique_list[k]+1 in unique_list):
								ii, = np.nonzero(ind_list == unique_list[k])
								jj, = np.nonzero(ind_list == unique_list[k]+1)
								if np.min(d[ii,unique_list[k]]) < np.min(d[jj,unique_list[k]]):
									remove_list.append(unique_list[k]+1)
								else:
									remove_list.append(unique_list[k])
					#-- remove duplicate elements
					remove_list = list(set(remove_list))
					for r in remove_list:
						unique_list.remove(r)

					#-- loop through unique indices and find all corresponding indices
					for u in unique_list:
						ii, = np.nonzero(ind_list == u)
						if len(ii) == 1:
							dist.append(np.min(d[ii,:]))
						else:
							#-- if more than one element exists, find the one
							#-- with the shortest distance
							shortest = np.argmin(d[ii,u])
							dist.append(np.min(d[ii[shortest],:]))
		distances[count] = np.mean(dist)
		if len(dist) != 0:
			minims[count] = np.min(dist)
			maxims[count] = np.max(dist)
		else:
			minims[count] = np.nan
			maxims[count] = np.nan
		outtxt.write('%.1f \t %.1f \t %.1f \t\t %s\n'%(distances[count],minims[count],maxims[count],f))

	#-- also save the overal average
	outtxt.write('%.1f \tMEAN\n'%(np.nanmean(distances)))
	outtxt.write('%.1f \tMIN\n'%(np.nanmin(minims)))
	outtxt.write('%.1f \tMAX\n'%(np.nanmax(maxims)))
	outtxt.close()
	
#-- run main program
if __name__ == '__main__':
	main()
