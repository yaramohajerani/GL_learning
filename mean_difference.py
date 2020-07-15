#!/usr/bin/env python
u"""
Yara Mohajerani (05/2020)

Update History
	06/2020	Use bounding box to get line pairs and add plotting
	05/2020	Written
"""
import os
import sys
import fiona
import getopt
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from descartes import PolygonPatch
from shapely.geometry import LineString,Polygon

#-- directory setup
ddir = os.path.expanduser('~/GL_learning_data/geocoded_v1')
gdrive = os.path.join(os.path.expanduser('~'),'Google Drive File Stream',
	'Shared drives','GROUNDING_LINE_TEAM_DRIVE','ML_Yara')
lbl_dir = os.path.join(gdrive,'SOURCE_SHP')

#-- main function
def main():
	#-- Read the system arguments listed after the program
	long_options=['DIR=','FILTER=','NAMES=','PLOT']
	optlist,arglist = getopt.getopt(sys.argv[1:],'D:F:N:P',long_options)

	#-- Set default settings
	subdir = 'atrous_32init_drop0.2_customLossR727.dir'
	FILTER = 0.
	flt_str = ''
	plot_dists = False
	NAMES = None
	for opt, arg in optlist:
		if opt in ("-D","--DIR"):
			subdir = arg
		elif opt in ("-F","--FILTER"):
			if arg not in ['NONE','none','None','N','n',0]:
				FILTER = float(arg)
				flt_str = '_%.1fkm'%(FILTER/1000)
		elif opt in ("-P","--PLOT"):
			plot_dists = True
		elif opt in ("-N","--NAMES"):
			NAMES = arg

	#-- Get list of postprocessed files
	pred_dir = os.path.join(ddir,'stitched.dir',subdir,'shapefiles.dir')
	if NAMES is None:
		fileList = os.listdir(pred_dir)
		pred_list = [f for f in fileList if (f.endswith('%s.shp'%flt_str) and (f.startswith('gl_')))]
	else:
		pred_list = NAMES.split(',')

	#-- out file for saving average error for each file
	outtxt = open(os.path.join(pred_dir,'error_summary%s.txt'%(flt_str)),'w')
	outtxt.write('Average\tError(m)\tMIN(m)\tMAX(m) \t File\n')
	#-- initialize array for distances, minimums, and maxmimums
	distances = np.zeros(len(pred_list))
	minims = np.zeros(len(pred_list))
	maxims = np.zeros(len(pred_list))
	haus_max = np.zeros(len(pred_list))
	#-- go through files and get pairwise distances
	for count,f in enumerate(pred_list):
		#-- read ML line
		fid1 = fiona.open(os.path.join(pred_dir,f),'r')
		#-- loop over ML lines and save all test lines
		ml_lines = []
		for j in range(len(fid1)):
			g = fid1.next()
			gid = g['properties']['ID']
			if (('err' not in gid) and (g['properties']['Type']=='Test')):
				if (g['properties']['Class'] == 'Grounding Line'):
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

		#-- plot distnaces if specified
		if plot_dists:
			fig = plt.figure(1, figsize=(8,8))
			ax = fig.add_subplot(111)

		#-- initialize array of all pairwise distances and used indices
		dist = []
		#-- loop over ML line segments and form bounding boxes
		for ml_original in ml_lines:
			#-- if the line is more than 50 km long, break it into smaller lines
			if ml_original.length >= 50e3:
				#-- this is a rough breakdown. We can just break it into 100 indices at a time
				lcoords = list(ml_original.coords)
				ml_broken = []
				bc = 0
				while bc+100 < len(lcoords):
					ml_broken.append(LineString(lcoords[bc:bc+100]))
					bc += 100
				if bc < len(lcoords)-2:
					ml_broken.append(LineString(lcoords[bc:]))
			else:
				ml_broken = [ml_original]
			for ml in ml_broken:
				box = ml.minimum_rotated_rectangle
				#-- get hd line segment that intersects the box
				for hd in hd_lines:
					overlap = hd.intersection(box)
					#-- if more than 20% of length is within the box, consider the line
					if overlap.length > hd.length/5:
						if plot_dists:
							if box.geom_type == 'Polygon':
								ppatch = PolygonPatch(box,alpha=0.2,facecolor='skyblue')
								ax.add_patch(ppatch)
						#-- we have found the line pairning. Get mean distance
						#-- lines intersect. Now Find the shorter line to use as reference
						if ml.length <= hd.length:
							#-- use ML line as reference
							x1,y1 = ml.coords.xy
							x2,y2 = hd.coords.xy
							if plot_dists:
								ax.plot(x1,y1,color='red')
								ax.plot(x2,y2,color='blue')
						else:
							#-- use manual line as reference (set as x1,y1)
							x1,y1 = hd.coords.xy
							x2,y2 = ml.coords.xy
							if plot_dists:
								ax.plot(x1,y1,color='blue')
								ax.plot(x2,y2,color='red')

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
						#-- NOTE we make a list of the total indices, which allows us to also delete 
						#-- repeated indices (if not deleting, this is redundant. Can just use ':')
						xlist = np.arange(len(x1))
						for u in unique_list:
							w = np.argmin(d[xlist,u])
							dist.append(d[xlist[w],u])
							if plot_dists:
								ax.plot([x1[xlist[w]],x2[u]],[y1[xlist[w]],y2[u]],color='gray')
							#-- since we used this index, take it out
							# xlist = np.delete(xlist,w)
		distances[count] = np.mean(dist)
		if len(dist) != 0:
			minims[count] = np.min(dist)
			maxims[count] = np.max(dist)
		else:
			minims[count] = np.nan
			maxims[count] = np.nan
			haus_max[count] = np.nan
		outtxt.write('%.1f \t %.1f \t %.1f \t\t %s\n'%(distances[count],minims[count],maxims[count],f))
	
		if plot_dists:
			plt.savefig(os.path.join(pred_dir,f.replace('.shp','_dist.pdf')),format='PDF')
			plt.close(fig)

	#-- also save the overal average
	outtxt.write('\nMEAN\t\t\t\t%.1f m\n'%(np.nanmean(distances)))
	outtxt.write('MIN\t\t\t\t\t%.1f m\n'%(np.nanmin(minims)))
	outtxt.write('MAX\t\t\t\t\t%.1f m\n'%(np.nanmax(maxims)))
	outtxt.write('Interquartile Range\t%.1f m\n'%(stats.iqr(distances,nan_policy='omit')))
	outtxt.write('MAD\t\t\t\t\t%.1f m\n'%(stats.median_absolute_deviation(distances,nan_policy='omit')))
	outtxt.write('STD\t\t\t\t\t%.1f m\n'%(np.nanstd(distances)))
	 
	outtxt.close()
	
#-- run main program
if __name__ == '__main__':
	main()
