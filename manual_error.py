#!/usr/bin/env python
u"""
manual_error.py
Yara Mohajerani

Calculate human error
"""
import os
import sys
import fiona
import getopt
import numpy as np
from scipy import stats
from shapely.geometry import LineString,Polygon
import matplotlib.pyplot as plt
from descartes import PolygonPatch

#-- main function
def main():
	#-- Read the system arguments listed after the program
	long_options=['DIR1=','DIR2=']
	optlist,arglist = getopt.getopt(sys.argv[1:],'O:T:',long_options)

	#-- Set default settings
	dir1 = os.path.expanduser('~/Google Drive File Stream/Shared drives/GROUNDING_LINE_TEAM_DRIVE/ML_Yara/SOURCE_SHP')
	dir2 = os.path.expanduser('~/GL_learning_data/Archive_Track007_Second_Draw_Bernd')
	for opt, arg in optlist:
		if opt in ("-O","--DIR1"):
			dir1 = os.path.expanduser(arg)
		elif opt in ("-T","--DIR2"):
			dir2 = os.path.expanduser(arg)
	
	#-- read list of files from both directories
	fileList = os.listdir(dir1)
	list1 = [f for f in fileList if (f.startswith('gl') and f.endswith('.shp'))]
	fileList = os.listdir(dir2)
	list2 = [f for f in fileList if (f.startswith('gl') and f.endswith('.shp'))]
	
	#------------------------------------------------------------------------------------------
	#-- go through list2 and get corresponding files in list1 and calculate pairwise errors
	#------------------------------------------------------------------------------------------
	#-- out file for saving average error for each file
	outtxt = open(os.path.join(dir2,'error_summary.txt'),'w')
	outtxt.write('Average\tError(m)\tMIN(m)\tMAX(m) \t File\n')
	#-- initialize array for distances, minimums, and maxmimums
	distances = np.zeros(len(list2))
	minims = np.zeros(len(list2))
	maxims = np.zeros(len(list2))
	#-- go through files and get pairwise distances
	for count,f in enumerate(list2):
		#-- read file
		fid1 = fiona.open(os.path.join(dir2,f),'r')
		#-- loop over ML lines and save all test lines
		ml_lines = []
		for j in range(len(fid1)):
			g = fid1.next()
			if not g['geometry'] is None:
				#-- read geometry and filter out pinning points for an equivalent 
				#-- error comparison to ML
				temp_pol = Polygon(g['geometry']['coordinates'])
				box_ll = temp_pol.length
				box_ww = temp_pol.area/box_ll
				#-- if the with is larger than 1/25 of the length, it's a pinning point
				if box_ww <= box_ll/25:
					ml_lines.append(LineString(g['geometry']['coordinates']))
		fid1.close()

		#-- read second file file
		try:
			ind = int(list1.index(f))
		except:
			print("Couldn't find file:", f)
			continue
		else: 
			fid2 = fiona.open(os.path.join(dir1,list1[ind]),'r')
			#-- loop over the hand-written lines and save all coordinates
			hd_lines = []
			for j in range(len(fid2)):
				g2 = fid2.next()
				if not g2['geometry'] is None:
					#-- read geometry and filter out pinning points for an equivalent 
					#-- error comparison to ML
					temp_pol = Polygon(g2['geometry']['coordinates'])
					box_ll = temp_pol.length
					box_ww = temp_pol.area/box_ll
					#-- if the with is larger than 1/25 of the length, it's a pinning point
					if box_ww <= box_ll/25:
						hd_lines.append(LineString(g2['geometry']['coordinates']))
			fid2.close()

			#-- initialize plot for the large errors
			fig = plt.figure(1, figsize=(8,8))
			ax = fig.add_subplot(111)

			#-- plot distnaces if specified
			#-- initialize array of all pairwise distances and used indices
			dist = []
			#-- loop over ML line segments and form bounding boxes
			for ml_original in ml_lines:
				#-- break the line into n_seg segments
				lcoords = list(ml_original.coords)
				if len(lcoords) < 15:
					ml_broken = [ml_original]
				else:
					#-- start from segments of 10 coordiantes, and increase until 
					#-- you find a divisible number
					n_seg = 15
					while len(lcoords)%n_seg < 5:
						n_seg += 1
					ml_broken = []
					bc = 0
					while bc < len(lcoords):
						ml_broken.append(LineString(lcoords[bc:bc+n_seg]))
						bc += n_seg
				for ml in ml_broken:
					box = ml.minimum_rotated_rectangle
					#-- get hd line segment that intersects the box
					for hd in hd_lines:
						overlap = hd.intersection(box)
						#-- if more than 20% of length is within the box, consider the line
						if overlap.length > hd.length/5:
							if box.geom_type == 'Polygon':
								ppatch = PolygonPatch(box,alpha=0.2,facecolor='skyblue')
								ax.add_patch(ppatch)
							#-- we have found the line pairning. Get mean distance
							#-- lines intersect. Now Find the shorter line to use as reference
							if ml.length <= hd.length:
								#-- use ML line as reference
								x1,y1 = ml.coords.xy
								x2,y2 = hd.coords.xy
								ax.plot(x1,y1,color='red')
								ax.plot(x2,y2,color='blue')
							else:
								#-- use manual line as reference (set as x1,y1)
								x1,y1 = hd.coords.xy
								x2,y2 = ml.coords.xy
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
			outtxt.write('%.1f \t %.1f \t %.1f \t\t %s\n'%(distances[count],minims[count],maxims[count],f))

			plt.savefig(os.path.join(dir2,f.replace('.shp','_dist.pdf')),format='PDF')
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


