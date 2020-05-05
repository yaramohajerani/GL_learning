#!/usr/bin/env python
u"""
combined_dates.py
Yara Mohajerani

Combine GL shapefiles for the same dates into one without the
training labels
"""
import os
import re
import sys
import getopt
import numpy as np
import fiona

base_dir = os.path.expanduser('~/GL_learning_data/geocoded_v1')
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

	#-- make full directory paths
	indir = os.path.join(base_dir,'Test_predictions.dir',subdir,'shapefiles.dir')
	outdir = os.path.join(base_dir,'Test_predictions.dir',subdir,'combinedDates.dir')
	#-- make directories if they don't exist
	if not os.path.exists(outdir):
		os.mkdir(outdir)

	#-- get list of files
	fileList = os.listdir(indir)
	flist = [f for f in fileList if (f.endswith('.shp') and f.startswith('post'))]

	#-- extract dates from filenames
	regex = re.compile('\d+')
	dates = [None]*len(flist)
	for i,f in enumerate(flist):
		nums = re.findall(regex,f)
		#-- objects are strings. Get the combination as 1 long number
		dates[i] = int(nums[1]+nums[2]+nums[3]+nums[4])
	dates = np.array(dates)

	#-- get number of unique dates
	ud = np.unique(dates)
	print(len(dates),len(ud))

	#-- go through dates and read and combine all corresponding files
	for u in ud:
		#-- make output label for given date
		s = str(u)
		date_lbl = '%s-%s-%s-%s'%(s[:6],s[6:12],s[12:18],s[18:24])
		#-- get indices of all matching dates
		ii, = np.nonzero(dates==u)
		for i in ii:
			#-- read shapefile flist[i]
			src = fiona.open(os.path.join(indir,flist[i]))
			meta = src.meta
			#-- counters
			c1,c2,c3 = 0,0,0
			#-- open output file for centerlines and errors
			with fiona.open(os.path.join(outdir,'centerline_%s.shp'%date_lbl),\
				'w', **meta) as dst1:
				with fiona.open(os.path.join(outdir,'errors_%s.shp'%date_lbl),\
					'w', **meta) as dst2:
					with fiona.open(os.path.join(outdir,'handdrawn_%s.shp'%date_lbl),\
						'w', **meta) as dst3:
						for j in range(len(src)):
							g = next(src)
							if g['properties']['ID'].startswith('cntr'):
								#-- change ID to file name
								g['properties']['ID'] = flist[i][flist[i].find('T'):-4]+'_%i'%c1
								#-- write center line to file
								dst1.write(g)
								c1 += 1
							elif (not g['properties']['ID'].startswith('lbl')):
								#-- change ID to file name
								g['properties']['ID'] = flist[i][flist[i].find('T'):-4]+'_%i'%c2
								#-- write center line to file
								dst2.write(g)
								c2 += 1
							else:
								#-- change ID to file name
								g['properties']['ID'] = flist[i][flist[i].find('T'):-4]+'_%i'%c3
								#-- write center line to file
								dst3.write(g)
								c3 += 1
			src.close()

#-- run main program
if __name__ == '__main__':
	main()

