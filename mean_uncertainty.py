#!/usr/bin/env python
u"""
mean_uncertainty.py
Yara Mohajerani

Output the mean width of the uncertainty contours
Note this is indepedent from the error comparison with
the reference manual data.
"""
import os
import sys
import fiona
import getopt
import numpy as np
from scipy import stats
from shapely.geometry import Polygon

#-- directory setup
ddir = os.path.expanduser('~/GL_learning_data/geocoded_v1')

#-- main function
def main():
	#-- Read the system arguments listed after the program
	long_options=['DIR=','FILTER=']
	optlist,arglist = getopt.getopt(sys.argv[1:],'D:F:',long_options)

	#-- Set default settings
	subdir = 'atrous_32init_drop0.2_customLossR727.dir'
	FILTER = 6000.
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
	pred_list = [f for f in fileList if (f.endswith('%s_ERR.shp'%flt_str) and (f.startswith('gl_')))]

	#-- out file for saving average error for each file
	outtxt = open(os.path.join(pred_dir,'uncertainty_range%s.txt'%(flt_str)),'w')
	#-- initialize array for distances, minimums, and maxmimums
	ws = []
	#-- go through files and get pairwise distances
	for count,f in enumerate(pred_list):
		#-- read ML line
		fid1 = fiona.open(os.path.join(pred_dir,f),'r')
		#-- loop over ML lines and save all test lines
		for j in range(len(fid1)):
			g = fid1.next()
			if ((g['properties']['Type']=='Test') and (g['properties']['Class'] == 'GL Uncertainty')):
				# pol = Polygon(g['geometry']['coordinates'])
				# ws.append(pol.area/pol.length)
				ws.append(float(g['properties']['Width']))
		fid1.close()
	print(len(ws))
	#-- also save the overal average
	outtxt.write('MEAN\t%.1f m\n'%(np.nanmean(ws)))
	outtxt.write('MIN \t%.1f m\n'%(np.nanmin(ws)))
	outtxt.write('MAX \t%.1f m\n'%(np.nanmax(ws)))
	outtxt.write('IQR \t%.1f m\n'%(stats.iqr(ws,nan_policy='omit')))
	outtxt.write('MAD \t%.1f m\n'%(stats.median_absolute_deviation(ws,nan_policy='omit')))
	outtxt.write('STD \t%.1f m\n'%(np.nanstd(ws)))
	 
	outtxt.close()
	
#-- run main program
if __name__ == '__main__':
	main()

