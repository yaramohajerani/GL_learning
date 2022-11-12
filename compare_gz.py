#!/usr/bin/env python
u"""
compare_gz.py
by Yara Mohajerani (01/2021)

Compare delineated and expected GZ widths along
pre-determiend points
"""
import os
import sys
import getopt
import pathlib
import numpy as np
import pandas as pd
import rasterio as rio
from copy import copy
from shapely import ops
from descartes import PolygonPatch
import matplotlib.pyplot as plt

#-- compare GZ widths
def compare_gz(REF_FILE='',EXP_FILE=''):
	#-- read the delineated GZ
	df = pd.read_csv(REF_FILE)

	#-- read the expectation field
	src = rio.open(EXP_FILE)
	wi_field = src.read(1)
	xp,yp = rio.transform.xy(src.transform,np.arange(src.width),np.arange(src.height))
	src.close()

	#-- initialize output field
	fid = open(REF_FILE.replace('.csv','_comparison.csv'),'w')
	fid.write('ML_X,ML_Y,ML_WIDTH(km),EXP_X,EXP_Y,EXP_WIDHT(km),WIDTH DIFFERENCE(km)\n')
	#-- go through point, extract corresponding expectation, and save to file
	for i in range(len(df)):
		if df['date1'][i] != 'None' and df['date2'][i] != 'None':
			#-- extract delineated parameters
			xr = df['X (m)'][i]
			yr = df['Y (m)'][i]
			#-- Get widths
			wi_ml  = df['width (km)'][i]
			#-- get the closest point in the expectation field
			ii = np.argmin(np.abs(xr-xp))
			jj = np.argmin(np.abs(yr-yp))
			wi_ep = wi_field[ii,jj]
			#-- write to file
			fid.write('{0:.6f},{1:.6f},{2:.3f},{3:.6f},{4:.6f},{5:.3f},{6:.3f}\n'.\
				format(xr,yr,wi_ml,xp[ii],yp[jj],wi_ep,wi_ml-wi_ep))
	fid.close()

#-- main function
def main():
	#-- Read the system arguments listed after the program
	long_options=['REF_FILE=','EXP_FILE=']
	optlist,arglist = getopt.getopt(sys.argv[1:],'R:E:',long_options)

	REF_FILE = os.path.join(pathlib.Path.home(),'GL_learning_data',\
		'6d_results','GZ_widths-hybrid_Getz.csv')
	EXP_FILE = os.path.join(pathlib.Path.home(),'GL_learning_data',\
		'GL_Width_1meterTide_Bed_Machine-Velo18_GAMMA_along_Flow_percent.tif')
	
	for opt, arg in optlist:
		if opt in ("-R","--REF_FILE"):
			REF_FILE = os.path.expanduser(arg)
		elif opt in ("-E","--EXP_FILE"):
			EXP_FILE = os.path.expanduser(arg)

	#-- call the function to calculate the grounding zone width
	compare_gz(REF_FILE=REF_FILE,EXP_FILE=EXP_FILE)

#-- run main program
if __name__ == '__main__':
	main()