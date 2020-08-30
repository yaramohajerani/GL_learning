u"""
cleanup.py
Yara Mohajerani (08/2020)

Use a static coast line to clean up the GL shapefile
"""
import os
import sys
import getopt
import pandas as pd
import geopandas as gpd

#-- main function
def main():
	#-- Read the system arguments listed after the program
	long_options=['THRESHOLD=','MASK_DIR=','INFILE=']
	optlist,arglist = getopt.getopt(sys.argv[1:],'T:M:I:',long_options)

	#-- Set default settings
	thresh = 15e3
	MASK_DIR = '/DFS-L/DATA/gl_ml/auxiliary/'
	INFILE = '/DFS-L/DATA/gl_ml/SENTINEL1_2018/combined_AllTracks.shp'
	for opt, arg in optlist:
		if opt in ("-T","--THRESHOLD"):
			thresh = int(arg)
		elif opt in ("-M","--MASK_DIR"):
			MASK_DIR = os.path.expanduser(arg)
		elif opt in ("-I","--INFILE"):
			INFILE = os.path.expanduser(arg)
	
	#-- read mask file
	mask = gpd.read_file(os.path.join(MASK_DIR,'ais_mask_%ikm.shp'%(thresh/1e3)))

	#-- read GLS
	gls = gpd.read_file(INFILE)

	#-- now go through and get list of GLs to delete
	n = len(mask)
	remove_ind = []
	for i in range(len(gls)):
		#-- start assuming GL is noise
		rm = True
		j = 0
		#-- as soon as one intersecting element is found, stop and move on.
		while rm and j < n:
			if mask['geometry'][j].intersects(gls['geometry'][i]):
				rm = False
			j += 1
		if rm:
			remove_ind.append(i)

	print("deleting %i lines"%len(remove_ind))

	#-- remove extra elements
	gls_out = gls.drop(remove_ind)

	#-- save to file
	gls_out.to_file(INFILE.replace('.shp','_cleaned_%ikm.shp'%(thresh/1e3)),driver='ESRI Shapefile')

#-- run main program
if __name__ == '__main__':
	main()

