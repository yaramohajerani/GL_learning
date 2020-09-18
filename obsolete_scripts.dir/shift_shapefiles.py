#!/usr/bin/env python
u"""
shift_shapefiles.py
Yara Mohajerani

Shift centerlines to fix the point mode / area mode
geotiff shift

http://geotiff.maptools.org/spec/geotiff2.5.html
"""
import os
import sys
import getopt
import numpy as np
import pandas as pd
import geopandas as gpd
from rasterio.crs import CRS
from shapely.geometry import LineString

#-- main function
def main():
	#-- Read the system arguments listed after the program
	long_options=['DIR=','FILTER=','DX=','DY=']
	optlist,arglist = getopt.getopt(sys.argv[1:],'D:F:X:Y:',long_options)

	#-- Set default settings
	ddir = os.path.join(os.path.expanduser('~'),'GL_learning_data',\
		'geocoded_v1','stitched.dir',\
		'atrous_32init_drop0.2_customLossR727.dir','shapefiles.dir')
	FILTER = 6000
	dx = 100.
	dy = 100.
	for opt, arg in optlist:
		if opt in ("-D","--DIR"):
			ddir = os.path.expanduser(arg)
		elif opt in ("-F","--FILTER"):
			if arg not in ['NONE','none','None','N','n',0]:
				FILTER = float(arg)
		elif opt in ('-X','--DX'):
			dx = float(arg)
		elif opt in ('-Y','--DY'):
			dy = float(arg)
		
	flt_str = '_%.1fkm'%(FILTER/1000)

	#-- Get list of files
	fileList = os.listdir(ddir)
	pred_list = [f for f in fileList if (f.endswith('%s.shp'%flt_str) and ('ERR' not in f))]
	print(pred_list)
	print(ddir)
	#-- read one error scene to get projection
	gdf = gpd.read_file(os.path.join(ddir,pred_list[0]))
	crs_wkt = CRS.from_dict(gdf.crs).to_wkt()

	#-- go through shapefiles and shift them by half a pixel
	for f in pred_list:
		gdf = gpd.read_file(os.path.join(ddir,f))
		for g in range(len(gdf['geometry'])):
			#-- get coordinates
			x,y = gdf['geometry'][g].coords.xy
			#-- convert to numpy array
			x = np.array(x)
			y = np.array(y)
			#-- add offset
			x -= dx/2
			y += dy/2
			#-- make new linestring
			ll = LineString(zip(x,y))
			#-- replace original geometry
			gdf['geometry'][g] = ll
		try:
			#-- save update geodataframe to file
			gdf.to_file(os.path.join(ddir,f),driver='ESRI Shapefile',crs_wkt=crs_wkt)
		except:
			print('empty file.')
			print(f)
			pass

#-- run main program
if __name__ == '__main__':
	main()
