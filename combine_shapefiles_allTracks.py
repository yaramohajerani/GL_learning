#!/usr/bin/env python
u"""
combine_shapfiles.py
Yara Mohajerani (08/2020)

Combine all individual shapefiles for all tracks into 1 file
"""
import os
import sys
import getopt
import pandas as pd
import geopandas as gpd
from rasterio.crs import CRS

#-- main function
def main():
	#-- Read the system arguments listed after the program
	long_options=['DIR=','FILTER=','MODEL=','ERROR']
	optlist,arglist = getopt.getopt(sys.argv[1:],'D:F:M:E',long_options)

	#-- Set default settings
	ddir = '/DFS-L/DATA/gl_ml/SENTINEL1_2018/'
	model_str = 'atrous_32init_drop0.2_customLossR727'
	FILTER = 8000
	error = False
	for opt, arg in optlist:
		if opt in ("-D","--DIR"):
			ddir = os.path.expanduser(arg)
		elif opt in ("-F","--FILTER"):
			if arg not in ['NONE','none','None','N','n',0]:
				FILTER = float(arg)
		elif opt in ("-M","--MODEL"):
			model_str = arg
		elif opt in ("-E","--ERROR"):
			error = True
	flt_str = '_%.1fkm'%(FILTER/1000)

	#-- get list of all folders (tracks)
	folderList = os.listdir(ddir)
	folder_list = [f for f in folderList if os.path.isdir(os.path.join(ddir,f))]
	print(folder_list)
	#-- initialize list to be converted to geopandas dataframe
	gdf = []
	for d in folder_list:
		#-- get list of files
		fileList = os.listdir(os.path.join(ddir,d,'%s.dir'%model_str,'stitched.dir','shapefiles.dir'))
		if error:
			file_list = [f for f in fileList if (f.endswith('%s_ERR.shp'%flt_str))]
		else:
			file_list = [f for f in fileList if (f.endswith('%s.shp'%flt_str))]

		print(d,len(file_list))
		for f in file_list:
			#-- read file
			g = gpd.read_file(os.path.join(ddir,d,'%s.dir'%model_str,'stitched.dir','shapefiles.dir',f))
			#-- remove rows corresponding to noise
			ind_remove = []
			for i in range(len(g)):
				if g['ID'][i] == None:
					ind_remove.append(i)
			g = g.drop(ind_remove)
			#-- also add file name to attribute table
			g['FILENAME'] = [f]*len(g['ID'])
			#-- add to main dataframe list
			gdf.append(g)

	#print(g.crs)
	#-- get projection to save file (same for all files, so just read last one)
	#crs_wkt = CRS.from_dict(g.crs).to_wkt()
	#crs_wkt = CRS.from_dict(init='epsg:3031').to_wkt()
	#print(crs_wkt)
	#-- concatenate dataframes
	combined = gpd.GeoDataFrame(pd.concat(gdf))
	#-- save to file
	if error:
		suffix = ''
	else:
		suffix = '_centerLines'
	combined.to_file(os.path.join(ddir,'combined_AllTracks%s.shp'%suffix),driver='ESRI Shapefile')#,crs_wkt=crs_wkt)

#-- run main program
if __name__ == '__main__':
	main()
