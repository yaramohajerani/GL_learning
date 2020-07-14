u"""
combine_shapfiles.py
Yara Mohajerani (07/2020)

Combine the individual GL shapefiles run in parallel
into combined files for each scene
"""
import os
import sys
import getopt
import pandas as pd
import geopandas as gpd
from rasterio.crs import CRS

in_base = os.path.expanduser('~')

#-- main function
def main():
	#-- Read the system arguments listed after the program
	long_options=['DIR=','FILTER=']
	optlist,arglist = getopt.getopt(sys.argv[1:],'D:F:',long_options)

	#-- Set default settings
	subdir = os.path.join('GL_learning_data','geocoded_v1'\
		,'stitched.dir','atrous_32init_drop0.2_customLossR727.dir')
	FILTER = 0.
	flt_str = ''
	for opt, arg in optlist:
		if opt in ("-D","--DIR"):
			subdir = arg
		elif opt in ("-F","--FILTER"):
			if arg not in ['NONE','none','None','N','n',0]:
				FILTER = float(arg)
				flt_str = '_%.1fkm'%(FILTER/1000)
	
	indir = os.path.join(in_base,subdir,'shapefiles.dir')

	#-- Get list of center files and complete scenes
	fileList = os.listdir(indir)
	center_list = [f for f in fileList if (f.endswith('.shp') and ('ERR' not in f) and ('%s_'%flt_str in f))]
	scene_list = [f for f in fileList if (f.endswith('_ERR.shp') and (flt_str in f))]

	print(len(scene_list))
	print(len(center_list))

	#-- read one error scene to get projection
	gdf = gpd.read_file(os.path.join(indir,scene_list[0]))
	crs_wkt = CRS.from_dict(gdf.crs).to_wkt()

	#-- loop over scenes and get all corresponding centerlines and combine
	for sc in scene_list:
		filename = sc.replace('ERR.shp','')
		#-- get all files in center list that start with `filename`
		sub_list = [f for f in center_list if f.startswith(filename)]
		#-- now combine each of the centerlines and combine
		gdf = []
		for ll in sub_list:
			gdf.append(gpd.read_file(os.path.join(indir,ll)))
		#-- concatenate dataframes
		combined = gpd.GeoDataFrame(pd.concat(gdf))
		#-- save to file
		combined.to_file(os.path.join(indir,filename[:-1]+'.shp'),driver='ESRI Shapefile',crs_wkt=crs_wkt)

#-- run main program
if __name__ == '__main__':
	main()
