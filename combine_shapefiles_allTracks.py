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
	long_options=['DIR=','FILTER=','MODEL=']
	optlist,arglist = getopt.getopt(sys.argv[1:],'D:F:M:',long_options)

	#-- Set default settings
	ddir = '/DFS-L/DATA/gl_ml/SENTINEL1_2018/'
	model_str = 'atrous_32init_drop0.2_customLossR727'
	FILTER = 8000
	for opt, arg in optlist:
		if opt in ("-D","--DIR"):
			ddir = os.path.expanduser(arg)
		elif opt in ("-F","--FILTER"):
			if arg not in ['NONE','none','None','N','n',0]:
				FILTER = float(arg)
		elif opt in ("-M","--MODEL"):
			model_str = arg
	flt_str = '_%.1fkm'%(FILTER/1000)

	#-- get list of all folders (tracks)
	folderList = os.listdir(ddir)
	folder_list = [f for f in folderList if os.path.isdir(f)]
	print(folder_list)
	#-- initialize list to be converted to geopandas dataframe
	gdf = []
	for d in folder_list:
		#-- get list of files
		fileList = os.listdir(os.path.join(ddir,d,'%s.dir'%model_str,'stitched.dir','shapefiles.dir'))
		file_list = [f for f in fileList if (f.endswith('%s_ERR.shp'%flt_str))]

		print(d,len(file_list))
		for f in file_list:
			#-- read file
			g = gpd.read_file(os.path.join(ddir,d,'%s.dir'%model_str,'stitched.dir','shapefiles.dir',f))
			#-- also add file name to attribute table
			g['FILENAME'] = [f]*len(g['ID'])
			#-- add to main dataframe list
			gdf.append(g)

	#-- get projection to save file (same for all files, so just read last one)
	crs_wkt = CRS.from_dict(g.crs).to_wkt()

	#-- concatenate dataframes
	combined = gpd.GeoDataFrame(pd.concat(gdf))
	#-- save to file
	combined.to_file(os.path.join(ddir,'combined_AllTracks.shp'),driver='ESRI Shapefile',crs_wkt=crs_wkt)

#-- run main program
if __name__ == '__main__':
	main()
