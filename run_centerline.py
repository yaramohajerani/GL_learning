u"""
run_centerline.py
Yara Mohajerani (Last update 07/2020)

read polygonized contours and get centerlines for GLs
"""
import os
import sys
from copy import copy
import numpy as np
import shapefile
import geopandas as gpd
from shapely.geometry import Polygon,LineString,Point
from label_centerlines import get_centerline

#-- main function
def main():
	#-- Read the system arguments listed after the program
	if len(sys.argv) == 1:
		sys.exit('No input file given')
	else:
		input_list = sys.argv[1:]
	
	for INPUT in input_list:
		#-- read shapefile
		gdf = gpd.read_file(INPUT)
		out_gdf = copy(gdf)
		#-- remove 'err' from ID
		out_gdf['ID'][0] = gdf['ID'][0].replace('err','')
		#-- check if this is a pinning point
		if gdf['Class'][0] == 'Pinning Contour':
			#-- pinning point. Just get perimeter of polygon
			out_gdf['Class'][0] = 'Pinning Point'
		else:
			#-- convert to polygon
			p = Polygon(gdf['geometry'][0])
			dis = p.length/10
			mx = p.length/80
			out_gdf['geometry'][0] = get_centerline(p,segmentize_maxlen=dis,max_points=mx)
			out_gdf['Class'][0] = 'Grounding Line'

		#-- save centerline to file
		gl_file = INPUT.replace('_ERR','')
		out_gdf.to_file(gl_file)

#-- run main program
if __name__ == '__main__':
	main()
