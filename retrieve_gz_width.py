#!/usr/bin/env
u"""
retrieve_gz_width.py
by Yara Mohajerani (12/2020)

Retreive GZ widths at a given coordinate and determine date of max/min GL
"""
from logging import warn
import os
import sys
import getopt
import pathlib
import random
import numpy as np
import geopandas as gpd
import netCDF4 as nc
from shapely.geometry import Point,MultiPoint,LineString,Polygon,MultiPolygon
from shapely import ops
from descartes import PolygonPatch
import matplotlib.pyplot as plt
from shapely.geometry.multilinestring import MultiLineString
from shapely.geometry.multipolygon import MultiPolygon

#-- function to calculate the GZ width
def calc_gz(GL_FILE='',WIDTH_FILE='',BASIN_FILE='',region='',N=0):
	#-- read the grounding lines and widths
	df_gl = gpd.read_file(GL_FILE)
	df_w = gpd.read_file(WIDTH_FILE)
	#-- read the basin file
	basins = gpd.read_file(BASIN_FILE)
	idx = basins.index[basins['NAME']==region]
	#-- get polygon
	poly = basins['geometry'][idx[0]]

	#-- add a 5km buffer to find the corresponding GLs
	region_poly = poly.buffer(5e3)
	
	lines = []
	dates = []
	for i in range(len(df_gl)):
		#-- extract geometry to see if it's in region of interest
		ll = df_gl['geometry'][i]
		if ll.intersects(region_poly):
			lines.append(ll)
			dates.append(df_gl['FILENAME'][i].split("_")[2])

	#-- get width lines
	ws = []
	for i in range(len(df_w)):
		ws.append(df_w['geometry'][i])
	widths = MultiLineString(ws)

	
	#-- get middle coordinates
	xlist = np.zeros(N)
	ylist = np.zeros(N)
	gz = np.zeros(N)
	date1_list = [None]*N
	date2_list = [None]*N
	ind_list = np.zeros(N,dtype=int)
	random.seed(11)
	for i in range(N):
		ind_list[i]= random.randrange(0,len(widths))
		mid_pt = widths[ind_list[i]].interpolate(0.5, normalized = True)
		xx,yy = mid_pt.coords.xy
		xlist[i] = float(xx[0])
		ylist[i] = float(yy[0])
		gz[i] = widths[ind_list[i]].length		
		#-- also get teh corresponding dates
		pt0 = widths[ind_list[i]].interpolate(0,normalized=True)
		pt1 = widths[ind_list[i]].interpolate(1,normalized=True)
		for l in range(len(lines)):
			if lines[l].distance(pt1) < 0.2:
				date1_list[i] = dates[l]
			elif lines[l].distance(pt0) < 0.2:
				date2_list[i] = dates[l]

	#-- write grounding zone widths to file
	outfile = os.path.join(os.path.dirname(GL_FILE),'GZ_retreived-widths_{0}.csv'.format(region))
	outfid = open(outfile,'w')
	outfid.write('X (m),Y (m),width (km),date1,date2\n')
	for i in range(N):
		outfid.write('{0:.6f},{1:.6f},{2:.3f},{3},{4}\n'.\
		format(xlist[i],ylist[i],gz[i]/1e3,date1_list[i],date2_list[i]))
	outfid.close()

	#-- plot a sample of points to check the grounding zones
	fig = plt.figure(1,figsize=(10,8))
	ax = fig.add_subplot(111)
	pp = PolygonPatch(poly,alpha=0.3,fc='lawngreen',ec='lawngreen',zorder=1)
	ax.add_patch(pp)
	for il in lines:
		xs,ys = il.coords.xy
		ax.plot(xs,ys,linewidth=0.4,alpha=0.8,color='k',zorder=2)
	for i in range(35):
		ip = random.randrange(0,N)
		#-- while distance to any of the previous points is less than 20km,
		#-- keep trying new indices (doesn't apply to 1st point)
		if i == 0:
			plot_pts = [Point(xlist[ip],ylist[ip])]
		else:
			pt = Point(xlist[ip],ylist[ip])
			while (pt.distance(MultiPoint(plot_pts)) < 10e3):
				ip = random.randrange(0,N)
				pt = Point(xlist[ip],ylist[ip])
			#-- now we can ensure the points aren't overlapping
			print("minimum distance to previous points: ", pt.distance(MultiPoint(plot_pts)))
			plot_pts.append(pt)
		#-- Now plot the transect for the given index
		lx,ly = widths[ind_list[ip]].coords.xy
		ax.plot(lx,ly,linewidth=2.0,alpha=1.0,color='red',zorder=3)
		ax.text(xlist[ip]+5e3,ylist[ip]+5e3,'{0:.1f}km'.format(gz[ip]/1e3),color='darkred',\
			fontsize=6,fontweight='bold',bbox=dict(facecolor='mistyrose', alpha=0.5))
	ax.get_xaxis().set_ticks([])
	ax.get_yaxis().set_ticks([])
	ax.set_title("Grounding Zone Width for {0}".format(region))
	plt.tight_layout()
	plt.savefig(outfile.replace('.csv','.pdf'),format='PDF')
	plt.close(fig)

#-- main function
def main():
	#-- Read the system arguments listed after the program
	long_options=['GL_FILE=','GZ_FILE=','WIDTH_FILE=','CENTER_FILE=','NUMBER=']
	optlist,arglist = getopt.getopt(sys.argv[1:],'L:Z:W:C:N:',long_options)

	GL_FILE = os.path.join(pathlib.Path.home(),'GL_learning_data',\
		'6d_results','AllTracks_6d_GL.shp')
	GZ_FILE = os.path.join(pathlib.Path.home(),'GL_learning_data',\
		'6d_results','GZ_Getz_final.shp')
	WIDTH_FILE = os.path.join(pathlib.Path.home(),'GL_learning_data',\
		'6d_results','GZ_Getz_widths_400m.shp')
	CENTER_FILE = os.path.join(pathlib.Path.home(),'GL_learning_data',\
		'6d_results','GZ_Getz_centerline.shp')
	BASIN_FILE = os.path.join(pathlib.Path.home(),'data.dir','basin.dir',\
		'Gates_Basin_v1.7','Basins_v2.4.shp')
	region = 'Getz'
	N = 100
	for opt, arg in optlist:
		if opt in ("-L","--GL_FILE"):
			GL_FILE = os.path.expanduser(arg)
		elif opt in ("-W","--WIDTH_FILE"):
			GZ_FILE = os.path.expanduser(arg)
		elif opt in ("-B","--BASIN_FILE"):
			BASIN_FILE = os.path.expanduser(arg)
		elif opt in ("-R","--REGION"):
			region = arg
		elif opt in ("-N","--NUMBER"):
			N = int(arg)

	#-- call the function to calculate the grounding zone width
	calc_gz(GL_FILE=GL_FILE,WIDTH_FILE=WIDTH_FILE,BASIN_FILE=BASIN_FILE,region=region,N=N)

#-- run main program
if __name__ == '__main__':
	main()