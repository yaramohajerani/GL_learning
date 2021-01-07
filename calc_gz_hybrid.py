#!/usr/bin/env
u"""
calc_gz_hybrid.py
by Yara Mohajerani (01/2021)

Calculate the width of the grounding zone by drawing an intersecting 
line in the direction of flow based on the velocity field in areas of 
fast flow, and retrieve widths from centerline calculation in QGIS
for areas of slow flow.
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
from shapely.geometry import Point,MultiPoint,LineString,MultiLineString,Polygon,MultiPolygon
from shapely import ops
from descartes import PolygonPatch
import matplotlib.pyplot as plt

#-- function to calculate the GZ width
def calc_gz(GL_FILE='',WIDTH_FILE='',BASIN_FILE='',VEL_FILE='',region='',dist=0,N=0,vel_thr=0):
	#-- read the grounding lines
	df_gl = gpd.read_file(GL_FILE)
	#-- read widths
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
	
	#-- merge all lines into linestring
	lm = ops.linemerge(lines)

	#-- also create a polygon to represent the GZ with a small buffer (10cm)
	gz_file = os.path.join(os.path.dirname(GL_FILE),'GZ_{0}.shp'.format(region))
	if os.path.exists(gz_file):
		print('Reading GZ polygon from file.')
		gz_df = gpd.read_file(gz_file)
		gz_poly = []
		for i in range(len(gz_df)):
			gz_poly.append(gz_df['geometry'][i])
		gz_poly = MultiPolygon(gz_poly)
	else:
		print('Creating GZ polygon.')
		ep = lm.buffer(1e-1)
		#-- get the boundary of the polygon containing all lines to make new polygon of just the envelope
		gz_poly = []
		for ip in ep:
			x,y = ip.exterior.coords.xy
			gz_poly.append(Polygon(zip(x,y)))
		#-- save the error polygon
		#-- first make DataFrame
		df = {'REGION':[],'center_x':[],'center_y':[]}
		out_geo = []
		for p in gz_poly:
			#-- note the width can be calculated from area and perimeter
			w = (-p.length+np.sqrt(p.length**2 - 16*p.area))/4
			l = p.length/2 - w
			print(w,l)
			if (w > 1 and l > 1):
				df['REGION'].append(region)
				x,y = p.centroid.coords.xy
				df['center_x'].append(x[0])
				df['center_y'].append(y[0])
				out_geo.append(p)
		out_gdf = gpd.GeoDataFrame(df,geometry=out_geo,crs=df_gl.crs)
		out_gdf.to_file(gz_file,driver='ESRI Shapefile')

	#-- read velocity field
	vel_fid = nc.Dataset(VEL_FILE,'r')
	x = vel_fid['x'][:]
	y = vel_fid['y'][:]
	vx = vel_fid['VX'][:]
	vy = vel_fid['VY'][:]
	#-- also read lat and lon
	# vel_lat = vel_fid['lat'][:]
	# vel_lon = vel_fid['lon'][:]
	vel_fid.close()
	
	#-- select the points for the calculation of GZ width
	#-- in order to generate points, we randomly draw a line from the
	#-- mutliline, and then draw a random distance to go along the line to get
	#-- a coordinate. We repeat until the specified number of points is reached
	xlist = np.zeros(N)
	ylist = np.zeros(N)
	gz = np.zeros(N)
	date1_list = [None]*N
	date2_list = [None]*N
	vel_transects = {}
	cn_transects = {}
	random.seed(13)
	for i in range(N):
		#-- draw a random index for line along multilines
		ind_line = random.randrange(0,len(lm))
		rand_line = lm[ind_line]
		rand_dist = random.uniform(0, rand_line.length)
		rand_pt = rand_line.interpolate(rand_dist)
		xx,yy = rand_pt.coords.xy
		xlist[i] = float(xx[0])
		ylist[i] = float(yy[0])
	
	#-- loop through points and calculate GZ
	for i,(xi,yi) in enumerate(zip(xlist,ylist)):
		if i%100 == 0:
			print(i)
		
		#-- A) velocity based approach
		#-- get list of distances to get a list of closest points
		#- For a given coordinate, get the flow angle and then the intersecting line
		ii = np.argmin(np.abs(x - xi))
		jj = np.argmin(np.abs(y - yi))

		#-- chech if velocity is above required threshold
		vel_mag = np.sqrt(vy[jj,ii]**2 + vx[jj,ii]**2)
		if vel_mag > vel_thr:
			#-- find flow angle
			ang = np.arctan(vy[jj,ii]/vx[jj,ii])
			#-- Now constuct a line of a given length, centered at the 
			#-- chosen coordinates, with the angle above
			dx,dy = dist*np.cos(ang),dist*np.sin(ang)
			vel_transects[i] = LineString([[x[ii]-dx,y[jj]-dy],[x[ii],y[jj]],[x[ii]+dx,y[jj]+dy]])
			#-- get intersection length
			vel_int = vel_transects[i].intersection(gz_poly)
			gz[i] = vel_int.length

			#-- get dates
			pt0 = vel_int.interpolate(0,normalized=True)
			pt1 = vel_int.interpolate(1,normalized=True)
			for l in range(len(lines)):
				if lines[l].distance(pt1) < 0.2:
					date1_list[i] = dates[l]
				elif lines[l].distance(pt0) < 0.2:
					date2_list[i] = dates[l]
		else:
			#-- B) retrieve width from QGIS centerline width calculation
			#-- first get the closest line to the point
			po = Point(xi,yi)
			wdist = np.zeros(len(widths))
			for wi in range(len(widths)):
				wdist[wi] = widths[wi].distance(po)
			ind_w = np.argmin(wdist)

			cn_transects[i] = widths[ind_w]
			#-- get length
			gz[i] = cn_transects[i].length	
			#-- also get the corresponding dates
			pt0 = cn_transects[i].interpolate(0,normalized=True)
			pt1 = cn_transects[i].interpolate(1,normalized=True)
			for l in range(len(lines)):
				if lines[l].distance(pt1) < 0.2:
					date1_list[i] = dates[l]
				elif lines[l].distance(pt0) < 0.2:
					date2_list[i] = dates[l]

	#-- write grounding zone widths to file
	outfile = os.path.join(os.path.dirname(GL_FILE),'GZ_widths-hybrid_{0}.csv'.format(region))
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
	for i in range(20):
		ip = random.randrange(0,N)
		#-- while distance to any of the previous points is less than 20km,
		#-- keep trying new indices (doesn't apply to 1st point)
		if i == 0:
			plot_pts = [Point(xlist[ip],ylist[ip])]
		else:
			pt = Point(xlist[ip],ylist[ip])
			while (pt.distance(MultiPoint(plot_pts)) < 20e3):
				ip = random.randrange(0,N)
				pt = Point(xlist[ip],ylist[ip])
			#-- now we can ensure the points aren't overlapping
			print("minimum distance to previous points: ", pt.distance(MultiPoint(plot_pts)))
			plot_pts.append(pt)
		#-- Now plot the transect for the given index
		if ip in vel_transects.keys():
			lx,ly = vel_transects[ip].coords.xy
			ax.plot(lx,ly,linewidth=2.0,alpha=1.0,color='red',zorder=3)
		elif ip in cn_transects.keys():
			lx2,ly2 = cn_transects[ip].coords.xy
			ax.plot(lx2,ly2,linewidth=2.0,alpha=1.0,color='orange',zorder=4)
		ax.text(xlist[ip]+5e3,ylist[ip]+5e3,'{0:.1f}km'.format(gz[ip]/1e3),color='darkred',\
			fontsize='small',fontweight='bold',bbox=dict(facecolor='mistyrose', alpha=0.5))
	ax.plot([],[],color='red',label="Velocity-based Intersect")
	ax.plot([],[],color='orange',label="Centerline-based Intersect")
	ax.get_xaxis().set_ticks([])
	ax.get_yaxis().set_ticks([])
	ax.set_title("Grounding Zone Width for {0}".format(region))
	plt.legend()
	plt.tight_layout()
	plt.savefig(outfile.replace('.csv','.pdf'),format='PDF')
	plt.close(fig)

#-- main function
def main():
	#-- Read the system arguments listed after the program
	long_options=['GL_FILE=','BASIN_FILE=','VEL_FILE=','REGION=','DIST=','NUMBER=','THRESHOLD']
	optlist,arglist = getopt.getopt(sys.argv[1:],'G:B:V:R:D:N:T:',long_options)

	GL_FILE = os.path.join(pathlib.Path.home(),'GL_learning_data',\
		'6d_results','AllTracks_6d_GL.shp')
	WIDTH_FILE = os.path.join(pathlib.Path.home(),'GL_learning_data',\
		'6d_results','GZ_Getz_widths_400m.shp')
	BASIN_FILE = os.path.join(pathlib.Path.home(),'data.dir','basin.dir',\
		'Gates_Basin_v1.7','Basins_v2.4.shp')
	VEL_FILE = os.path.join(pathlib.Path.home(),'data.dir','basin.dir',\
		'ANT_velocity.dir','antarctica_ice_velocity_450m_v2.nc')
	region = 'Getz'
	dist = 20e3
	N = 500
	vel_thr = 50
	for opt, arg in optlist:
		if opt in ("-G","--GL_FILE"):
			GL_FILE = os.path.expanduser(arg)
		elif opt in ("-W","--WIDTH_FILE"):
			WIDTH_FILE = os.path.expanduser(arg)
		elif opt in ("-B","--BASIN_FILE"):
			BASIN_FILE = os.path.expanduser(arg)
		elif opt in ("-V","--VEL_FILE"):
			VEL_FILE = os.path.expanduser(arg)
		elif opt in ("-R","--REGION"):
			region = arg
		elif opt in ("-D","--DIST"):
			dist = int(arg)
		elif opt in ("-N","--NUMBER"):
			N = int(arg)
		elif opt in ("-T","--THRESHOLD"):
			vel_thr = float(arg)

	#-- call the function to calculate the grounding zone width
	calc_gz(GL_FILE=GL_FILE,WIDTH_FILE=WIDTH_FILE,BASIN_FILE=BASIN_FILE,\
		VEL_FILE=VEL_FILE,region=region,dist=dist,N=N,vel_thr=vel_thr)

#-- run main program
if __name__ == '__main__':
	main()