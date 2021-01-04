#!/usr/bin/env
u"""
calc_gz.py
by Yara Mohajerani (12/2020)

Calculate the width of the grounding zone by drawing an intersecting 
line in the direction of flow based on the velocity field
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
from shapely.geometry.multipolygon import MultiPolygon

#-- function to calculate the GZ width
def calc_gz(GL_FILE='',BASIN_FILE='',VEL_FILE='',region='',dist=0,N=0):
	#-- read the grounding lines
	gdf = gpd.read_file(GL_FILE)
	#-- read the basin file
	basins = gpd.read_file(BASIN_FILE)
	idx = basins.index[basins['NAME']==region]
	#-- get polygon
	poly = basins['geometry'][idx[0]]

	#-- add a 5km buffer to find the corresponding GLs
	region_poly = poly.buffer(5e3)

	lines = []
	for i in range(len(gdf)):
		#-- extract geometry to see if it's in region of interest
		ll = gdf['geometry'][i]
		if ll.intersects(region_poly):
			lines.append(ll)
	
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
		out_gdf = gpd.GeoDataFrame(df,geometry=out_geo,crs=gdf.crs)
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
	vel_transects = {}
	perp_transects = {}
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
		ii = np.argsort(np.abs(x - xi))
		jj = np.argsort(np.abs(y - yi))

		#-- loop through the first 20 points and get the minimum width, so we dont
		#-- rely on a single point
		tmp_trans = {}
		tmp_dist = np.zeros(25)
		cc = 0
		for k in range(5):
			for w in range(5):
				#-- find flow angle
				ang = np.arctan(vy[ii[k],jj[w]]/vx[ii[k],jj[w]])
				#-- Now constuct a line of a given length, centered at the 
				#-- chosen coordinates, with the angle above
				dx,dy = dist*np.cos(ang),dist*np.sin(ang)
				tmp_trans[cc] = LineString([[x[ii[k]]-dx,y[jj[w]]-dy],[x[ii[k]],y[jj[w]]],[x[ii[k]]+dx,y[jj[w]]+dy]])
				
				if tmp_trans[cc].intersects(gz_poly):
					vel_int = tmp_trans[cc].intersection(gz_poly)
					tmp_dist[cc] = vel_int.length
				else:
					print("No intersection. i={0:d}, k={1:d}, w={2:d}".format(i,k,w))
				cc += 1

		ind_min = np.argmin(tmp_dist)	
		vel_transects[i] = tmp_trans[ind_min]
		gz[i] = tmp_dist[ind_min]

		
		# vel_int = vel_transects[i].intersection(gz_poly)
		# gz[i] = vel_int.length

		#-- B) tangent based appriach
		found_match = False
		if gz_poly.geom_type == 'Polygon':
			print("GZ object is polygon.")
			if vel_transects[i].intersects(gz_poly):
				x_ext,y_ext = gz_poly.exterior.coords.xy
				found_match = True
		elif gz_poly.geom_type == 'MultiPolygon':
			for sp in gz_poly:
				if vel_transects[i].intersects(sp):
					if found_match:
						print("More than one intersecting polygon found for point {0:d}".format(i))
					else:
						#-- get coordinates of the exterior of GZ polygon to be used later
						x_ext,y_ext = sp.exterior.coords.xy
						found_match = True
		else:
			sys.exit("Exiting. GZ object type: ",gz_poly.geom_type)
		
		#-- Check if any of the polygons intersect
		if not found_match:
			print("No matches found for point {0:d}".format(i))
			#-- move on to the next point and skip rest of iteration
			continue

		#-- Calculate GZ without velocity for comparison by constructing a perpendicular
		#-- line to the gz polygon at each point
		#-- 1) to get the tangent, get the index of the closest point on the boundary
		#-- of the gz polython
		dist2 = (x_ext - xi)**2 + (y_ext - yi)**2
		# ind = np.argmin(dist2)
		ind = np.argsort(dist2)

		#-- loop through the first few points and get the minimum width, so we dont
		#-- rely on a single point
		tmp_trans = {}
		tmp_dist = np.zeros(10)
		for k in range(10):
			#-- 2) calculate the slope of the tangent
			tangent = (y_ext[ind[k]]-y_ext[ind[k-1]])/(x_ext[ind[k]]-x_ext[ind[k-1]])

			#-- 3) calculate slope of perpendicular line
			slope_ang = np.arctan(-1/tangent)

			#-- 4) construct new transect and calculate width
			dx,dy = dist*np.cos(slope_ang),dist*np.sin(slope_ang)
			tmp_trans[k] = LineString([[x_ext[ind[k]]-dx,y_ext[ind[k]]-dy],[x_ext[ind[k]],y_ext[ind[k]]],[x_ext[ind[k]]+dx,y_ext[ind[k]]+dy]])

			perp_int = tmp_trans[k].intersection(gz_poly)
			tmp_dist[k] = perp_int.length

		ind_min = np.argmin(tmp_dist)
		perp_transects[i] = tmp_trans[ind_min]
		gz[i] = tmp_dist[ind_min]
		

	#-- write grounding zone widths to file
	outfile = os.path.join(os.path.dirname(GL_FILE),'GZ_widths_{0}.csv'.format(region))
	outfid = open(outfile,'w')
	outfid.write('X (m),Y (m),width (km)\n')
	for i in range(N):
		outfid.write('{0:.6f},{1:.6f},{2:.3f}\n'.format(xlist[i],ylist[i],gz[i]/1e3))
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
		lx,ly = vel_transects[ip].coords.xy
		ax.plot(lx,ly,linewidth=2.0,alpha=1.0,color='pink',zorder=3)
		if ip in perp_transects.keys():
			lx2,ly2 = perp_transects[ip].coords.xy
			ax.plot(lx2,ly2,linewidth=2.0,alpha=1.0,color='red',zorder=4)
		ax.text(xlist[ip]+5e3,ylist[ip]+5e3,'{0:.1f}km'.format(gz[ip]/1e3),color='darkred',\
			fontsize='small',fontweight='bold',bbox=dict(facecolor='mistyrose', alpha=0.5))
	ax.plot([],[],color='pink',label="Velocity-based Intersect")
	ax.plot([],[],color='red',label="Tangent-based Intersect")
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
	long_options=['GL_FILE=','BASIN_FILE=','VEL_FILE=','REGION=','DIST=','NUMBER=']
	optlist,arglist = getopt.getopt(sys.argv[1:],'G:B:V:R:D:N:',long_options)

	GL_FILE = os.path.join(pathlib.Path.home(),'GL_learning_data',\
		'6d_results','AllTracks_6d_GL.shp')
	BASIN_FILE = os.path.join(pathlib.Path.home(),'data.dir','basin.dir',\
		'Gates_Basin_v1.7','Basins_v2.4.shp')
	VEL_FILE = os.path.join(pathlib.Path.home(),'data.dir','basin.dir',\
		'ANT_velocity.dir','antarctica_ice_velocity_450m_v2.nc')
	region = 'Getz'
	dist = 10e3
	N = 500
	for opt, arg in optlist:
		if opt in ("-G","--GL_FILE"):
			GL_FILE = os.path.expanduser(arg)
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

	#-- call the function to calculate the grounding zone width
	calc_gz(GL_FILE=GL_FILE,BASIN_FILE=BASIN_FILE,VEL_FILE=VEL_FILE,\
		region=region,dist=dist,N=N)

#-- run main program
if __name__ == '__main__':
	main()