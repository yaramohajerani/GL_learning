#!/usr/bin/env python
u"""
hydrostatic_gz.py
by Yara Mohajerani (01/2021)

Calculate GZ width based on Hydrostatic Equilibrium using 
TSAI and GUDMUNDSSON (2015), doi: 10.3189/2015JoG14J152
"""
import os
import pathlib
import numpy as np
import pandas as pd
import geopandas as gpd
import netCDF4 as nc
from copy import copy
import rasterio as rio
from shapely.geometry import Point,MultiPoint,LineString
from descartes import PolygonPatch
import matplotlib.pyplot as plt
from rasterio.plot import show

#-- constants
Rho_i = 917.
Rho_w = 1028.
rho = Rho_i/Rho_w
dist = 4e3

GL_FILE = os.path.join(pathlib.Path.home(),'GL_learning_data',\
	'6d_results','AllTracks_6d_GL.shp')

VEL_FILE = os.path.join(pathlib.Path.home(),'data.dir','basin.dir',\
	'ANT_velocity.dir','v_mix.v8Jul2019.nc')

VEL_PLOT_FILE = os.path.join(pathlib.Path.home(),'data.dir','basin.dir',\
	'ANT_velocity.dir','AIS_ice_velocity_magnitude_1km_bilinear.tif')

BEDMACHINE_FILE = os.path.join(pathlib.Path.home(),'data.dir',\
	'elevation_models.dir','BedMachineAntarctica-2020-12-15.nc')

REF_FILE = os.path.join(pathlib.Path.home(),'GL_learning_data',\
	'6d_results','GZ_widths-hybrid_Getz.csv')

BASIN_FILE = os.path.join(pathlib.Path.home(),'data.dir','basin.dir',\
	'Gates_Basin_v1.7','Basins_v2.4.shp')
region = 'Getz'

#-- read the delineated GZ
df = pd.read_csv(REF_FILE)

#-- read velocity field
vel_fid = nc.Dataset(VEL_FILE,'r')
xvel = vel_fid['x'][:]
yvel = vel_fid['y'][:]
vx = vel_fid['VX'][:]
vy = vel_fid['VY'][:]
vel_fid.close()

#-- read bedmachine
fid = nc.Dataset(BEDMACHINE_FILE,'r')
x = fid['x'][:]
y = fid['y'][:]
bed = fid['bed'][:]
surf = fid['surface'][:]
fid.close()

dx = x[1]-x[0]
dy = y[1]-y[0]

#-- load region polygon
basins = gpd.read_file(BASIN_FILE)
idx = basins.index[basins['NAME']==region]
#-- get polygon
poly = basins['geometry'][idx[0]]
#-- add a 5km buffer to find the corresponding GLs
region_poly = poly.buffer(5e3)

#-- read delineated lines
#-- read the grounding lines
df_gl = gpd.read_file(GL_FILE)
lines = []
for i in range(len(df_gl)):
	#-- extract geometry to see if it's in region of interest
	ll = df_gl['geometry'][i]
	if ll.intersects(region_poly):
		lines.append(ll)

#-- make plot
fig = plt.figure(1,figsize=(10,8))
ax = fig.add_subplot(111)
#-- plot velocity field
src = rio.open(VEL_PLOT_FILE)
show(src.read(), transform=src.transform,cmap='Blues',vmin=0,vmax=600,ax=ax,zorder=0,alpha=0.8)
_ = ax.imshow(np.arange(400).reshape(20,20), cmap='Blues',vmin=0,vmax=600)
cbar = fig.colorbar(_, ax=ax)
cbar.ax.get_yaxis().labelpad = 20
cbar.ax.set_ylabel('Ice Velocity (m/yr)', rotation=270,fontweight='bold')
pp = PolygonPatch(poly,alpha=0.2,fc='lawngreen',ec='lawngreen',zorder=1)
ax.add_patch(pp)
for il in lines:
	xs,ys = il.coords.xy
	ax.plot(xs,ys,linewidth=0.4,alpha=0.8,color='k',zorder=2)

#-- initialize output field
fid = open(REF_FILE.replace('.csv','_comparison.csv'),'w')
fid.write('X,Y,ML_WIDTH(m),HE_WIDTH(m),ML/HE ratio\n')
#-- go through point, extract corresponding expectation, and save to file
count = 0
#-- initialize list of points for keeping track of distance for plotting
plot_pts = [Point([0,0])]
for i in range(len(df)):
	if df['date1'][i] != 'None' and df['date2'][i] != 'None':
		#-- extract delineated parameters
		xr = df['X (m)'][i]
		yr = df['Y (m)'][i]
		#-- Get widths
		wi_ml  = df['width (km)'][i]*1e3
		#-- get the closest point in the velocity field
		ii = np.argmin(np.abs(xr-x))
		jj = np.argmin(np.abs(yr-y))
		#-- now we want to get the slope at this point along the velocity direction
		#-- get the closest velocity coordinate
		iv = np.argmin(np.abs(xr - xvel))
		jv = np.argmin(np.abs(yr - yvel))
		#-- find flow angle
		ang = np.arctan(vy[jv,iv]/vx[jv,iv])
		xtrans,ytrans = dist*np.cos(ang),dist*np.sin(ang)
		#-- get the number of pixels in each direction
		nx = int(xtrans/dx)
		ny = int(ytrans/dy)
		#-- alpha should be positive in the downstream direction
		if surf[jj,ii] > surf[jj-ny,ii-nx]:
			alpha1 = (surf[jj,ii]-surf[jj-ny,ii-nx])/np.sqrt((nx*dx)**2 + (ny*dy)**2)
			beta1 = (bed[jj,ii]-bed[jj-ny,ii-nx])/np.sqrt((nx*dx)**2 + (ny*dy)**2)
			transect1 = LineString([[x[ii],y[jj]],[x[ii-nx],y[jj-ny]]])
		else:
			alpha1 = (surf[jj,ii]-surf[jj+ny,ii+nx])/np.sqrt((nx*dx)**2 + (ny*dy)**2)
			beta1 = (bed[jj,ii]-bed[jj+ny,ii+nx])/np.sqrt((nx*dx)**2 + (ny*dy)**2)
			transect1 = LineString([[x[ii],y[jj]],[x[ii+nx],y[jj+ny]]])
		gamma1 = beta1 + rho*(alpha1-beta1)
		#-- also test gamma in the other direction
		if surf[jj+ny,ii+nx] > surf[jj,ii]:
			alpha2 = (surf[jj+ny,ii+nx]-surf[jj,ii])/np.sqrt((nx*dx)**2 + (ny*dy)**2)
			beta2 = (bed[jj+ny,ii+nx]-bed[jj,ii])/np.sqrt((nx*dx)**2 + (ny*dy)**2)
			transect2 = LineString([[x[ii+nx],y[jj+ny]],[x[ii],y[jj]]])
		else:
			alpha2 = (surf[jj-ny,ii-nx]-surf[jj,ii])/np.sqrt((nx*dx)**2 + (ny*dy)**2)
			beta2 = (bed[jj-ny,ii-nx]-bed[jj,ii])/np.sqrt((nx*dx)**2 + (ny*dy)**2)
			transect2 = LineString([[x[ii-nx],y[jj-ny]],[x[ii],y[jj]]])
		gamma2 = beta2 + rho*(alpha2-beta2)
		#-- take the smallest positive gamma
		if gamma1 > 0 and gamma2 < 0:
			gamma = copy(gamma1)
			transect = copy(transect1)
		elif gamma2 > 0 and gamma1 < 0:
			gamma = copy(gamma2)
			transect = copy(transect2)
		else:
			gii = np.argmin([gamma1,gamma2])
			gamma = [gamma1,gamma2][gii]
			transect = [transect1,transect2][gii]

		#-- get the width for a 2m tide
		wi_he = 2/gamma
		#-- write to file
		fid.write('{0:.6f},{1:.6f},{2:.3f},{3:.3f},{4:.3f}\n'.\
			format(xr,yr,wi_ml,wi_he,wi_ml/wi_he))

		if count < 14 and wi_he >= 100 and int(round(wi_ml)) not in [1584,2894]:
			pt = Point(xr,yr)
			if (pt.distance(MultiPoint(plot_pts)) > 20e3):
				plot_pts.append(copy(pt))
				#-- Now plot the transect for the given index
				lx,ly = transect.coords.xy
				ax.plot(lx,ly,linewidth=2.0,alpha=1.0,color='red',zorder=3)
				ax.scatter(xr,yr,s=8,color='darkorchid',zorder=4,alpha=0.5)
				yoffset = -3e3 if (int(round(wi_ml)) == 7192) else 3e3
				xoffset = -1.3e5 if (int(round(wi_ml)) in [15282,4283]) else 8e3
				ax.text(xr+xoffset,yr+yoffset,'ML:{0:d}m; HE:{1:d}m'.format(int(round(wi_ml)),int(round(wi_he))),\
					color='darkred',fontsize=10,fontweight='bold',bbox=dict(facecolor='mistyrose', alpha=0.5))
				count += 1

fid.close()

ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
minx,miny,maxx,maxy = poly.bounds
ax.set_xlim([minx,maxx])
ax.set_ylim([miny,maxy])
ax.set_title("Grounding Zone Width Comparison on {0}".format(region))
plt.tight_layout()
plt.savefig(REF_FILE.replace('.csv','_comparison.pdf'),format='PDF')
plt.close(fig)
		

