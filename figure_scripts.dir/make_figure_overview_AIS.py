u"""
make_figure_overview_AIS.py
Yara Mohajerani

Figure with all Antarctic GLs
"""
import os
import sys
import numpy as np
import numpy.ma as ma
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import datetime
from PyAstronomy import pyasl
from shapely.geometry import Polygon
from descartes import PolygonPatch
import rasterio
from rasterio.plot import show

base_dir = os.path.expanduser('~')

#-- specify GL files
file_6d = os.path.join(base_dir,'GL_learning_data','6d_results','AllTracks_6d_GL.shp')
file_12d = os.path.join(base_dir,'GL_learning_data','12d_results','AllTracks_12d_GL.shp')
#-- specify error files
file_6d_err = os.path.join(base_dir,'GL_learning_data','6d_results','AllTracks_6d_uncertainty.shp')
file_12d_err = os.path.join(base_dir,'GL_learning_data','12d_results','AllTracks_12d_uncertainty.shp')

#-- set up region of interest for zoomed-in plots
xz,yz = np.zeros((3,2)),np.zeros((3,2))

#-- point on peninsula
xz[0,:] = -2328e3,-2310e3
yz[0,:] = 1249e3,1268e3

#-- point on Getz
# xz[1,:] = -1523e3,-1490e3
# yz[1,:] = -803e3,-760e3
# xz[1,:] = -149e4,-145e4
# yz[1,:] = -835e3,-805e3
xz[1,:] = -149e4,-146e4  #-1340e3,-1320e3
yz[1,:] = -960e3,-933e3  #-1025e3,-1000e3

xz[2,:] = 1022e3,1040e3
yz[2,:] = 1856e3,1873e3

#-- make polygons
poly = {}
for i in range(len(xz)):
	poly[i] = Polygon([(xz[i,0],yz[i,0]),(xz[i,0],yz[i,1]),(xz[i,1],yz[i,1]),(xz[i,1],yz[i,0])])

#-- Set up figure
fig = plt.figure(figsize=(8, 8))
gs = fig.add_gridspec(45, 40)

ax = {}
ax[0] = fig.add_subplot(gs[0:40,8:32])
ax[1] = fig.add_subplot(gs[28:42,28:])
#-- subplots on left
ax[2] = fig.add_subplot(gs[6:21,1:9])
ax[3] = fig.add_subplot(gs[16:31,1:9])
#-- subplots below
ax[4] = fig.add_subplot(gs[27:42,4:13])
ax[5] = fig.add_subplot(gs[27:42,15:24])
#-- subplots on right
ax[6] = fig.add_subplot(gs[6:21,31:40])
ax[7] = fig.add_subplot(gs[16:31,31:40])
#-- colorbar axis
# ax[8] = fig.add_subplot(gs[43,6:16])
ax[8] = fig.add_subplot(gs[40,2:26])
gs.update(wspace=0.0, hspace=0.0)
#-- get colormap so sample from
cmap = cm.get_cmap('viridis')  #('brg')

#-- add background velocity field
vel_file = os.path.join(base_dir,'data.dir','basin.dir','ANT_velocity.dir',\
	'AIS_ice_velocity_magnitude_1km_bilinear.tif')
src = rasterio.open(vel_file,'r')
print(src.nodata)

vel_cmap = plt.get_cmap('pink')
vel_cmap.set_bad(color='white')
vel = src.read()
vel_masked = ma.masked_where(vel <= 0, vel)
vel_img = show(vel_masked, transform=src.transform,cmap=vel_cmap,ax=ax[0],vmin=0,vmax=2000,alpha=0.3)
# fig.colorbar(tmp_img,ax=ax[0],orientation="horizontal",pad=0.02,extend='max',alpha=0.5)
src.close()

#-------------------------------------------------------
#- 1) Plot all GLs
#-------------------------------------------------------
gdf_tot1 = gpd.read_file(file_6d)
gdf_tot2 = gpd.read_file(file_12d)
gdf_err1 = gpd.read_file(file_6d_err)
gdf_err2 = gpd.read_file(file_12d_err)
#-- combine the two dataframes together 
gdf = gpd.GeoDataFrame(pd.concat([gdf_tot1,gdf_tot2], ignore_index=True), crs=gdf_tot1.crs)
gdf_err = gpd.GeoDataFrame(pd.concat([gdf_err1,gdf_err2], ignore_index=True), crs=gdf_err1.crs)
#-- free up memory
gdf_tot1,gdf_tot2 = [],[]
gdf_err1,gdf_err2 = [],[]
for g in range(len(gdf['geometry'])):
	if gdf['geometry'][g].length > 10e3:
		x_all,y_all = gdf['geometry'][g].coords.xy
		ax[0].plot(x_all,y_all,'-k',linewidth=0.5,zorder=1)

#-- plot boxes around zoomed in areas
for i,x_offset,y_offset in zip(range(len(xz)),[0,-3e5,0],[2e5,-3e5,2.5e5]):
	zoom_area = PolygonPatch(poly[i],alpha=0.8,facecolor='red',edgecolor='red',zorder=2)
	ax[0].add_patch(zoom_area)
	ax[0].text(np.mean(xz[i])+x_offset,np.mean(yz[i])+y_offset,i+1,color='red', weight='bold')

#-------------------------------------------------------
#-- add track info
#-------------------------------------------------------
g6d = gpd.read_file(os.path.join(base_dir,'GL_learning_data','2018_Sentinel-1_tracks','6d_PS.shp'))
g12d = gpd.read_file(os.path.join(base_dir,'GL_learning_data','Archive_2018_GL_only','2018_12d_PS_GL_only_for_Yara.shp'))
coast = gpd.read_file(os.path.join(base_dir,'data.dir','basin.dir','Gates_Basin_v1.7','ANT_Basins_IMBIE2_v1.6.shp'))

g6d.plot(ax=ax[1],color='skyblue',alpha=1,zorder=1,edgecolor='gray')
g12d.plot(ax=ax[1],color='goldenrod',alpha=1,zorder=2,edgecolor='gray')
for geom in coast['geometry']:
	if geom.type == 'MultiPolygon':
		for g in geom:
			x,y = g.exterior.coords.xy
			ax[1].plot(x,y,color='black',linewidth=0.5,zorder=3)
	else:
		x,y = geom.exterior.coords.xy
		ax[1].plot(x,y,color='black',linewidth=0.5,zorder=3)
ax[1].scatter([],[],s=20,marker='s',color='skyblue',label='6-Day')
ax[1].scatter([],[],s=20,marker='s',color='goldenrod',label='12-Day')
ax[1].legend(bbox_to_anchor=(0., -0.15, 1., .102), loc='lower left',
           ncol=2, mode="expand", borderaxespad=0.)#,edgecolor='black')
 
#-------------------------------------------------------
#- 2) Plot zoomed-in GZ
#-------------------------------------------------------
#-- initialize dictionaries
tmean = {0: [], 1: [], 2: []}
indices = {0: [], 1: [], 2: []}
lines = {0: [], 1: [], 2: []}
ind_sort = {0: [], 1: [], 2: []}
for i in range(len(gdf)):
	#-- get filename to check date
	f = gdf['FILENAME'][i]
	#-- get dates
	dates = os.path.basename(f).split('_')[2].split('-')
	#-- extract geometry to see if it's in region of interest
	ll = gdf['geometry'][i]
	#-- Loop through the subplots
	for m in range(len(xz)):
		if ll.intersects(poly[m]):
			indices[m].append(i)
			lines[m].append(ll)
			tdec = np.zeros(4)
			for j in range(4):
				#-- convert to datetime format
				dd = datetime.datetime(int(dates[j][:2]),int(dates[j][2:4]),int(dates[j][4:]))
				tdec[j] = pyasl.decimalYear(dd)
			#-- get mean of 4 dates
			tmean[m].append(np.mean(tdec) + 2000)

for m in range(len(xz)):
	#-- convert tmean to numpy array
	tmean[m] = np.array(tmean[m])
	#-- sort dates
	ind_sort[m] = np.argsort(tmean[m])
	for c,i in enumerate(ind_sort[m]):
		if lines[m][i].length > 8e3:
			xs,ys = lines[m][i].coords.xy
			ax[2*(m+1)].plot(xs,ys,linewidth=0.4,alpha=0.8,color=cmap(c/len(ind_sort[m])),zorder=1)

#-- add colorbar for dates
img = ax[4].imshow([tmean[1]], cmap=cmap)
img.set_clim(2018,2019)
cb = plt.colorbar(img,cax=ax[8],orientation="horizontal")#,pad=0.02)#,format='%.1f')

#-------------------------------------------------------
#- 3) Plot uncertanties
#-------------------------------------------------------
#-- initialize dictionaries
err_tmean = {0: [], 1: [], 2: []}
err_indices = {0: [], 1: [], 2: []}
err_lines = {0: [], 1: [], 2: []}
err_ind_sort = {0: [], 1: [], 2: []}
for i in range(len(gdf_err)):
	#-- get filename to check date
	f = gdf_err['FILENAME'][i]
	#-- get dates
	dates = os.path.basename(f).split('_')[2].split('-')
	#-- extract geometry to see if it's in region of interest
	ll = gdf_err['geometry'][i]
	#-- Loop through the subplots
	for m in range(len(xz)):
		if ll.intersects(poly[m]):
			err_indices[m].append(i)
			err_lines[m].append(ll)
			tdec = np.zeros(4)
			for j in range(4):
				#-- convert to datetime format
				dd = datetime.datetime(int(dates[j][:2]),int(dates[j][2:4]),int(dates[j][4:]))
				tdec[j] = pyasl.decimalYear(dd)
			#-- get mean of 4 dates
			err_tmean[m].append(np.mean(tdec) + 2000)

for m,start,mid,end in zip(range(len(xz)),[0,5,0],[2,5,2],[-6,-8,-7]):
	#-- convert tmean to numpy array
	err_tmean[m] = np.array(err_tmean[m])
	#-- sort dates
	err_ind_sort[m] = np.argsort(err_tmean[m])
	for i in [start,int(len(err_ind_sort[m])/2)+mid ,len(err_ind_sort[m])+end]:
		if err_lines[m][i].length > 12e3:
			x,y = err_lines[m][i].coords.xy
			ax[2*(m+1)+1].plot(x,y,color=cmap(i/len(err_ind_sort[m])),linewidth=0.8,alpha=0.8)


#-- set up limits of main plot
x1m,x2m = -3000000,3100000
y1m,y2m = -2600000,2600000

ax[0].set_xlim((x1m,x2m))
ax[0].set_ylim((y1m,y2m))
# #-- add ruler for main plot
ax[0].plot([x2m-5e5,x2m-5e5],[y2m-2e5,y2m-4e5],color='black',linewidth=2.)
ax[0].text(x2m-10e5,y2m-3e5,'200 km',horizontalalignment='center',\
	verticalalignment='center', color='black')

for i in range(len(xz)):
	#-- limit of zoomed in plot
	ax[2*(i+1)].set_xlim(xz[i])
	ax[2*(i+1)].set_ylim(yz[i])
	#-- limit of uncertainty plot
	ax[2*(i+1)+1].set_xlim(xz[i])
	ax[2*(i+1)+1].set_ylim(yz[i])
	if i == 0:
		y_offset = np.abs(yz[i,1]-yz[i,0])/2
		barsize = 1000
	elif i == 1:
		y_offset = 1800
		barsize = 2000
	else:
		y_offset = 1000
		barsize = 1000
	#-- add ruler for zoomed in plots
	ax[2*(i+1)].plot([xz[i,1]-barsize,xz[i,1]-barsize],[yz[i,1]-y_offset,yz[i,1]-y_offset-barsize],color='black',linewidth=2.)
	ax[2*(i+1)].text(xz[i,1]-4*barsize,yz[i,1]-y_offset-barsize/2,'%i km'%(barsize/1000),horizontalalignment='center',\
		verticalalignment='center', color='black')
	#-- number the plot
	ax[2*(i+1)].text(xz[i,0]+barsize,yz[i,1]-2000,i+1,bbox={'edgecolor':'darkgray','facecolor':'lightgray'},color='red',weight='bold')
	#-- ruler for uncertainty plots
	ax[2*(i+1)+1].plot([xz[i,1]-barsize,xz[i,1]-barsize],[yz[i,1]-y_offset,yz[i,1]-y_offset-barsize],color='black',linewidth=2.)
	ax[2*(i+1)+1].text(xz[i,1]-4*barsize,yz[i,1]-y_offset-barsize/2,'%i km'%(barsize/1000),horizontalalignment='center',\
		verticalalignment='center', color='black')
	#-- number the uncertainty plot
	ax[2*(i+1)+1].text(xz[i,0]+barsize,yz[i,1]-2000,i+1,bbox={'edgecolor':'darkgray','facecolor':'lightgray'},color='red',weight='bold')

for i in range(8):
	ax[i].get_xaxis().set_ticks([])
	ax[i].get_yaxis().set_ticks([])
	ax[i].set_aspect('equal')

# ax[0].set_title('a) All Delineations',bbox={'edgecolor':'darkgray','facecolor':'lightgray'})
# ax[1].set_title('b) Tracks',bbox={'edgecolor':'darkgray','facecolor':'lightgray'})
# ax[2].set_title('c) Grounding Zone (1)',x=0.5, y=0.9, bbox={'edgecolor':'darkgray','facecolor':'lightgray'})
fig.subplots_adjust(wspace=0.0, hspace=0.0)
plt.savefig(os.path.join(base_dir,'GL_learning_data','overview_AIS.pdf'),format='PDF',bbox_inches='tight')
plt.close(fig)