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
import geopandas as gpd
import datetime
from PyAstronomy import pyasl
from shapely.geometry import Polygon
from descartes import PolygonPatch
import rasterio
from rasterio.plot import show

base_dir = os.path.expanduser('~')

infile = os.path.join(base_dir,'GL_learning_data','combined_AllTracks_cleaned_15km.shp')


ddir1 = os.path.join(base_dir,'GL_learning_data','geocoded_v1','stitched.dir',\
	'atrous_32init_drop0.2_customLossR727.dir','shapefiles.dir')
flt_str1 = '_6.0km'

ddir2 = os.path.join(base_dir,'GL_learning_data','S1_Pope-Smith-Kohler','UNUSED',\
	'coco_PSK-UNUSED_with_null','atrous_32init_drop0.2_customLossR727.dir','stitched.dir','shapefiles.dir')
flt_str2 = '_8.0km'

fileList = os.listdir(ddir1)
gl_list1 = [os.path.join(ddir1,f) for f in fileList if (f.endswith('%s.shp'%flt_str1) and (f.startswith('gl_')))]
er_list1 = [os.path.join(ddir1,f) for f in fileList if (f.endswith('%s_ERR.shp'%flt_str1) and (f.startswith('gl_')))]

print('# in ddir1: ',len(gl_list1))

fileList = os.listdir(ddir2)
gl_list2 = [os.path.join(ddir2,f) for f in fileList if (f.endswith('%s.shp'%flt_str2) and (f.startswith('gl_')))]
er_list2 = [os.path.join(ddir2,f) for f in fileList if (f.endswith('%s_ERR.shp'%flt_str2) and (f.startswith('gl_')))]

gl_list = sorted(gl_list1 + gl_list2)
er_list = sorted(er_list1 + er_list2)

print('# in ddir2: ',len(gl_list2))

print('# of GLs: ', len(gl_list))
print('# of ERs: ', len(er_list))

#-- set up region of interest for zoomed-in plots
x1,x2 = -149e4,-145e4
y1,y2 = -835e3,-805e3
#-- make polygon
poly = Polygon([(x1,y1),(x1,y2),(x2,y2),(x2,y1)])

#-- Set up figure
fig = plt.figure(figsize=(8, 8))
gs = fig.add_gridspec(2, 2)

ax = {}
ax[0] = fig.add_subplot(gs[0, :])
ax[1] = fig.add_subplot(gs[1,0])
ax[2] = fig.add_subplot(gs[1,1])
#-- get Blues colormap so sample from
cmap = cm.get_cmap('brg')

#-- add background velocity field
vel_file = os.path.join(base_dir,'data.dir','basin.dir','ANT_velocity.dir',\
	'AIS_ice_velocity_magnitude_1km_bilinear.tif')
src = rasterio.open(vel_file,'r')
print(src.nodata)

vel_cmap = plt.get_cmap('pink')
vel_cmap.set_bad(color='white')
vel = src.read()
vel_masked = ma.masked_where(vel <= 0, vel)
# tmp_img = ax[0].imshow(np.arange(4).reshape(2,2), cmap=vel_cmap, vmin=0, vmax=1000,alpha=0.5)
vel_img = show(vel_masked, transform=src.transform,cmap=vel_cmap,ax=ax[0],vmin=0,vmax=2000,alpha=0.3)
# fig.colorbar(tmp_img,ax=ax[0],orientation="horizontal",pad=0.02,extend='max',alpha=0.5)
src.close()

#-- plot all grounding lines
tmean = []
lines = []
indices = []
#-------------------------------------------------------
#- 1) Plot all GLs
#-------------------------------------------------------
gdf_tot = gpd.read_file(infile)
for g in range(len(gdf_tot['geometry'])):
		if gdf_tot['geometry'][g].length > 10e3:
			x,y = gdf_tot['geometry'][g].coords.xy
			ax[0].plot(x,y,'-k',linewidth=0.5,zorder=2)
#-- plot box around zoomed in area
zoom_area = PolygonPatch(poly,alpha=0.7,facecolor='cyan',zorder=1)
ax[0].add_patch(zoom_area)


for i,f in enumerate(gl_list):
	gdf = gpd.read_file(f)
	#-- get dates
	dates = os.path.basename(f).split('_')[2].split('-')

	#-- extract geometry to see if it's in region of interest
	for k in range(len(gdf['geometry'])):
		ll = gdf['geometry'][k]
		if ll.intersects(poly):
			indices.append(i)
			lines.append(ll)
			tdec = np.zeros(4)
			for j in range(4):
				#-- convert to datetime format
				dd = datetime.datetime(int(dates[j][:2]),int(dates[j][2:4]),int(dates[j][4:]))
				tdec[j] = pyasl.decimalYear(dd)
			#-- get mean of 4 dates
			tmean.append(np.mean(tdec) + 2000)


#-- convert tmean to numpy array
tmean = np.array(tmean)

#-- sort dates
ind_sort = np.argsort(tmean)
#-------------------------------------------------------
#- 2) Plot zoomed-in GZ
#-------------------------------------------------------
for c,i in enumerate(ind_sort):
	if lines[i].length > 8e3:
		xs,ys = lines[i].coords.xy
		ax[1].plot(xs,ys,linewidth=0.4,alpha=0.8,color=cmap(c/len(ind_sort)),zorder=1)
#-- add colorbar for dates
img = ax[1].imshow([tmean], cmap=cmap)
img.set_clim(2018,2019)

#-------------------------------------------------------
#- 3) Plot uncertanties
#-------------------------------------------------------
for i in [0,int(len(ind_sort)/2)+2 ,len(ind_sort)-7]:
	idx = ind_sort[i]
	file_ind = indices[idx]
	gdf = gpd.read_file(er_list[file_ind])
	for g in range(len(gdf['geometry'])):
		if gdf['geometry'][g].length > 12e3:
			x,y = gdf['geometry'][g].coords.xy
			ax[2].plot(x,y,color=cmap(i/len(ind_sort)),linewidth=0.8,alpha=0.8)
#-- add colorbar for dates
fig.subplots_adjust(bottom=-0.2)
cbar_ax = fig.add_axes([0.2, 0.07, 0.6, 0.02])
cb = plt.colorbar(img,cax=cbar_ax,orientation="horizontal")#,pad=0.02)#,format='%.1f')


#-- set up limits of main plot
x1m,x2m = -3000000,3000000
y1m,y2m = -2600000,2600000

ax[0].set_xlim((x1m,x2m))
ax[0].set_ylim((y1m,y2m))
# #-- add ruler for main plot
ax[0].plot([x2m-1.5e5,x2m-1.5e5],[y2m-2e5,y2m-4e5],color='black',linewidth=2.)
ax[0].text(x2m-10e5,y2m-3e5,'200 km',horizontalalignment='center',\
	verticalalignment='center', color='black')

for i in [1,2]:
	ax[i].set_xlim([x1,x2])
	ax[i].set_ylim([y1,y2])
	#-- add ruler for zoomed in plots
	ax[i].plot([x2-1000,x2-1000],[y2-1000,y2-2000],color='black',linewidth=2.)
	ax[i].text(x2-5000,y2-1500,'1 km',horizontalalignment='center',\
		verticalalignment='center', color='black')

for i in range(3):
	ax[i].get_xaxis().set_ticks([])
	ax[i].get_yaxis().set_ticks([])
	ax[i].set_aspect('equal')

ax[0].set_title('a) All Delineations')
ax[1].set_title('b) Grounding Zone',x=0.5, y=0.9)
ax[2].set_title('c) Uncertainty Bars',x=0.5, y=0.9)
fig.subplots_adjust(wspace=0.0, hspace=0.0)
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(base_dir,'GL_learning_data','overview_raw_AIS.pdf'),format='PDF')
plt.close(fig)