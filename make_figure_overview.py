u"""
make_figure_overview
Yara Mohajerani

Figure with the entire Getz and all GL
second panel zooming on an area where the GZ is particularly wide. 
Also make a map of all the errors
"""
import os
import sys
import numpy as np
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
ddir = os.path.join(base_dir,'GL_learning_data','geocoded_v1','stitched.dir',\
	'atrous_32init_drop0.2_customLossR727.dir','shapefiles.dir')
FILTER = 6000.
flt_str = '_%.1fkm'%(FILTER/1000)

fileList = os.listdir(ddir)
gl_list = sorted([f for f in fileList if (f.endswith('%s.shp'%flt_str) and (f.startswith('gl_')))])
er_list = sorted([f for f in fileList if (f.endswith('%s_ERR.shp'%flt_str) and (f.startswith('gl_')))])

print('# of GLs: ', len(gl_list))
print('# of ERs: ', len(er_list))

#-- set up region of interest for zoomed-in plots
x1,x2 = -149e4,-145e4
y1,y2 = -840e3,-805e3
#-- make polygon
poly = Polygon([(x1,y1),(x1,y2),(x2,y2),(x2,y1)])

#-- Set up figure
fig,ax = plt.subplots(1,3,figsize=(10,4))

#-- get Blues colormap so sample from
cmap = cm.get_cmap('brg')

#-- add background velocity field
vel_file = os.path.join(base_dir,'data.dir','basin.dir','ANT_velocity.dir',\
	'AIS_ice_velocity_magnitude_1km_bilinear.tif')
src = rasterio.open(vel_file,'r')
show(src.read(), transform=src.transform,cmap='pink',ax=ax[0],vmin=0,vmax=400,alpha=0.4)
src.close()

#-- plot all grounding lines
tmean = []
lines = []
indices = []
#-------------------------------------------------------
#- 1) Plot all GLs
#-------------------------------------------------------
for i,f in enumerate(gl_list):
	gdf = gpd.read_file(os.path.join(ddir,f))
	gdf.plot(ax=ax[0],color='black',linewidth=0.7)

	#-- get dates
	dates = f.split('_')[2].split('-')

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

#-- plot box around zoomed in area
zoom_area = PolygonPatch(poly,alpha=0.3,facecolor='cyan')
ax[0].add_patch(zoom_area)
#-- get axis limits to plot zoom-in lines
# x_end = ax[0].get_xlim()
# y_end = ax[0].get_ylim()
# y_mean = np.mean([y1,y2])
# y_delta = np.abs(y_end[1]-y_end[0])
# ax[0].plot([x2,x_end[1]],[y2,-624800],color='red',linewidth=0.5)
# ax[0].plot([x2,x_end[1]],[y1,-1.26865e6],color='red',linewidth=0.5)


#-- convert tmean to numpy array
tmean = np.array(tmean)

#-- sort dates
ind_sort = np.argsort(tmean)
#-------------------------------------------------------
#- 2) Plot zoomed-in GZ
#-------------------------------------------------------
for i in ind_sort:
	xs,ys = lines[i].coords.xy
	ax[1].plot(xs,ys,linewidth=0.4,alpha=0.8,color=cmap(i/len(ind_sort)))
#-- add colorbar for dates
img = ax[1].imshow([tmean], cmap=cmap)
img.set_clim(2018,2019)
cb = plt.colorbar(img,ax=ax[1],orientation="horizontal",pad=0.02)#,format='%.1f')
#-------------------------------------------------------
#- 3) Plot uncertanties
#-------------------------------------------------------
for i in [0,int(len(ind_sort)/2),len(ind_sort)-1]:
	file_ind = indices[i]
	gdf = gpd.read_file(os.path.join(ddir,er_list[file_ind]))
	gdf.plot(ax=ax[2],color=cmap(i/len(ind_sort)),linewidth=0.6,alpha=0.8)
#-- add colorbar for dates
cb = plt.colorbar(img,ax=ax[2],orientation="horizontal",pad=0.02)#,format='%.1f')

ax[0].set_xlim((-1678433.1825792969, -908143.1658347661))
ax[0].set_ylim((-1330369.81441327, -504273.8973213372))
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

ax[0].set_title('All Delineations in Getz')
ax[1].set_title('Grounding Zones')
ax[2].set_title('Uncertainty Bars')
fig.subplots_adjust(wspace=0.0, hspace=0.0)
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(ddir,'overview.pdf'),format='PDF')
plt.close(fig)
