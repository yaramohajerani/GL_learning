u"""
make_comparison_figure.py
Yara Mohajerani
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import rasterio
import rasterio.plot as rasplt
import geopandas as gpd

#-- Directory setup
gdrive =  os.path.join(os.path.expanduser('~'),'Google Drive File Stream',\
	'Shared drives','GROUNDING_LINE_TEAM_DRIVE','ML_Yara')
base_dir = os.path.expanduser('~/GL_learning_data/')

#*****************
#-- first figure
#*****************
fig = plt.figure(1,figsize=(4,4))
ax = fig.add_subplot(111)
fname = 'gl_069_180808-180814-180820-180826_012170-023241-012345-023416_T110454_T110455'
#---------------------
#-- read interferogram
#---------------------
src = rasterio.open(os.path.join(gdrive,'SOURCE_GEOTIFF','%s.tif'%fname),'r')
#-- get colormap from the geotif
color_dic = src.colormap(1)
#-- convert to matplotlib colormap with range 0-1
color_list = [None]*len(color_dic)
for i,k in enumerate(color_dic.keys()):
	color_list[i] = np.array(color_dic[k])/255
cmap_tif = ListedColormap(color_list)
#-- plot
rasplt.show(src.read(1), ax=ax, transform=src.transform, cmap=cmap_tif)
#---------------------
#-- read NN GL
#---------------------
gdf = gpd.read_file(os.path.join(base_dir,'geocoded_v1','stitched.dir',\
	'atrous_32init_drop0.2_customLossR727.dir','shapefiles.dir','%s_6.0km.shp'%fname))
gdf = gdf.to_crs(src.crs)
gdf.plot(ax=ax, legend=True,linewidth=2,color='indigo')
#---------------------
#-- read manual GL
#---------------------
gdf = gpd.read_file(os.path.join(gdrive,'SOURCE_SHP','%s.shp'%fname))
gdf = gdf.to_crs(src.crs)
gdf.plot(ax=ax, legend=True,linewidth=2,color='white')

x1,x2 = -1.487e6,-1.476e6
y1,y2 = -9.49e5,-9.39e5
ax.plot([x2-1500,x2-1500],[y2-200,y2-1200],color='black',linewidth=2.)
ax.text(x2-700,y2-700,'1 km',horizontalalignment='center',\
	verticalalignment='center', color='black',fontweight='bold')

ax.set_xlim([x1,x2])
ax.set_ylim([y1,y2])

ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(base_dir,'geocoded_v1','stitched.dir',\
	'atrous_32init_drop0.2_customLossR727.dir','shapefiles.dir',\
		'%s_6.0km_comparison.pdf'%fname),format='PDF')
plt.close(fig)


#*****************
#-- second figure
#*****************
fig = plt.figure(2,figsize=(10,6))
fname = 'gl_069_181124-181130-181206-181212_013745-024816-013920-024991_T110456_T110456'
ax = fig.add_subplot(111)
#---------------------
#-- read interferogram
#---------------------
src = rasterio.open(os.path.join(gdrive,'SOURCE_GEOTIFF','%s.tif'%fname),'r')
#-- get colormap from the geotif
color_dic = src.colormap(1)
#-- convert to matplotlib colormap with range 0-1
color_list = [None]*len(color_dic)
for i,k in enumerate(color_dic.keys()):
	color_list[i] = np.array(color_dic[k])/255
cmap_tif = ListedColormap(color_list)
#-- plot
rasplt.show(src.read(1), ax=ax, transform=src.transform, cmap=cmap_tif)
#---------------------
#-- read NN GL
#---------------------
gdf = gpd.read_file(os.path.join(base_dir,'geocoded_v1','stitched.dir',\
	'atrous_32init_drop0.2_customLossR727.dir','shapefiles.dir','%s_6.0km.shp'%fname))
gdf = gdf.to_crs(src.crs)
gdf.plot(ax=ax, legend=True,linewidth=2,color='indigo')
#---------------------
#-- read manual GL
#---------------------
gdf = gpd.read_file(os.path.join(gdrive,'SOURCE_SHP','%s.shp'%fname))
gdf = gdf.to_crs(src.crs)
gdf.plot(ax=ax, legend=True,linewidth=2,color='white')

x1,x2 = -1.463e6,-1.413e6
y1,y2 = -1.016e6,-9.82e5
ax.set_xlim([x1,x2])
ax.set_ylim([y1,y2])

# ax.plot([x2-6000,x2-6000],[y2-500,y2-1500],color='black',linewidth=2.)
# ax.text(x2-4500,y2-1000,'1 km',horizontalalignment='center',\
# 	verticalalignment='center', color='black',fontweight='bold')
ax.plot([x2-3000,x2-3000],[y1+500,y1+1500],color='black',linewidth=2.)
ax.text(x2-1500,y1+1000,'1 km',horizontalalignment='center',\
	verticalalignment='center', color='black',fontweight='bold')

ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(base_dir,'geocoded_v1','stitched.dir',\
	'atrous_32init_drop0.2_customLossR727.dir','shapefiles.dir',\
		'%s_6.0km_comparison.pdf'%fname),format='PDF')
plt.close(fig)