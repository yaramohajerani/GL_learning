#!/usr/bin/env python
u"""
plot_pipeline.py
Yara Mohajerani (06/2020)
"""
import os
import sys
import cmocean
import numpy as np
import rasterio
import rasterio.plot as rasplt
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import geopandas as gpd
from shapely.geometry import Polygon
from descartes import PolygonPatch

# fname = 'gl_054_171228-180103-180109-180115_019901-009005-020076-009180_T102441_T102440'
# fname = 'gl_007_180518-180524-180530-180605_021954-011058-022129-011233_T050854_T050855'
fname = 'gl_069_181124-181130-181206-181212_013745-024816-013920-024991_T110456_T110456'

#-- Directory setup
gdrive =  os.path.join(os.path.expanduser('~'),'Google Drive File Stream',\
	'Shared drives','GROUNDING_LINE_TEAM_DRIVE','ML_Yara')
base_dir = os.path.expanduser('~/GL_learning_data/')


#-- Set up figure
fig,ax = plt.subplots(2,2,figsize=(7,8))

#-----------------------------------------------------
#-- read geotif and label shapefile and plot
#-----------------------------------------------------
#-- set file names
tiffile = os.path.join(gdrive,'SOURCE_GEOTIFF','%s.tif'%fname)
shpfile = os.path.join(gdrive,'SOURCE_SHP','%s.shp'%fname)
#--read files
src = rasterio.open(tiffile,'r')
#-- make sure both files have the same projection
gdf = gpd.read_file(shpfile)
gdf = gdf.to_crs(src.crs)
#-- get colormap from the geotif
color_dic = src.colormap(1)
#-- convert to matplotlib colormap with range 0-1
color_list = [None]*len(color_dic)
for i,k in enumerate(color_dic.keys()):
	color_list[i] = np.array(color_dic[k])/255
cmap_tif = ListedColormap(color_list)
#-- plot interferogram
rasplt.show(src.read(1), ax=ax[0,0], transform=src.transform, cmap=cmap_tif)
ax[0,0].set_title('a)', fontsize=14, fontweight='bold', loc='left') #Input & Label
#-- plot GL labels
gdf.plot(ax=ax[0,0], legend=True,linewidth=2,color='white')

#-- figure limits
x1,x2 = -1.5e6,-1.4e6
y1,y2 = -9e5,-10.6e5

#-- add ruler for zoomed in plots
ax[0,0].plot([x1+5000,x1+5000],[y1-5000,y1-10000],color='white',linewidth=2.)
ax[0,0].text(x1+18000,y1-7500,'10 km',horizontalalignment='center',\
	verticalalignment='center', color='white',fontweight='bold')

#-----------------------------------------------------
#-- read output tif and plot
#-----------------------------------------------------
infile = os.path.join(base_dir,'geocoded_v1','stitched.dir',\
	'atrous_32init_drop0.2_customLossR727.dir','%s.tif'%fname)
#--read files
src1 = rasterio.open(infile,'r')
src2 = rasterio.open(infile.replace('.tif','_mask.tif'),'r')
#-- plot stiched prediction file
rasplt.show(src1.read(1), ax=ax[0,1], transform=src.transform, cmap='binary',zorder=1)
rasplt.show(src2.read(1), ax=ax[0,1], transform=src.transform, cmap='binary',zorder=2,alpha=0.3)
ax[0,1].set_title('b)', fontsize=14, fontweight='bold', loc='left') #ML Output & Mask
src1.close()
src2.close()
#-- add ruler for zoomed in plots
ax[0,1].plot([x1+5000,x1+5000],[y1-5000,y1-10000],color='white',linewidth=2.,zorder=10)
ax[0,1].text(x1+18000,y1-7500,'10 km',horizontalalignment='center',\
	verticalalignment='center', color='white',fontweight='bold')

#-----------------------------------------------------
#-- read output shapefiles and plot
#-----------------------------------------------------
infile = os.path.join(base_dir,'geocoded_v1','stitched.dir',\
	'atrous_32init_drop0.2_customLossR727.dir','shapefiles.dir','%s_6.0km.shp'%fname)
gdf1 = gpd.read_file(infile)
gdf2 = gpd.read_file(infile.replace('.shp','_ERR.shp'))

#- 1) complete scene
gdf1.plot(ax=ax[1,0],linewidth=2,color='black',zorder=2)
# gdf2.plot(ax=ax[1,0],linewidth=1,linestyle='--',color='gray',zorder=2)
#-- plot interferogram
rasplt.show(src.read(1), ax=ax[1,0], transform=src.transform, cmap=cmap_tif,zorder=1)

#-- add ruler for zoomed in plots
ax[1,0].plot([x1+5000,x1+5000],[y1-5000,y1-10000],color='white',linewidth=2.)
ax[1,0].text(x1+18000,y1-7500,'10 km',horizontalalignment='center',\
	verticalalignment='center', color='white',fontweight='bold')

#- 2) zoomed-in scene
gdf1.plot(ax=ax[1,1],linewidth=2.5,color='black',zorder=4)
gdf2.plot(ax=ax[1,1],linewidth=2.,linestyle='--',color='black',zorder=3)
#-- also plot hand-drawn labels for zoomed in panel
gdf.plot(ax=ax[1,1],linewidth=2.5,color='white',zorder=2)
#-- plot interferogram
rasplt.show(src.read(1), ax=ax[1,1], transform=src.transform, cmap=cmap_tif,zorder=1)

#-- limits of zoomed-in area
x1z,x2z = -1.437e6,-1.4239e6
y1z,y2z = -957.5e3,-977e3
ax[1,1].plot([x2z-2600,x2z-2600],[y2z+1000,y2z+2000],color='white',linewidth=2.)
ax[1,1].text(x2z-1200,y2z+1500,'1 km',horizontalalignment='center',\
	verticalalignment='center', color='white',fontweight='bold')

#-- make polygon for zoomed-in area
poly = Polygon([(x1z,y2z),(x1z,y1z),(x2z,y1z),(x2z,y2z)])
#-- plot box around zoomed in area
# zoom_area = PolygonPatch(poly,alpha=0.4,facecolor=None,edgecolor='gray', linewidth=2,zorder=10)
# ax[1,0].add_patch(zoom_area)
xedge, yedge = poly.exterior.xy
ax[1,0].plot(xedge,yedge,color='maroon',linewidth=2)#,linestyle='--')

#-- add legend to zoomed in plot
ax[1,1].plot([],[],color='black',linewidth=1.5,label='ML GL')
ax[1,1].plot([],[],color='black',linewidth=1,linestyle='--',label='ML Uncertainty')
ax[1,1].plot([],[],color='white',linewidth=1.5,label='Manual GL')
ax[1,1].legend(loc='lower left',facecolor='silver')

ax[1,0].set_title('c)', fontsize=14, fontweight='bold', loc='left') #Vectorized GL & Uncertainty
ax[1,1].set_title('d)', fontsize=14, fontweight='bold', loc='left') #Zoomed-in Comparison

src.close()

#-- save figure to file
for i in range(2):
	for j in range(2):
		ax[i,j].get_xaxis().set_visible(False)
		ax[i,j].get_yaxis().set_visible(False)
		# ax[i,j].set_aspect('equal')
		if i == 1 and j == 1:
			ax[i,j].set_xlim([x1z,x2z])
			ax[i,j].set_ylim([y2z,y1z])
		else:
			ax[i,j].set_xlim([x1,x2])
			ax[i,j].set_ylim([y2,y1])
plt.tight_layout()
plt.savefig(os.path.join(base_dir,'geocoded_v1','stitched.dir',\
	'atrous_32init_drop0.2_customLossR727.dir','Pipeline_Figure.pdf'),format='PDF')
plt.close(fig)