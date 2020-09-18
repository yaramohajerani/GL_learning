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
fig = plt.figure(figsize=(8, 8))
gs = fig.add_gridspec(2, 3)

ax = {}
ax[0] = fig.add_subplot(gs[0,0])
ax[1] = fig.add_subplot(gs[0,1])
ax[2] = fig.add_subplot(gs[0,2])
ax[3] = fig.add_subplot(gs[1,0])
ax[4] = fig.add_subplot(gs[1,1:])
gs.update(wspace=0.0, hspace=0.1 )

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
rasplt.show(src.read(1), ax=ax[0], transform=src.transform, cmap=cmap_tif)
ax[0].set_title('a)', fontsize=14, fontweight='bold', loc='left') #Input & Label
#-- plot GL labels
gdf.plot(ax=ax[0], legend=True,linewidth=2,color='white')

#-- figure limits
x1,x2 = -1.5e6,-1.4e6
y1,y2 = -9e5,-10.6e5

#-- add ruler for zoomed in plots
ax[0].plot([x1+5000,x1+5000],[y1-5000,y1-10000],color='white',linewidth=2.)
ax[0].text(x1+20000,y1-7500,'10 km',horizontalalignment='center',\
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
rasplt.show(src1.read(1), ax=ax[1], transform=src.transform, cmap='binary',zorder=1)
rasplt.show(src2.read(1), ax=ax[1], transform=src.transform, cmap='binary',zorder=2,alpha=0.3)
ax[1].set_title('b)', fontsize=14, fontweight='bold', loc='left') #ML Output & Mask
src1.close()
src2.close()
#-- add ruler for zoomed in plots
ax[1].plot([x1+5000,x1+5000],[y1-5000,y1-10000],color='white',linewidth=2.,zorder=10)
ax[1].text(x1+20000,y1-7500,'10 km',horizontalalignment='center',\
	verticalalignment='center', color='white',fontweight='bold')

#-----------------------------------------------------
#-- read output shapefiles and plot
#-----------------------------------------------------
infile = os.path.join(base_dir,'geocoded_v1','stitched.dir',\
	'atrous_32init_drop0.2_customLossR727.dir','shapefiles.dir','%s_6.0km.shp'%fname)
gdf1 = gpd.read_file(infile)
gdf2 = gpd.read_file(infile.replace('.shp','_ERR.shp'))

#- 1) complete scene
gdf1.plot(ax=ax[2],linewidth=2,color='black',zorder=2)
# gdf2.plot(ax=ax[2],linewidth=1,linestyle='--',color='gray',zorder=2)
#-- plot interferogram
rasplt.show(src.read(1), ax=ax[2], transform=src.transform, cmap=cmap_tif,zorder=1)

#-- add ruler for zoomed in plots
ax[2].plot([x1+5000,x1+5000],[y1-5000,y1-10000],color='white',linewidth=2.)
ax[2].text(x1+20000,y1-7500,'10 km',horizontalalignment='center',\
	verticalalignment='center', color='white',fontweight='bold')

#- 2) zoomed-in scene
gdf1.plot(ax=ax[3],linewidth=2.5,color='black',zorder=4)
gdf2.plot(ax=ax[3],linewidth=2.,linestyle='--',color='black',zorder=3)
#-- also plot hand-drawn labels for zoomed in panel
gdf.plot(ax=ax[3],linewidth=2.5,color='white',zorder=2)
#-- plot interferogram
rasplt.show(src.read(1), ax=ax[3], transform=src.transform, cmap=cmap_tif,zorder=1)

#-- limits of zoomed-in area
x1z,x2z = -1.437e6,-1.4239e6
y1z,y2z = -957.5e3,-977e3
ax[3].plot([x2z-3000,x2z-3000],[y2z+1000,y2z+2000],color='white',linewidth=2.)
ax[3].text(x2z-1500,y2z+1500,'1 km',horizontalalignment='center',\
	verticalalignment='center', color='white',fontweight='bold')

#-- make polygon for zoomed-in area
poly = Polygon([(x1z,y2z),(x1z,y1z),(x2z,y1z),(x2z,y2z)])
#-- plot box around zoomed in area
# zoom_area = PolygonPatch(poly,alpha=0.4,facecolor=None,edgecolor='gray', linewidth=2,zorder=10)
# ax[2].add_patch(zoom_area)
xedge, yedge = poly.exterior.xy
ax[2].plot(xedge,yedge,color='maroon',linewidth=2)#,linestyle='--')


ax[2].set_title('c)', fontsize=14, fontweight='bold', loc='left') #Vectorized GL & Uncertainty
ax[3].set_title('d)', fontsize=14, fontweight='bold', loc='left') #Zoomed-in Comparison

src.close()

#--------------------------------------------------------------------------------------------
#-- add comparison panel
#--------------------------------------------------------------------------------------------
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
rasplt.show(src.read(1), ax=ax[4], transform=src.transform, cmap=cmap_tif)
#---------------------
#-- plot NN GL
#---------------------
gdf1 = gdf1.to_crs(src.crs)
gdf1.plot(ax=ax[4], legend=True,linewidth=2,color='black')
#---------------------
#-- read manual GL
#---------------------
gdf = gpd.read_file(os.path.join(gdrive,'SOURCE_SHP','%s.shp'%fname))
gdf = gdf.to_crs(src.crs)
gdf.plot(ax=ax[4], legend=True,linewidth=2,color='white')


x1c,x2c = -1.463e6,-1.413e6
y1c,y2c = -1.016e6,-9.82e5

ax[4].plot([x2c-18e3,x2c-18e3],[y1c+700,y1c+1700],color='white',linewidth=2.)
ax[4].text(x2c-15e3,y1c+1200,'1 km',horizontalalignment='center',\
	verticalalignment='center', color='white',fontweight='bold')
ax[4].set_title('e)', fontsize=14, fontweight='bold', loc='left')

src.close()

#-- plot box highlighting panel (e)
#-- make polygon for zoomed-in area
poly = Polygon([(x1c,y2c),(x1c,y1c),(x2c,y1c),(x2c,y2c)])
xedge, yedge = poly.exterior.xy
ax[2].plot(xedge,yedge,color='blue',linewidth=2)

#-- plot boxes for comparison highlights
x1b,x2b = -1.458e6,-1.452e6
y1b,y2b = -997293,-994430
#-- make polygon for zoomed-in area
poly = Polygon([(x1b,y2b),(x1b,y1b),(x2b,y1b),(x2b,y2b)])
xedge, yedge = poly.exterior.xy
ax[4].plot(xedge,yedge,color='gold',linewidth=2)

x1b,x2b = -1.4165e6,-1.41336e6
y1b,y2b = -990331,-986218
#-- make polygon for zoomed-in area
poly = Polygon([(x1b,y2b),(x1b,y1b),(x2b,y1b),(x2b,y2b)])
xedge, yedge = poly.exterior.xy
ax[4].plot(xedge,yedge,color='gold',linewidth=2)

x1b,x2b = -1.4156e6,-1.41312e6
y1b,y2b = -999645,-995532
#-- make polygon for zoomed-in area
poly = Polygon([(x1b,y2b),(x1b,y1b),(x2b,y1b),(x2b,y2b)])
xedge, yedge = poly.exterior.xy
ax[4].plot(xedge,yedge,color='dodgerblue',linewidth=2)

x1b,x2b = -1.4627e6,-1.46066e6
y1b,y2b = -987548,-984887
#-- make polygon for zoomed-in area
poly = Polygon([(x1b,y2b),(x1b,y1b),(x2b,y1b),(x2b,y2b)])
xedge, yedge = poly.exterior.xy
ax[4].plot(xedge,yedge,color='dodgerblue',linewidth=2)

#-- add legend to zoomed in plot
ax[4].plot([],[],color='black',linewidth=1.5,label='ML GL')
ax[4].plot([],[],color='black',linewidth=1,linestyle='--',label='ML Uncertainty')
ax[4].plot([],[],color='white',linewidth=1.5,label='Manual GL')
ax[4].plot([],[],color='gold',linewidth=1.5,label='ML-only')
ax[4].plot([],[],color='dodgerblue',linewidth=1.5,label='Manual-only')
ax[4].legend(loc='lower center',facecolor='silver')


#-- format axes
for i in range(5):
	ax[i].get_xaxis().set_visible(False)
	ax[i].get_yaxis().set_visible(False)
	# ax[i,j].set_aspect('equal')
	if i == 3:
		ax[i].set_xlim([x1z,x2z])
		ax[i].set_ylim([y2z,y1z])
	elif i == 4:
		ax[i].set_xlim([x1c,x2c])
		ax[i].set_ylim([y2c,y1c])
	else:
		ax[i].set_xlim([x1,x2])
		ax[i].set_ylim([y2,y1])
# plt.tight_layout()
# plt.show()
#-- save figure to file
# plt.savefig(os.path.join(base_dir,'geocoded_v1','stitched.dir',\
# 	'atrous_32init_drop0.2_customLossR727.dir','Pipeline_Figure.pdf'),format='PDF')
plt.savefig(os.path.join(base_dir,'Pipeline_Figure.pdf'),format='PDF')
plt.close(fig)