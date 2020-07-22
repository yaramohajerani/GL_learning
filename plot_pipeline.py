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
ax[0,0].set_title('Input & Label', fontsize=14, fontweight='bold')
#-- plot GL labels
gdf.plot(ax=ax[0,0], legend=True,linewidth=2,color='black')

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
ax[0,1].set_title('NN Output & Mask', fontsize=14, fontweight='bold')
src1.close()
src2.close()

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

#- 2) zoomed-in scene
gdf1.plot(ax=ax[1,1],linewidth=2.5,color='indigo',zorder=4)
gdf2.plot(ax=ax[1,1],linewidth=2.,linestyle='--',color='indigo',zorder=3)
#-- also plot hand-drawn labels for zoomed in panel
gdf.plot(ax=ax[1,1],linewidth=2.5,color='white',zorder=2)
#-- plot interferogram
rasplt.show(src.read(1), ax=ax[1,1], transform=src.transform, cmap=cmap_tif,zorder=1)



#-- add legend to zoomed in plot
ax[1,1].plot([],[],color='indigo',linewidth=1.5,label='NN GL')
ax[1,1].plot([],[],color='indigo',linewidth=1,linestyle='--',label='NN Uncertainty')
ax[1,1].plot([],[],color='white',linewidth=1.5,label='Manual GL')
ax[1,1].legend(loc='lower left',facecolor='silver')

ax[1,0].set_title('Vectorized GL & Uncertainty', fontsize=14, fontweight='bold')
ax[1,1].set_title('Zoomed-in Comparison', fontsize=14, fontweight='bold')

src.close()

#-- save figure to file
for i in range(2):
	for j in range(2):
		ax[i,j].get_xaxis().set_visible(False)
		ax[i,j].get_yaxis().set_visible(False)
		# ax[i,j].set_aspect('equal')
		if i == 1 and j == 1:
			ax[i,j].set_xlim([-1.437e6,-1.4239e6])
			ax[i,j].set_ylim([-977e3,-957.5e3])
		else:
			ax[i,j].set_xlim([-1.5e6,-1.4e6])
			ax[i,j].set_ylim([-10.6e5,-9e5])
plt.tight_layout()
plt.savefig(os.path.join(base_dir,'geocoded_v1','stitched.dir',\
	'atrous_32init_drop0.2_customLossR727.dir','Pipeline_Figure.pdf'),format='PDF')
plt.close(fig)