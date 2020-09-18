u"""
make_inset_maps.py
Yara Mohajerani
"""
import os
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
import geopandas as gpd

#-- Directory setup
gdrive =  os.path.join(os.path.expanduser('~'),'Google Drive File Stream',\
	'Shared drives','GROUNDING_LINE_TEAM_DRIVE','ML_Yara')
base_dir = os.path.expanduser('~/GL_learning_data/')

#-- file name
# fname = 'gl_069_180808-180814-180820-180826_012170-023241-012345-023416_T110454_T110455'
fname = 'gl_069_181124-181130-181206-181212_013745-024816-013920-024991_T110456_T110456'

#-- also read the corresponding shapefile
shpfile = os.path.join(base_dir,'geocoded_v1','stitched.dir','atrous_32init_drop0.2_customLossR727.dir',\
                       'shapefiles.dir','%s_6.0km.shp'%fname)
gdf = gpd.read_file(shpfile)

x,y = gdf['geometry'][4].coords.xy
x,y = np.array(x),np.array(y)
x1 = x.min() - 2e5
x2 = x.max() + 2e5
y1 = y.min() - 2e5
y2 = y.max() + 2e5

fig= plt.figure(1,figsize=(5,5))
ax = fig.add_subplot(111)

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
world = world.to_crs(gdf.crs)
world.plot(ax=ax, color='lightgray', edgecolor='lightgray')
ax.set_xlim([-2700000,2700000])
ax.set_ylim([-2400000,2400000])
gdf.plot(ax=ax,color='red', edgecolor='red')
ax.plot([x1,x1,x2,x2,x1],[y1,y2,y2,y1,y1],color='black',linewidth=3)
ax.set_aspect('equal')
ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(base_dir,'antarctic_inset_%s.pdf'%fname),format='PDF')
plt.close()
