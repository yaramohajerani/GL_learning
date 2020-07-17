#!/usr/bin/env python
u"""
stitch_tile.py

Stitch tiles together before postprcessing.
"""
import os
import sys
import getopt
import numpy as np
from osgeo import gdal,osr
# import imageio
import rasterio

#-- directory setup
gdrive = os.path.join(os.path.expanduser('~'),'Google Drive File Stream',
	'Shared drives','GROUNDING_LINE_TEAM_DRIVE','ML_Yara','geocoded_v1')
gdrive_out = output_dir = os.path.join(os.path.expanduser('~'),'Google Drive File Stream',
	'My Drive','GL_Learning')
outdir = os.path.expanduser('~/GL_learning_data/geocoded_v1')

#-- main function
def main():
	#-- Read the system arguments listed after the program
	long_options=['DIR=','KERNEL=','noFLAG']
	optlist,arglist = getopt.getopt(sys.argv[1:],'D:K:F',long_options)

	#-- Set default settings
	ddir = '/Volumes/GoogleDrive/Shared drives/GROUNDING_LINE_TEAM_DRIVE/ML_Yara/S1_Pope-Smith-Kohler/UNUSED/pred_PSK-UNUSED_with_null/atrous_32init_drop0.2_customLossR727.dir'
	flag_gaussian_weight = True
	sigma_kernel = 0.05
	for opt, arg in optlist:
		if opt in ("-D","--DIR"):
			ddir = arg
		elif opt in ("-K","--KERNEL"):
			sigma_kernel = float(arg)
		elif opt in ("-F","--noFLAG"):
			flag_gaussian_weight = False
			print('Not using Gaussian kernel.')

	#-- Get list of geotiff label files
	fileList = os.listdir(ddir)
	pred_list = [os.path.join(ddir,f) for f in fileList if (f.endswith('.tif') and f.startswith('pred'))]
	
	#-- output directory
	path_stitched = os.path.join(ddir,'stitched.dir')
	#-- make directories if they don't exist
	if not os.path.exists(path_stitched):
		os.mkdir(path_stitched)

	#-- read first file to get dimensions
	raster = rasterio.open(pred_list[0],'r')
	nx_tile = raster.width
	ny_tile = raster.height
	#-- get transformation matrix
	trans = raster.transform
	out_crs = raster.crs.to_epsg()
	raster.close()

	#-- buid the kernel
	if flag_gaussian_weight:
		gx = np.arange(nx_tile)
		gx = (gx-gx[-1]/2.0)/(nx_tile/2)
		gy = np.arange(ny_tile)
		gy = (gy-gy[-1]/2.0)/(ny_tile/2)
		gxx,gyy = np.meshgrid(gx,gy)
		kernel_weight = np.exp(-(gxx**2+gyy**2)/sigma_kernel)
		del(gx,gy,gxx,gyy)
	else:
		kernel_weight = np.ones((ny_tile,nx_tile))

	#-- make a dictionary of all tiles that belong together
	tdict = {}
	print('Identifying the tiles and source DInSAR names...')
	for tilename in pred_list:
		name_dinsar = os.path.basename(tilename).split('pred_')[1].split('_x')[0]
		if not name_dinsar in tdict.keys():
			tdict[name_dinsar] = [tilename]
		else:
			tdict[name_dinsar].append(tilename)
	#-- get list of all the scenes (keys of dict)
	list_dinsar_src = list(tdict.keys())
	print('Done!')

	for dinsar_to_stitch in list_dinsar_src:
		print(dinsar_to_stitch,'- # tiles:',len(tdict[dinsar_to_stitch]))
		list_tile_to_stitch = tdict[dinsar_to_stitch]
		numtiles = len(list_tile_to_stitch)
		list_x0 = np.zeros(numtiles,dtype=np.int32)
		list_y0 = np.zeros(numtiles,dtype=np.int32)
		for i,tile_to_stitch in enumerate(list_tile_to_stitch):
			list_x0[i]=int(tile_to_stitch.split('_x')[1].split('_y')[0])
			list_y0[i]=int(tile_to_stitch.split('_y')[1].split('_DIR')[0])
		
		#-- determine the output tile size
		nx_out=list_x0.max()+nx_tile
		ny_out=list_y0.max()+ny_tile

		#-- initialize sum of tiles and weights
		arr_sum=np.zeros((ny_out,nx_out))
		arr_weight=np.zeros((ny_out,nx_out))
		#-- initialize tile mask
		arr_mask = np.zeros((ny_out,nx_out),dtype=int)
		#-- loop through tiles and adding to larger scene array
		for i,tile_to_stitch in enumerate(list_tile_to_stitch):
			raster = rasterio.open(tile_to_stitch,'r')
			tile_in = raster.read(1).astype(float)

			arr_sum[list_y0[i]:list_y0[i]+ny_tile,list_x0[i]:list_x0[i]+nx_tile] += tile_in*kernel_weight
			arr_weight[list_y0[i]:list_y0[i]+ny_tile,list_x0[i]:list_x0[i]+nx_tile] += kernel_weight
		
			raster.close()

		#-- noramlize
		arr_out = arr_sum/arr_weight

		#-- nan values from division by 0 are set to 0 (no tile coverage)
		arr_out[np.isnan(arr_out)] = 0.0
		arr_out[arr_out < 0] = 0.0

		#-- get pixel size
		x1,y1 = rasterio.transform.xy(trans, 0, 0)
		x2,y2 = rasterio.transform.xy(trans, 0, 1)
		x3,y3 = rasterio.transform.xy(trans, 1, 0)
		dx = np.abs(x2 - x1)
		dy = np.abs(y3 - y1)
		#-- Now find the coordinates of the upper left corner of scene based on total size
		#-- note the x1,y1 refers to position list_x0[i],list_y0[i]
		x_orig = x1 - (dx*list_x0[i])
		y_orig = y1 + (dy*list_y0[i])

		#-- get transformation for output
		#-- output as geotiff
		driver = gdal.GetDriverByName("GTiff")
		#-- set up the dataset with compression options (1 is for band 1)
		OPTS = ['COMPRESS=LZW']
		ds = driver.Create(os.path.join(path_stitched,'%s.tif'%dinsar_to_stitch), \
			int(nx_out), int(ny_out), 1, gdal.GDT_Float32, OPTS)
		#-- top left x, w-e pixel resolution, rotation
		#-- top left y, rotation, n-s pixel resolution
		ds.SetGeoTransform([x_orig, dx, 0, y_orig, 0, -dy])
		#-- set the reference info
		srs = osr.SpatialReference()
		srs.ImportFromEPSG(out_crs)
		#-- export
		ds.SetProjection( srs.ExportToWkt() )
		#-- write to geotiff array
		ds.GetRasterBand(1).WriteArray(arr_out)
		ds.FlushCache()
		ds = None
		
		#-- also save image for reference
		# outfile = os.path.join(path_stitched,'%s.png'%dinsar_to_stitch)
		# imageio.imsave(outfile,(arr_out*255).astype(np.ubyte))
		
#-- run main program
if __name__ == '__main__':
	main()

