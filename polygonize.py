u"""
convert_shapefile_thinning.py
Yara Mohajerani (Last update 07/2020)

Read output predictions and convert to shapefile lines
"""
import os
import sys
import rasterio
import numpy as np
import getopt
import shapefile
from skimage.measure import find_contours
from shapely.geometry import Polygon,LineString,Point

in_base = os.path.expanduser('~')

#-- main function
def main():
	#-- Read the system arguments listed after the program
	long_options=['DIR=','FILTER=','OUT_BASE=','noMASK']
	optlist,arglist = getopt.getopt(sys.argv[1:],'D:F:O:M',long_options)

	#-- Set default settings
	subdir = os.path.join('GL_learning_data','geocoded_v1'\
		,'stitched.dir','atrous_32init_drop0.2_customLossR727.dir')
	FILTER = 0.
	flt_str = ''
	out_base = '/DFS-L/DATA/isabella/ymohajer/'
	make_mask = True
	for opt, arg in optlist:
		if opt in ("-D","--DIR"):
			subdir = arg
		elif opt in ("-F","--FILTER"):
			if arg not in ['NONE','none','None','N','n',0]:
				FILTER = float(arg)
				flt_str = '_%.1fkm'%(FILTER/1000)
		elif opt in ("O","--OUT_BASE"):
			out_base = os.path.expanduser(arg)
		elif opt in ("M","--noMASK"):
			make_mask = False
	
	indir = os.path.join(in_base,subdir)

	#-- Get list of files
	fileList = os.listdir(indir)
	pred_list = [f for f in fileList if (f.endswith('.tif') and ('mask' not in f))]
	#-- LOCAL output directory
	local_output_dir = os.path.join(indir,'shapefiles.dir')
	#-- slurm directory
	slurm_dir = os.path.join(indir,'slurm.dir')
	#-- make directories if they don't exist
	if not os.path.exists(local_output_dir):
		os.mkdir(local_output_dir)
	
	print('# of files: ', len(pred_list))
	
	#-- threshold for getting contours and centerlines
	eps = 0.2

	#-- open file for list of polygons to run through centerline routine
	list_fid = open(os.path.join(slurm_dir,'total_job_list.sh'),'w')

	#-- loop through prediction files
	#-- get contours and save each as a line in shapefile format
	for pcount,f in enumerate(pred_list):
		#-- open job list for this file
		sub_list_fid = open(os.path.join(slurm_dir,f.replace('.tif','.sh')),'w')
		#-- read file
		raster = rasterio.open(os.path.join(indir,f),'r')
		im = raster.read(1)
		#-- get transformation matrix
		trans = raster.transform

		if make_mask:
			#-- also read the corresponding mask file
			mask_file = os.path.join(indir,f.replace('.tif','_mask.tif'))
			mask_raster = rasterio.open(mask_file,'r')
			mask = mask_raster.read(1)
			mask_raster.close()

		#-- get contours of prediction
		#-- close contour ends to make polygons
		im[np.nonzero(im[:,0] > eps),0] = eps
		im[np.nonzero(im[:,-1] > eps),-1] = eps
		im[0,np.nonzero(im[0,:] > eps)] = eps
		im[-1,np.nonzero(im[-1,:] > eps)] = eps
		contours = find_contours(im, eps)
		#-- make contours into closed polyons to find pinning points
		#-- also apply noise filter and append to noise list
		x = {}
		y = {}
		noise = []
		pols = [None]*len(contours)
		pol_type = [None]*len(contours)
		for n,contour in enumerate(contours):
			#-- convert to coordinates
			x[n],y[n] = rasterio.transform.xy(trans, contour[:,0], contour[:,1])

			pols[n] = Polygon(zip(x[n],y[n]))
			if make_mask:
				#-- get elements of mask the contour is on
				submask = mask[np.round(contour[:, 0]).astype('int'), np.round(contour[:, 1]).astype('int')]
				#-- if more than half of the elements are from test tile, count contour as test type
				if np.count_nonzero(submask) > submask.size/2.:
					pol_type[n] = 'Test'
				else:
					pol_type[n] = 'Train'
			else:
				pol_type[n] = 'Test'

		#-- loop through remaining polygons and determine which ones are 
		#-- pinning points based on the width and length of the bounding box
		pin_list = []
		box_ll = [None]*len(contours)
		box_ww = [None]*len(contours)
		for n in range(len(pols)):
			box_ll[n] = pols[n].length
			box_ww[n] = pols[n].area/box_ll[n]
			#-- if the with is larger than 1/25 of the length, it's a pinning point
			if box_ww[n] > box_ll[n]/25:
				pin_list.append(n)

		#-- Loop through all the polygons and take any overlapping areas out
		#-- of the enclosing polygon and ignore the inside polygon
		ignore_list = []
		for i in range(len(pols)):
			for j in range(len(pols)):
				if (i != j) and pols[i].contains(pols[j]):
					# pols[i] = pols[i].difference(pols[j])
					if (i in pin_list) and (j in pin_list):
						#-- if it's a pinning point, ignore outer loop
						ignore_list.append(i)
					else:
						#-- if not, add inner loop to ignore list
						ignore_list.append(j)

		#-- get rid of duplicates in ignore list
		ignore_list = list(set(ignore_list))

		#-- loop through and apply noise filter
		for n in range(len(contours)):
			#-- apply filter
			if (n not in ignore_list) and (len(x[n]) < 2 or LineString(zip(x[n],y[n])).length <= FILTER):
				noise.append(n)

		#-- find overlap between ignore list nad noise list
		if len(list(set(noise) & set(ignore_list))) != 0:
			sys.exit('Overlap not empty: ', list(set(noise) & set(ignore_list)))

		#-- initialize list of contour linestrings
		er = [None]*len(contours)
		n = 0  # total center line counter
		er_type = [None]*len(er)
		er_class = [None]*len(er)
		er_lbl = [None]*len(er)
		count = 1 #-- file count
		pc = 1 # pinning point counter
		lc = 1 # line counter
		#-- loop through polygons and save to separate files
		for idx,p in enumerate(pols):
			er[idx] = [list(a) for a in zip(x[idx],y[idx])]
			er_type[idx] = pol_type[idx]
			if idx in noise:
				er_class[idx] = 'Noise'				
			elif idx in ignore_list:
				er_class[idx] = 'Ignored Contour'
			else:
				if idx in pin_list:
					er_class[idx] = 'Pinning Contour'
					er_lbl[idx] = 'pin_err%i'%pc
					pc += 1 #- incremenet pinning point counter
				else:
					er_class[idx] = 'GL Uncertainty'
					#-- set label
					er_lbl[idx] = 'err%i'%lc
					lc += 1 #- incremenet line counter
				
				#-- write individual polygon to file
				out_name = f.replace('.tif','%s_ERR_%i'%(flt_str,count))
		
				er_file = os.path.join(local_output_dir,'%s.shp'%out_name)
				w = shapefile.Writer(er_file)
				w.field('ID', 'C')
				w.field('Type','C')
				w.field('Class','C')
				w.field('Length','C')
				w.field('Width','C')
				w.line([er[idx]])
				w.record(er_lbl[idx] , er_type[idx], er_class[idx], box_ll[idx], box_ww[idx])
				w.close()
				# create the .prj file
				prj = open(er_file.replace('.shp','.prj'), "w")
				prj.write(raster.crs.to_wkt())
				prj.close()
				
				#-- write corresponding slurm file
				#-- calculate run time
				run_time = int(p.length/400)+10

				outfile = os.path.join(slurm_dir,'%s.sh'%out_name)
				fid = open(outfile,'w')
				fid.write("#!/bin/bash\n")
				fid.write("#SBATCH -N1\n")
				fid.write("#SBATCH -n1\n")
				fid.write("#SBATCH --mem=10G\n")
				fid.write("#SBATCH -t %i\n"%run_time)
				fid.write("#SBATCH -p sib2.9,nes2.8,has2.5,brd2.4,ilg2.3,m-c2.2,m-c1.9,m2090\n")
				fid.write("#SBATCH --job-name=gl_%i_%i_%i\n"%(pcount,idx,count))
				fid.write("#SBATCH --mail-user=ymohajer@uci.edu\n")
				fid.write("#SBATCH --mail-type=FAIL\n\n")

				fid.write('source ~/miniconda3/bin/activate gl_env\n')
				fid.write('python %s %s\n'%\
					(os.path.join(out_base,'GL_learning','run_centerline.py'),\
					os.path.join(out_base,subdir,'shapefiles.dir','%s.shp'%out_name)))
				fid.close()

				#-- add job to list 
				sub_list_fid.write('nohup sbatch %s\n'%os.path.join(out_base,subdir,'slurm.dir','%s.sh'%out_name))

				count += 1
		
		sub_list_fid.close()
		#-- add sub list fid to total job list
		list_fid.write('sh %s\n'%os.path.join(out_base,subdir,'slurm.dir',f.replace('.tif','.sh')))

		
		#-- save all contours to file
		er_file = os.path.join(local_output_dir,f.replace('.tif','%s_ERR.shp'%flt_str))
		w = shapefile.Writer(er_file)
		w.field('ID', 'C')
		w.field('Type','C')
		w.field('Class','C')
		w.field('Length','C')
		w.field('Width','C')
		#-- loop over contours and write them
		for n in range(len(er)):
			w.line([er[n]])
			w.record(er_lbl[n] , er_type[n], er_class[n], box_ll[n], box_ww[n])
		w.close()
		# create the .prj file
		prj = open(er_file.replace('.shp','.prj'), "w")
		prj.write(raster.crs.to_wkt())
		prj.close()
		
		#-- close input file
		raster.close()

	#-- close master list fid	
	list_fid.close()
	
#-- run main program
if __name__ == '__main__':
	main()
