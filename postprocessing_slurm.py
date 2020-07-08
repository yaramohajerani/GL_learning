#!/usr/bin/env python
u"""
postpocessing_slurm.py

Make slurm scripts for running postprocessing as 
separate job for each file on greenplanet
"""

import os

#-- directory setup
ddir = os.path.expanduser('~/GL_learning_data/geocoded_v1')
subdir = 'atrous_32init_drop0.2_customLossR727.dir'
outdir = os.path.join(ddir,'stitched.dir',subdir,'slurm.dir')
if (not os.path.isdir(outdir)):
	os.mkdir(outdir)

#-- Get list of files
pred_dir = os.path.join(ddir,'stitched.dir',subdir)
fileList = os.listdir(pred_dir)
pred_list = [f for f in fileList if (f.endswith('.tif') and ('mask' not in f))]

#-- make master list for running all jobs
lfid = open(os.path.join(outdir,'job_list.sh'),'w')
#-- loop through input files and write a slurm script for each
for i,f in enumerate(pred_list):
	outfile = os.path.join(outdir,f.replace('.tif','.sh'))
	fid = open(outfile,'w')
	fid.write("#!/bin/bash\n")
	fid.write("#SBATCH -N1\n")
	fid.write("#SBATCH -n1\n")
	fid.write("#SBATCH --mem=6G\n")
	fid.write("#SBATCH -t0-06:00:00\n")
	fid.write("#SBATCH -p sib2.9\n")
	fid.write("#SBATCH --job-name=gl_%i\n"%i)
	fid.write("#SBATCH --mail-user=ymohajer@uci.edu\n")
	fid.write("#SBATCH --mail-type=FAIL\n\n")

	fid.write('module load anaconda/2/5.1.0\n')
	fid.write('source activate GDAL\n')
	fid.write('python /DFS-L/DATA/isabella/ymohajer/GL_learning/convert_shapefile_centerline.py --INPUT=%s --FILTER=6000\n'%\
		(os.path.join('/DFS-L/DATA/isabella/ymohajer/GL_learning_data/geocoded_v1/stitched.dir/atrous_32init_drop0.2_customLossR727.dir',f)))
	fid.close()
	lfid.write('nohup sbatch /DFS-L/DATA/isabella/ymohajer/GL_learning_data/geocoded_v1/stitched.dir/%s/slurm.dir/%s\n'\
		%(subdir,f.replace('.tif','.sh')))
lfid.close()
