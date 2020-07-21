u"""
make_slurm.py
Yara Mohajerani

Make slurm scripts for GP for large data directories.
Calls run_prediction.py
"""
import os
import sys
import getopt

#-- main function
def main():
	#-- Read the system arguments listed after the program
	long_options=['DATA_DIR=','SLURM_DIR=','CODE_DIR=','NUM=']
	optlist,arglist = getopt.getopt(sys.argv[1:],'D:S:C:N:',long_options)

	#-- Set default settings
	ddir = '/DFS-L/DATA/gl_ml/SENTINEL1_2018/Track007'
	slurm_dir = '/DFS-L/DATA/gl_ml/slurm.dir' 
	code_dir = '/DFS-L/DATA/isabella/ymohajer/GL_learning'
	num = 1500
	for opt, arg in optlist:
		if opt in ("-D","--DATA_DIR"):
			ddir = os.path.expanduser(arg)
		elif opt in ("-S","--SLURM_DIR"):
			slurm_dir = os.path.expanduser(arg)
		elif opt in ("-C","--CODE_DIR"):
			code_dir = os.path.expanduser(arg)
		elif opt in ("-N","--NUM"):
			num = int(arg)

	#-- Get list of images
	fileList = os.listdir(ddir)
	file_list = sorted([f for f in fileList if ( (f.endswith('DIR00.tif') or f.endswith('DIR11.tif')) and f.startswith('coco') )])
	# file_list = sorted([f for f in fileList if (f.endswith('.tif') and f.startswith('coco'))])
	N = len(file_list)
	print(N)

	#-- open list of all jobs to run
	list_fid = open(os.path.join(slurm_dir,'job_list_%s.sh'%os.path.basename(ddir)),'w')
	#-- make slurm job for every 'num' files
	cc = 0
	while (cc < N):
		outfile = os.path.join(slurm_dir,'%s_%i.sh'%(os.path.basename(ddir),cc))
		fid = open(outfile,'w')
		fid.write("#!/bin/bash\n")
		fid.write("#SBATCH -N1\n")
		fid.write("#SBATCH -n1\n")
		fid.write("#SBATCH --mem=20G\n")
		fid.write("#SBATCH -t0-02:00:00\n")
		fid.write("#SBATCH -p sib2.9,nes2.8,has2.5,brd2.4,ilg2.3,m-c2.2,m-c1.9,m2090\n")
		fid.write("#SBATCH --job-name=%s_%i\n"%(os.path.basename(ddir),cc))
		fid.write("#SBATCH --mail-user=ymohajer@uci.edu\n")
		fid.write("#SBATCH --mail-type=FAIL\n\n")

		fid.write('source ~/miniconda3/bin/activate gl_env\n')
		fid.write('python %s --DIR=%s --NUM=%i --START=%i --MODEL_DIR=%s\n'%\
			(os.path.join(code_dir,'run_prediction.py'),ddir,num,cc,code_dir))
		fid.close()

		#-- add job to list 
		list_fid.write('nohup sbatch %s\n'%outfile)

		cc += num
		
	list_fid.close()

#-- run main program
if __name__ == '__main__':
	main()