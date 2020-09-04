u"""
combine_centerline_slurms.py

Combine centerline jobs for each file (instead of running
individual files are separete jobs)
"""
import os
import sys

#-- main function
def main():
	#-- Read the system arguments listed after the program
	if len(sys.argv) == 1:
		sys.exit('No input file given')
	else:
		inputs = sys.argv[1:]

	for infile in inputs:
		#-- get list of slurm files
		fid1 = open(os.path.expanduser(infile), 'r') 
		file_list = fid1.readlines() 
		#-- go through lines and read each slurm job
		for f in file_list:
			print("f: ",f)
			fname = f.split(' ')[1].replace('\n','')
			#-- initialize output file
			outname = os.path.join(os.path.basename(f),fname.replace('.sh','_combined.sh'))
			outfid = open(outname,'w')
			#-- now read the individual files
			fid2 = open(fname,'r')
			job_list = fid2.readlines()
			time = 0
			for job in job_list:
				print("job: ",job)
				subname = job.split(' ')[2].replace('\n','')
				fid3 = open(subname,'r')
				commands = fid3.readlines()
				for c in commands:
					if '#SBATCH -t' in c:
						time += int(c.split(' ')[2])
				fid3.close()
			fid2.close()
			#-- make output file
			for c in commands:
				if '#SBATCH -t' in c:
					outfid.write('#SBATCH -t %i\n'%time)
				elif not c.startswith('python'):
					outfid.write(c)
			#-- now write python command
			outfid.write('python /DFS-L/DATA/GL_learning/run_centerline.py ')
			for job in job_list:
				subname = job.split(' ')[2].replace('\n','')
				outfid.write('%s '%os.path.join(os.path.basename(job),subname))
			outfid.close()
		fid1.close()

#-- run main program
if __name__ == '__main__':
	main()
	
