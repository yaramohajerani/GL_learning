#!/usr/bin/env python
u"""
combined_dates.py
Yara Mohajerani

Combine GL shapefiles for the same dates into one without the
training labels
"""
import os
import re
import sys
import getopt
import numpy as np
import fiona
from shapely.geometry import LineString, mapping

base_dir = os.path.expanduser('~/GL_learning_data/geocoded_v1')

#-- make list of properties
props = ['Track','D1','D2','D3','D4','O1','O2','O3','O4','T1','T2','X','Y']

#-- get meta data from file name
def get_info(fname):
	a = re.split('_|-',fname)
	dic = {}
	for i,p in enumerate(props):
		dic[p] = a[i+2]
	return dic

#-- main function
def main():
	#-- Read the system arguments listed after the program
	long_options=['DIR=','FILTER=']
	optlist,arglist = getopt.getopt(sys.argv[1:],'D:F:',long_options)

	#-- Set default settings
	subdir = 'atrous_32init_drop0.2_customLossR727.dir'
	filter = 0.
	flt_str = ''
	for opt, arg in optlist:
		if opt in ("-D","--DIR"):
			subdir = arg
		elif opt in ("-F","--FILTER"):
			if arg not in ['NONE','none','None','N','n',0]:
				filter = float(arg)
				flt_str = '_%.1fkm'%(filter/1000)

	#-- make full directory paths
	indir = {}
	for t in ['Train','Test']:
		indir[t] = os.path.join(base_dir,'%s_predictions.dir'%t,subdir,'shapefiles.dir')
	out_subdir = os.path.join(base_dir,'CombinedDates.dir',subdir)
	#-- make directories if they don't exist
	if not os.path.exists(out_subdir):
		os.mkdir(out_subdir)
	outdir = os.path.join(out_subdir,'combined_shapefiles%s.dir'%flt_str)
	#-- make directories if they don't exist
	if not os.path.exists(outdir):
		os.mkdir(outdir)
	
	#-- get list of files
	flist = {}
	for t in ['Train','Test']:
		fileList = os.listdir(indir[t])
		flist[t] = [f for f in fileList if (f.endswith('.shp') and f.startswith('post'))]

	#-- Match the file labels
	#-- everything except the last part (x,y,DIR) has to match
	dates = {}
	for t in ['Train','Test']:
		dates[t] = [None]*len(flist[t])
		for i,f in enumerate(flist[t]):
			dates[t][i] = re.split('gl_|_x',f)[1]
		dates[t] = np.array(dates[t],dtype='str')

	#-- get number of unique dates
	ud = np.unique(np.concatenate((dates['Train'],dates['Test'])))
	print(len(ud))
	
	#-- schema for writing files
	out_schema = {'geometry': 'LineString','properties': {'ID':'str',\
		'Track':'int','D1':'int','D2':'int','D3':'int','D4':'int',\
		'O1':'int','O2':'int','O3':'int','O4':'int','T1':'str',\
		'T2':'str','X':'str','Y':'str','Type':'str'}}

	#-- go through dates and read and combine all corresponding files
	for u in ud:
		for t in ['Train','Test']:
			#-- get indices of all matching dates
			ii, = np.nonzero(dates[t]==u)
			for i in ii:
				#-- parse aquisition info from file name
				info = get_info(flist[t][i])
				#-- read shapefile flist[i]
				src = fiona.open(os.path.join(indir[t],flist[t][i]),'r')
				meta = src.meta
				#-- update schema
				meta['schema'] = out_schema
				#-- counters
				c1,c2,c3 = 0,0,0
				#-- open output file for centerlines and errors
				dst1 =  fiona.open(os.path.join(outdir,'centerline_%s%s.shp'%(u,flt_str)),'w', **meta)
				dst2 = fiona.open(os.path.join(outdir,'errors_%s%s.shp'%(u,flt_str)),'w', **meta)
				if filter == 0:
					dst3 = fiona.open(os.path.join(outdir,'handdrawn_%s.shp'%u),'w', **meta)
				for j in range(len(src)):
					g = next(src)
					#-- add filter
					if (len(g['geometry']['coordinates']) > 1) and \
						(LineString(g['geometry']['coordinates']).length > filter):
						#-- add properties
						g['properties']['Type'] = t
						for p in props:
							g['properties'][p] = info[p]
						if g['properties']['ID'].startswith('cntr'):
							g['properties']['ID'] = '%s_%i'%(info['Track'],c1)
							#-- write center line to file
							dst1.write(g)
							c1 += 1
						elif (not g['properties']['ID'].startswith('lbl')):
							g['properties']['ID'] = '%s_%i'%(info['Track'],c2)
							#-- write center line to file
							dst2.write(g)
							c2 += 1
						else:
							if filter == 0:
								g['properties']['ID'] = '%s_%i'%(info['Track'],c3)
								#-- write center line to file
								dst3.write(g)
								c3 += 1
				if filter == 0:
					dst3.close()
				dst2.close()
				dst1.close()
				src.close()

#-- run main program
if __name__ == '__main__':
	main()

