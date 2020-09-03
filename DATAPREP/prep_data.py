#!/usr/bin/env python3

import grup
import sys
import glob
import subprocess
import numpy as np
import datetime
import os
import numpy as np

dict_region_epsg={
    'antarctica':3031,
    'greenland':3413
}

class processor:
    def __init__(self,sensor='sentinel1',region='antarctica'):
        self._sensor=sensor
        self._region=region
        self.epsg=dict_region_epsg[region]
        
        self.path_shp=None
        self.path_coco_root=None
        self.path_tile_out_null_incl=None
        self.path_tile_out_null_excl=None #NOTE: Set either of them as 'None' when not in  not want of these.
        
        self.nx_tile=512
        self.ny_tile=512
        self.offset_x=0
        self.ofset_y=0
        self.dirx=True #True: From left to right, False: From right to left
        self.diry=True #True: From upper to bottom, False: From bottom to upper
        self.overwrite=True
        self.dn_burn=255

        self.format_tile_coco='{PATH}/coco_{SHPNAME}_x{X:03d)}_y{Y:03d}_DIR{DIRX}{DIRY}.tif'
        self.format_tile_delineation='{PATH}/delineation_{SHPNAME}_x{X:03d)}_y{Y:03d}_DIR{DIRX}{DIRY}.tif'

    @property
    def region(self):
         return self._region
    @region.setter
    def region(self,str_region):
        try:
            self.epsg=dict_region_epsg[str_region]
            self._region=str_region
        except:
            print('ERROR: EPSG for \'{}\' was not defined.'.format(str_region))

    

def split_raster(grupraster_in,path_out,prefix_outfile,nx_tile=512,ny_tile=512,dirx=True,diry=True,offset_x=0,offset_y=0,epsg_tile=3031,dryrun=False):
    format_filename_tile='{}/{}_x{:04d}_y{:04d}_DIR{:d}{:d}.tif'

    coco_tile_out=grup.raster()
    coco_tile_out.nx=nx_tile
    coco_tile_out.ny=ny_tile
    coco_tile_out.numbands=grupraster_in.numbands
    

    coco_tile_out.set_proj_by_EPSG(epsg_tile)
    coco_tile_out.option=['COMPRESS=LZW']

    #determine the grid to cut
    grid_x=np.arange(offset_x,grupraster_in.nx,nx_tile)
    if not dirx:
        grid_x=np.flipud(np.arange(grupraster_in.nx-offset_x,0,-nx_tile))
    
    grid_y=np.arange(offset_y,grupraster_in.ny,ny_tile)
    if not diry:
        grid_y=np.flipud(np.arange(grupraster_in.ny-offset_y,0,-ny_tile))

    for i in range(len(grid_y)-1):
        for j in range(len(grid_x)-1):
            print('x:{}-{}, y:{}-{}'.format(grid_x[j],grid_x[j+1],grid_y[i],grid_y[i+1]))
            filename_tile=format_filename_tile.format(path_out,
                                                      prefix_outfile,
                                                      grid_x[j],grid_y[i],
                                                      dirx,diry)


            if coco_tile_out.numbands==1:
                coco_tile_out.z=grupraster_in.z[grid_y[i]:grid_y[i+1],grid_x[j]:grid_x[j+1]]
            else:
                coco_tile_out.z=grupraster_in.z[:,grid_y[i]:grid_y[i+1],grid_x[j]:grid_x[j+1]]

            gtf_list=[grupraster_in.GeoTransform[0]+grid_x[j]*grupraster_in.GeoTransform[1],
                      grupraster_in.GeoTransform[1],
                      0,
                      grupraster_in.GeoTransform[3]+grid_y[i]*grupraster_in.GeoTransform[5],
                      0,
                      grupraster_in.GeoTransform[5]]
                        
            coco_tile_out.GeoTransform=tuple(gtf_list)
            
            if dryrun:
                print('Writing:',filename_tile)
            else:
                coco_tile_out.write(filename_tile)
        
            #determine the tile file name
            

def rasterize_shp(filename_shp,filename_out,nx,ny,GeoTransform_in,dn_burn=255):
    form_command_rasterize='gdal_rasterize -burn {BURN} -ot Byte -tr {RX} {RY} -te {XMIN} {YMIN} {XMAX} {YMAX} -co compress=LZW {SHP_IN} {TIFF_OUT}'
    x0=GeoTransform_in[0]-GeoTransform_in[1]/2
    x1=x0+GeoTransform_in[1]*nx
    y1=GeoTransform_in[3]-GeoTransform_in[5]/2
    y0=y1+GeoTransform_in[5]*ny
    command_rasterize=form_command_rasterize.format(BURN=dn_burn,
                                                    RX=GeoTransform_in[1],
                                                    RY=-GeoTransform_in[5],
                                                    XMIN=x0,
                                                    YMIN=y0,
                                                    XMAX=x1,
                                                    YMAX=y1,
                                                    SHP_IN=filename_shp,
                                                    TIFF_OUT=filename_out)

    subprocess.call(command_rasterize,shell=True)

        
class sentinel1:
    def __init__(self):
        self.form_path_pair='{PARENT}/{YEAR}/{DT}d/Track{ORBIT}/Track{ID0}-{ID1}_T{TIME0}'
        self.form_coco='coco{ID0}-{ID1}-{ID2}-{ID3}_T{TIME1}.flat.topo_off.deramp.geo.psfilt'
        self.path_parent='/u/oates-r0/eric/SENTINEL1' #default root directory of S1 processing data in UCI

    #example pair of coco and gltif
    #                            gl_164_180919-181001-181001-181013_012790-012965-012965-013140_T233047_T233048.tif
    #.../SENTINEL1/2018/12d/Track164/Track012790-012965_T233047/coco012790-012965-012965-013140_T233048.flat.topo_off.deramp.geo.psfilt
    
    #convert coco name into gl.tif name
    def coco2gl(self,str_coco):
        #TODO: Complete implementing this
        print(str_coco,'IMPLEMENT ME!!!')
        pass

    #convert gl.tif name into coco name
    def gl2coco(self,str_gl,path_root=None):

        if path_root==None:
            path_root=self.path_parent
        
        #parse str_gl
        #gl_164_180919-181001-181001-181013_012790-012965-012965-013140_T233047_T233048.tif
        #00 111 222222 222222 222222 222222 333333 333333 333333 333333 4444444 5555555
        str_orbit=str_gl.split('_')[1]
        seg_dates=str_gl.split('_')[2].split('-')
        seg_id=str_gl.split('_')[3].split('-')
        str_time0=str_gl.split('_')[4].replace('T','')
        str_time1=str_gl.split('_')[5].replace('T','').replace('.tif','')

        #detect the year string
        str_year='2000'
        for datestr in seg_dates:
            if int(datestr[0:3])>int(str_year[3:]):
                str_year='20{}'.format(datestr[0:2])
    
        #detect dt portion
        t0=datetime.datetime.strptime(seg_dates[0],'%y%m%d')
        t1=datetime.datetime.strptime(seg_dates[1],'%y%m%d')
        dt=abs((t1-t0).days)

        str_path_pair=self.form_path_pair.format(PARENT=self.path_parent,
                                                 YEAR=str_year,
                                                 DT=dt,
                                                 ORBIT=str_orbit,
                                                 ID0=seg_id[0],
                                                 ID1=seg_id[1],
                                                 TIME0=str_time0)

        str_coco=self.form_coco.format(ID0=seg_id[0],
                                       ID1=seg_id[1],
                                       ID2=seg_id[2],
                                       ID3=seg_id[3],
                                       TIME1=str_time1)

        return '{}/{}'.format(str_path_pair,str_coco)

    def load_coco(self,filename_coco,backscatter=False):
        #NOTE: Turn backscatter True when want to include backscatter
        #TODO: Implement the case whehn backscatter=True

        with open('{}/DEM_gc_par'.format(os.path.dirname(filename_coco)),'r') as fin:
            lines_in=fin.readlines()
        npix=0
        nrec=0
        for line in lines_in:
            if 'width:' in line:
                npix=int(line.split()[1])
            if 'nlines:' in line:
                nrec=int(line.split()[1])

            if npix>0 and nrec>0:
                break
        
        print('NPIX={}, NREC={}'.format(npix,nrec))

        raster_out=grup.raster()
        raster_out.nx=npix
        raster_out.ny=nrec
        raster_out.numbands=1
        raster_out.z=np.fromfile(filename_coco,dtype=np.complex64).byteswap().reshape((nrec,npix))

        return raster_out



